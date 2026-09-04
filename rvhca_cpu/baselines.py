"""Small hindsight selectors for the CPU contract probe.

These functions consume only rows that a receiver could have seen before the
current send time.  They do not train a controller and do not alter the
immutable ledger.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


def _pair_key(row: Mapping[str, Any]) -> Tuple[str, str, str, int]:
    return (
        str(row.get("sequence_id")),
        str(row.get("receiver")),
        str(row.get("source")),
        int(row.get("receiver_target_id")),
    )


def _horizon_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return _pair_key(row) + (round(float(row.get("horizon", 0.0)), 3),)


def _matured_rows_before(
    rows: Sequence[Mapping[str, Any]],
    send_time: float,
    send_time_strict: bool = True,
) -> Iterable[Mapping[str, Any]]:
    for row in rows:
        if not row.get("valid_mask"):
            continue
        row_send = float(row.get("send_time", 0.0))
        if send_time_strict and row_send >= send_time - 1e-9:
            continue
        observation_time = row.get("observation_time")
        if observation_time is None or float(observation_time) > send_time + 1e-9:
            continue
        if row.get("peer_realized_error") is None or row.get("ego_only_error") is None:
            continue
        yield row


def _summary(name: str, decisions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    usable = [row for row in decisions if row.get("selected_error") is not None]
    if not usable:
        return {
            "baseline": name,
            "rows": len(decisions),
            "usable_rows": 0,
            "peer_enabled_rate": None,
            "mean_selected_error": None,
            "mean_ego_error": None,
            "mean_peer_error": None,
            "mean_gain_vs_ego": None,
        }
    mean_selected = sum(float(row["selected_error"]) for row in usable) / len(usable)
    mean_ego = sum(float(row["ego_only_error"]) for row in usable) / len(usable)
    mean_peer = sum(float(row["peer_realized_error"]) for row in usable) / len(usable)
    return {
        "baseline": name,
        "rows": len(decisions),
        "usable_rows": len(usable),
        "peer_enabled_rate": sum(bool(row["peer_enabled"]) for row in usable) / len(usable),
        "mean_selected_error": mean_selected,
        "mean_ego_error": mean_ego,
        "mean_peer_error": mean_peer,
        "mean_gain_vs_ego": mean_ego - mean_selected,
    }


def _chronological_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("sequence_id")),
            str(row.get("receiver")),
            float(row.get("send_time", 0.0)),
            str(row.get("source")),
            int(row.get("receiver_target_id", -1)),
            float(row.get("horizon", 0.0)),
        ),
    )


def _decision_row(row: Mapping[str, Any], enabled: bool, reason: str) -> Dict[str, Any]:
    peer_error = row.get("peer_realized_error")
    ego_error = row.get("ego_only_error")
    selected = peer_error if enabled else ego_error
    return {
        "sequence_id": row.get("sequence_id"),
        "receiver": row.get("receiver"),
        "source": row.get("source"),
        "receiver_target_id": row.get("receiver_target_id"),
        "send_time": row.get("send_time"),
        "horizon": row.get("horizon"),
        "peer_enabled": bool(enabled),
        "decision_reason": reason,
        "selected_error": selected,
        "peer_realized_error": peer_error,
        "ego_only_error": ego_error,
    }


def recent_error(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ordered = _chronological_rows(rows)
    by_time: MutableMapping[float, List[Mapping[str, Any]]] = defaultdict(list)
    for row in ordered:
        by_time[float(row.get("send_time", 0.0))].append(row)
    matured = sorted(
        [row for row in ordered if row.get("valid_mask") and row.get("observation_time") is not None],
        key=lambda row: float(row["observation_time"]),
    )
    pointer = 0
    latest: Dict[Tuple[Any, ...], float] = {}
    decisions: List[Dict[str, Any]] = []
    for send_time in sorted(by_time):
        while pointer < len(matured) and float(matured[pointer]["observation_time"]) <= send_time + 1e-9:
            item = matured[pointer]
            if float(item.get("send_time", 0.0)) < send_time - 1e-9:
                latest[_horizon_key(item)] = float(item["peer_realized_error"]) - float(item["ego_only_error"])
            pointer += 1
        for row in by_time[send_time]:
            key = _horizon_key(row)
            if key in latest:
                enabled = latest[key] <= 0.0
                reason = "recent_diff<=0" if enabled else "recent_diff>0"
            else:
                enabled = False
                reason = "cold_start_fallback_ego"
            decisions.append(_decision_row(row, enabled, reason))
    return _summary("recent-error", decisions)


def ewma(rows: Sequence[Mapping[str, Any]], alpha: float = 0.3) -> Dict[str, Any]:
    ordered = _chronological_rows(rows)
    by_time: MutableMapping[float, List[Mapping[str, Any]]] = defaultdict(list)
    for row in ordered:
        by_time[float(row.get("send_time", 0.0))].append(row)
    matured = sorted(
        [row for row in ordered if row.get("valid_mask") and row.get("observation_time") is not None],
        key=lambda row: float(row["observation_time"]),
    )
    pointer = 0
    decisions: List[Dict[str, Any]] = []
    state: Dict[Tuple[Any, ...], float] = {}
    for send_time in sorted(by_time):
        while pointer < len(matured) and float(matured[pointer]["observation_time"]) <= send_time + 1e-9:
            item = matured[pointer]
            if float(item.get("send_time", 0.0)) < send_time - 1e-9:
                key = _horizon_key(item)
                diff = float(item["peer_realized_error"]) - float(item["ego_only_error"])
                state[key] = diff if key not in state else alpha * diff + (1.0 - alpha) * state[key]
            pointer += 1
        for row in by_time[send_time]:
            key = _horizon_key(row)
            if key not in state:
                enabled = False
                reason = "cold_start_fallback_ego"
            else:
                enabled = state[key] <= 0.0
                reason = "ewma<=0" if enabled else "ewma>0"
            decisions.append(_decision_row(row, enabled, reason))
    return _summary("EWMA", decisions)


def metadata_only(
    rows: Sequence[Mapping[str, Any]],
    min_confidence: float = 0.75,
    max_delay_s: float = 0.2,
) -> Dict[str, Any]:
    decisions: List[Dict[str, Any]] = []
    for row in _chronological_rows(rows):
        confidence = float(row.get("association_confidence") or 0.0)
        delay = float(row.get("arrival_time", 0.0)) - float(row.get("send_time", 0.0))
        enabled = confidence >= min_confidence and delay <= max_delay_s + 1e-9
        reason = "metadata_pass" if enabled else "metadata_fallback_ego"
        decisions.append(_decision_row(row, enabled, reason))
    return _summary("metadata-only", decisions)


def summarize_baselines(rows: Sequence[Mapping[str, Any]], alpha: float = 0.3) -> Dict[str, Any]:
    usable = [row for row in rows if row.get("target_scope") == "common"]
    return {
        "recent_error": recent_error(usable),
        "ewma": ewma(usable, alpha=alpha),
        "metadata_only": metadata_only(usable),
    }
