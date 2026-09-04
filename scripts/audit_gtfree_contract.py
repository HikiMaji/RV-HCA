#!/usr/bin/env python
"""Audit GT-free replay artifacts without loading raw labels.

This checker is deliberately independent of ``rvhca_cpu.evaluate``.  It only
reads the online artifacts and verifies required identity/time fields, unique
ledger keys, causal ordering, and the absence of GT-derived field names.  Raw
OPV2V YAML is never opened here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple


FORBIDDEN_KEYS = frozenset(
    {
        "object_id",
        "object_ids",
        "gt_object_id",
        "gt_object_ids",
        "matched_car_id",
        "gt_ids",
        "gt_trajs",
        "center_gt_trajs",
        "center_gt_trajs_src",
        "center_gt_trajs_mask",
        "obj_trajs_future_state",
        "obj_trajs_future_mask",
        "track_index_to_predict",
    }
)


def _scan_keys(value: Any, path: str, violations: List[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_KEYS:
                violations.append("%s.%s" % (path, key_text))
            _scan_keys(child, "%s.%s" % (path, key_text), violations)
    elif isinstance(value, (list, tuple)):
        # Forecast trajectories are large numeric tensors represented as
        # nested lists.  Only descend into list elements that can contain
        # mappings; iterating every coordinate makes a multi-scene audit
        # needlessly expensive without improving forbidden-key coverage.
        for index, child in enumerate(value):
            if isinstance(child, Mapping):
                _scan_keys(child, "%s[%d]" % (path, index), violations)
            elif isinstance(child, (list, tuple)) and child and isinstance(child[0], Mapping):
                _scan_keys(child, "%s[%d]" % (path, index), violations)


def _rows(path: Path) -> Iterable[Tuple[int, Mapping[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError("%s:%d is not an object" % (path, line_no))
                yield line_no, value


def _audit_jsonl(path: Path, kind: str) -> Dict[str, Any]:
    count = 0
    violations: List[str] = []
    missing: List[str] = []
    duplicate_keys = 0
    seen: Set[Tuple[Any, ...]] = set()
    required: Sequence[str]
    if kind == "tracks":
        required = ("sequence_id", "source", "frame_idx", "local_track_id", "center_world")
    elif kind == "association":
        required = ("sequence_id", "receiver", "source", "send_time", "target_scope")
    elif kind == "ledger":
        required = (
            "sequence_id",
            "receiver",
            "source",
            "receiver_target_id",
            "send_time",
            "horizon",
            "forecast",
        )
    elif kind == "mtr_manifest":
        required = ("sequence_id", "source", "source_track_id", "send_time", "history", "uses_gt")
    else:
        raise ValueError("unknown kind: %s" % kind)

    for line_no, row in _rows(path):
        count += 1
        _scan_keys(row, "%s:%d" % (path, line_no), violations)
        for key in required:
            if key not in row:
                missing.append("%s:%d missing %s" % (path, line_no, key))
        if kind == "ledger":
            key = (
                str(row.get("sequence_id")),
                str(row.get("receiver")),
                str(row.get("source")),
                str(row.get("receiver_target_id")),
                float(row.get("send_time", 0.0)),
                float(row.get("horizon", 0.0)),
            )
            if key in seen:
                duplicate_keys += 1
            seen.add(key)
            arrival = row.get("arrival_time")
            send = row.get("send_time")
            observation = row.get("observation_time")
            if arrival is not None and send is not None and float(arrival) + 1e-9 < float(send):
                violations.append("%s:%d arrival_before_send" % (path, line_no))
            if observation is not None and arrival is not None and float(observation) + 1e-9 < float(arrival):
                violations.append("%s:%d observation_before_arrival" % (path, line_no))
            if row.get("valid_mask") and (observation is None or row.get("realized_state") is None):
                violations.append("%s:%d valid_without_observation" % (path, line_no))
        if kind == "mtr_manifest" and bool(row.get("uses_gt")):
            violations.append("%s:%d uses_gt_true" % (path, line_no))
    return {
        "path": str(path),
        "rows": count,
        "missing_required_fields": missing,
        "forbidden_or_temporal_violations": violations,
        "duplicate_ledger_keys": duplicate_keys,
        "pass": not missing and not violations and duplicate_keys == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir", type=Path)
    parser.add_argument("--mtr-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    replay = args.replay_dir
    checks = {
        "tracks": _audit_jsonl(replay / "local_tracks.jsonl", "tracks"),
        "association": _audit_jsonl(replay / "association_events.jsonl", "association"),
        "ledger": _audit_jsonl(replay / "prediction_ledger.jsonl", "ledger"),
    }
    if args.mtr_manifest is not None:
        checks["mtr_manifest"] = _audit_jsonl(args.mtr_manifest, "mtr_manifest")
    result = {
        "schema_version": "rvhca.gtfree_contract_audit.v0",
        "raw_labels_read": False,
        "checks": checks,
        "pass": all(value["pass"] for value in checks.values()),
    }
    output = args.output or (replay / "contract_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print("output=%s" % output)
    print("pass=%s" % result["pass"])
    for name, value in checks.items():
        print("%s=%s rows=%s" % (name, value["pass"], value["rows"]))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
