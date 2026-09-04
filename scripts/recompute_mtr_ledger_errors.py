#!/usr/bin/env python
"""Recompute receiver-observation errors for attached real MTR forecasts.

The base CPU replay initially contains a constant-velocity probe.  After
``ingest_mtr_predictions.py`` attaches real CMP/MTR trajectories, this command
replaces only the peer/ego error fields for rows whose attached forecast is a
real MTR output.  ``realized_state`` is the receiver's already recorded local
observation; no raw labels or GT identity is read here.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _real_mtr(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    model = str(value.get("model", "")).lower()
    return model not in {"", "constant_velocity_probe", "mean_probe_not_cmp"} and (
        "mtr" in model or "cmp" in model
    )


def _position(value: Any) -> Optional[np.ndarray]:
    if not isinstance(value, Mapping):
        return None
    point = value.get("position_xyz")
    if point is None:
        return None
    array = np.asarray(point, dtype=np.float64).reshape(-1)
    if array.size < 3 or not np.isfinite(array[:3]).all():
        return None
    return array[:3]


def _distance_xy(a: Any, b: Any) -> Optional[float]:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size < 3 or bb.size < 3 or not np.isfinite(aa[:3]).all() or not np.isfinite(bb[:3]).all():
        return None
    return float(np.linalg.norm(aa[:2] - bb[:2]))


def _distance_3d(a: Any, b: Any) -> Optional[float]:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size < 3 or bb.size < 3 or not np.isfinite(aa[:3]).all() or not np.isfinite(bb[:3]).all():
        return None
    return float(np.linalg.norm(aa[:3] - bb[:3]))


def _read(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    accumulator = _SummaryAccumulator()
    for row in rows:
        accumulator.add(row)
    return accumulator.finish()


class _SummaryAccumulator:
    """Keep only scalar MTR error pairs while streaming a large ledger."""

    def __init__(self) -> None:
        self.input_rows = 0
        self.by_horizon: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.by_scene: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.by_scene_horizon: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.overall: List[Tuple[float, float]] = []
        self.overall_3d: List[Tuple[float, float]] = []
        self.by_horizon_3d: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.by_scene_3d: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.probe: List[Tuple[float, float]] = []

    @staticmethod
    def _stats(items: Sequence[Tuple[float, float]]) -> Dict[str, Any]:
        peer = np.asarray([item[0] for item in items])
        ego = np.asarray([item[1] for item in items])
        harm = peer > ego + 0.1
        return {
            "rows": len(items),
            "peer_realized_error_mean_m": float(peer.mean()) if len(peer) else None,
            "ego_only_error_mean_m": float(ego.mean()) if len(ego) else None,
            "peer_minus_ego_mean_m": float((peer - ego).mean()) if len(peer) else None,
            "harm_rate_eps_0.1m": float(harm.mean()) if len(harm) else None,
        }

    def add(self, row: Mapping[str, Any]) -> None:
        self.input_rows += 1
        if not (
            row.get("target_scope") == "common"
            and row.get("valid_mask")
            and row.get("peer_realized_error") is not None
            and row.get("ego_only_error") is not None
            and _real_mtr(row.get("forecast"))
            and _real_mtr(row.get("ego_forecast"))
        ):
            return
        pair = (float(row["peer_realized_error"]), float(row["ego_only_error"]))
        self.overall.append(pair)
        horizon = "%.1f" % float(row.get("horizon", 0.0))
        scene = str(row.get("sequence_id", ""))
        self.by_horizon[horizon].append(pair)
        self.by_scene[scene].append(pair)
        self.by_scene_horizon[scene][horizon].append(pair)
        if row.get("peer_realized_error_3d") is not None and row.get("ego_only_error_3d") is not None:
            pair_3d = (float(row["peer_realized_error_3d"]), float(row["ego_only_error_3d"]))
            self.overall_3d.append(pair_3d)
            self.by_horizon_3d[horizon].append(pair_3d)
            self.by_scene_3d[scene].append(pair_3d)
        if row.get("probe_peer_realized_error") is not None and row.get("probe_ego_only_error") is not None:
            self.probe.append((float(row["probe_peer_realized_error"]), float(row["probe_ego_only_error"])))

    def finish(self) -> Dict[str, Any]:
        probe_peer = np.asarray([item[0] for item in self.probe])
        probe_ego = np.asarray([item[1] for item in self.probe])
        usable_count = len(self.overall)

        return {
            "status": "OK" if usable_count else "NO_VALID_REAL_MTR_ROWS",
            "input_rows": self.input_rows,
            "valid_common_real_mtr_rows": usable_count,
            "overall": self._stats(self.overall),
            "by_horizon": {key: self._stats(value) for key, value in sorted(self.by_horizon.items())},
            "by_scene": {key: self._stats(value) for key, value in sorted(self.by_scene.items())},
            "supplemental_3d": {
                "overall": self._stats(self.overall_3d),
                "by_horizon": {key: self._stats(value) for key, value in sorted(self.by_horizon_3d.items())},
                "by_scene": {key: self._stats(value) for key, value in sorted(self.by_scene_3d.items())},
            },
            "by_scene_horizon": {
                scene: {
                    horizon: self._stats(value)
                    for horizon, value in sorted(horizons.items())
                }
                for scene, horizons in sorted(self.by_scene_horizon.items())
            },
            "same_rows_constant_velocity_probe": {
                "rows": len(self.probe),
                "peer_realized_error_mean_m": float(probe_peer.mean()) if len(probe_peer) else None,
                "ego_only_error_mean_m": float(probe_ego.mean()) if len(probe_ego) else None,
                "peer_minus_ego_mean_m": float((probe_peer - probe_ego).mean()) if len(probe_peer) else None,
            },
            "error_definition": "2D XY Euclidean distance between receiver realized_state and selected-score MTR position_xyz",
            "supplemental_error_definition": "3D Euclidean distance retained in peer_realized_error_3d/ego_only_error_3d",
            "harm_definition": "peer_realized_error > ego_only_error + 0.1m",
            "inference_contract": "realized_state comes from GT-free receiver tracker; this command reads no raw labels",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_ledger", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rewritten_count = 0
    ledger_path = output_dir / "prediction_ledger.jsonl"
    accumulator = _SummaryAccumulator()
    with args.input_ledger.open("r", encoding="utf-8") as source_handle, ledger_path.open(
        "w", encoding="utf-8"
    ) as handle:
        for line in source_handle:
            if not line.strip():
                continue
            original = json.loads(line)
            row = dict(original)
            realized = row.get("realized_state")
            peer = _position(row.get("forecast"))
            ego = _position(row.get("ego_forecast"))
            if row.get("valid_mask") and _real_mtr(row.get("forecast")) and _real_mtr(row.get("ego_forecast")):
                peer_error = _distance_xy(peer, realized)
                ego_error = _distance_xy(ego, realized)
                peer_error_3d = _distance_3d(peer, realized)
                ego_error_3d = _distance_3d(ego, realized)
                if peer_error is not None and ego_error is not None:
                    # Keep the old values auditable without using them downstream.
                    row["probe_peer_realized_error"] = row.get("peer_realized_error")
                    row["probe_ego_only_error"] = row.get("ego_only_error")
                    row["peer_realized_error"] = peer_error
                    row["ego_only_error"] = ego_error
                    row["peer_realized_error_3d"] = peer_error_3d
                    row["ego_only_error_3d"] = ego_error_3d
                    row["error_source"] = "receiver_observation:mtr_selected_mode"
                    rewritten_count += 1
            accumulator.add(row)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    summary = accumulator.finish()
    summary["rewritten_mtr_rows"] = rewritten_count
    with (output_dir / "mtr_error_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print("output_dir=%s" % output_dir)
    print("ledger=%s" % ledger_path)
    print("rewritten_mtr_rows=%d" % rewritten_count)
    print("summary=%s" % (output_dir / "mtr_error_summary.json"))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
