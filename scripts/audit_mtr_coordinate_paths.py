#!/usr/bin/env python
"""Audit world-vs-receiver geometry for the GT-free MTR ledger.

For the same attached MTR prediction this script evaluates two paths:

* world path: source-local top-score point -> source pose at send -> world,
  compared with the future receiver track's world centre;
* receiver path: attached ``receiver@send_time`` top-score point compared with
  the ledger's ``realized_state`` (the future observation transformed back
  with the receiver pose at send).

The ledger path is GT-free.  This audit only uses local tracks and normalized
MTR records; it does not read raw object labels or use GT IDs.  The world path
is therefore a coordinate-frame audit, not a final trajectory benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from rvhca_cpu.online import pose_to_world_matrix, transform_point


def _real_mtr(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    model = str(value.get("model", "")).lower()
    return model not in {"", "constant_velocity_probe", "mean_probe_not_cmp"} and (
        "mtr" in model or "cmp" in model
    )


def _xyz(value: Any) -> Optional[np.ndarray]:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size < 3 or not np.isfinite(array[:3]).all():
        return None
    return array[:3]


def _json_xyz(value: Any) -> Optional[np.ndarray]:
    if isinstance(value, Mapping):
        value = value.get("position_xyz")
    return _xyz(value)


def _is_finite_pose(value: Any) -> bool:
    try:
        pose = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return False
    return pose.size == 6 and np.isfinite(pose).all()


def _read_tracks(path: Path) -> Dict[Tuple[str, str, int, int], Mapping[str, Any]]:
    index: Dict[Tuple[str, str, int, int], Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                str(row.get("sequence_id")),
                str(row.get("source")),
                int(row.get("frame_idx")),
                int(row.get("local_track_id")),
            )
            if key in index:
                raise ValueError("duplicate local-track key at line %d: %s" % (line_no, key))
            index[key] = row
    return index


def _top_points_from_raw(path: Path) -> Dict[Tuple[str, str, int, str, str], Dict[str, Any]]:
    """Index only selected horizon points from raw source-local MTR output."""
    output: Dict[Tuple[str, str, int, str, str], Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            role = str(record.get("role", ""))
            if role not in {"peer", "ego"}:
                continue
            try:
                trajectories = np.asarray(record["pred_trajs"], dtype=np.float64)
                scores = np.asarray(record["pred_scores"], dtype=np.float64).reshape(-1)
                offsets = np.asarray(record["time_offsets_s"], dtype=np.float64).reshape(-1)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid MTR record at line %d" % line_no) from exc
            if trajectories.ndim != 3 or trajectories.shape[2] < 3:
                raise ValueError("pred_trajs must be (modes,timesteps,>=3) at line %d" % line_no)
            if len(scores) != trajectories.shape[0] or len(offsets) != trajectories.shape[1]:
                raise ValueError("MTR shape/score/offset mismatch at line %d" % line_no)
            if not np.isfinite(trajectories).all() or not np.isfinite(scores).all() or not np.isfinite(offsets).all():
                raise ValueError("non-finite MTR record at line %d" % line_no)
            mode = int(np.argmax(scores))
            points = {
                "0.1": trajectories[mode, int(np.argmin(np.abs(offsets - 0.1))), :3].copy(),
                "0.3": trajectories[mode, int(np.argmin(np.abs(offsets - 0.3))), :3].copy(),
                "0.5": trajectories[mode, int(np.argmin(np.abs(offsets - 0.5))), :3].copy(),
                "1.0": trajectories[mode, int(np.argmin(np.abs(offsets - 1.0))), :3].copy(),
                "2.0": trajectories[mode, int(np.argmin(np.abs(offsets - 2.0))), :3].copy(),
                "3.0": trajectories[mode, int(np.argmin(np.abs(offsets - 3.0))), :3].copy(),
                "5.0": trajectories[mode, int(np.argmin(np.abs(offsets - 5.0))), :3].copy(),
            }
            key = (
                str(record.get("sequence_id")),
                str(record.get("source")),
                int(record.get("send_frame_idx")),
                str(record.get("source_track_id")),
                role,
            )
            if key in output:
                raise ValueError("duplicate raw MTR key at line %d: %s" % (line_no, key))
            output[key] = {
                "points": points,
                "source_pose_at_send": record.get("source_pose_at_send"),
                "forecast_frame": str(record.get("forecast_frame", "source@send_time")),
                "send_time": float(record.get("send_time")),
                "model": str(record.get("model", "")),
            }
    return output


def _stats(values: Sequence[float], tol: float) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"rows": 0, "mean": None, "median": None, "p95": None, "max": None, "over_tol": 0}
    return {
        "rows": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "over_tol": int(np.sum(array > tol)),
        "over_0.01m": int(np.sum(array > 0.01)),
        "over_0.1m": int(np.sum(array > 0.1)),
    }


def _mean_stats(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else None


def audit(
    ledger_path: Path,
    tracks_path: Path,
    raw_predictions_path: Path,
    tolerance_m: float = 1e-4,
) -> Dict[str, Any]:
    tracks = _read_tracks(tracks_path)
    raw = _top_points_from_raw(raw_predictions_path)
    by_horizon: MutableMapping[str, MutableMapping[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_role: MutableMapping[str, MutableMapping[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    overall: MutableMapping[str, List[float]] = defaultdict(list)
    counts: MutableMapping[str, int] = defaultdict(int)
    mismatch_examples: List[Dict[str, Any]] = []
    xy_projection_examples: List[Dict[str, Any]] = []
    input_rows = 0

    with ledger_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            input_rows += 1
            row = json.loads(line)
            if row.get("target_scope") != "common" or not row.get("valid_mask"):
                continue
            if not (_real_mtr(row.get("forecast")) and _real_mtr(row.get("ego_forecast"))):
                continue
            if row.get("realized_state") is None or row.get("matched_receiver_track_id") is None:
                continue
            send_pose = row.get("receiver_pose_at_send")
            if not _is_finite_pose(send_pose):
                continue
            scene = str(row.get("sequence_id"))
            receiver = str(row.get("receiver"))
            source = str(row.get("source"))
            send_frame = int(row.get("send_frame_idx"))
            obs_frame = int(row.get("observation_frame_idx"))
            horizon = "%.1f" % float(row.get("horizon", 0.0))
            target_key = (scene, receiver, obs_frame, int(row["matched_receiver_track_id"]))
            target_track = tracks.get(target_key)
            if target_track is None or not _is_finite_pose(target_track.get("pose")):
                continue
            target_world = _xyz(target_track.get("center_world"))
            realized_receiver = _json_xyz(row.get("realized_state"))
            if target_world is None or realized_receiver is None:
                continue
            receiver_to_world = pose_to_world_matrix(send_pose)
            world_to_receiver = np.linalg.inv(receiver_to_world)
            # A ground-plane control: zero roll/pitch but retain the receiver
            # send-time x/y/yaw.  Under this planar transform, world XY error
            # must equal receiver XY error exactly.  The production path uses
            # the full 6-DoF pose, so its XY projection can differ when the
            # prediction/target have different z values.
            planar_send_pose = [
                float(np.asarray(send_pose)[0]),
                float(np.asarray(send_pose)[1]),
                float(np.asarray(send_pose)[2]),
                0.0,
                float(np.asarray(send_pose)[4]),
                0.0,
            ]
            planar_world_to_receiver = np.linalg.inv(pose_to_world_matrix(planar_send_pose))
            # This is the target state reconstructed in the exact receiver
            # frame used by the ledger.  It should equal realized_state.
            target_receiver_from_world = transform_point(target_world, world_to_receiver)
            target_frame_residual = float(np.linalg.norm(target_receiver_from_world - realized_receiver))
            obs_to_world = pose_to_world_matrix(target_track["pose"])
            target_receiver_at_observation = transform_point(target_world, np.linalg.inv(obs_to_world))
            observation_pose_shift = float(np.linalg.norm(target_receiver_at_observation - realized_receiver))
            # Isolate ego-pose motion from target motion.  The observation
            # frame is not the frame used by the ledger; this origin shift is
            # a diagnostic control for what a wrong pose-time choice would do.
            observation_origin_in_send = transform_point(np.zeros(3), world_to_receiver @ obs_to_world)
            observation_pose_origin_shift = float(np.linalg.norm(observation_origin_in_send))
            observation_pose_origin_shift_xy = float(np.linalg.norm(observation_origin_in_send[:2]))
            send_yaw = float(np.asarray(send_pose, dtype=np.float64)[4])
            observation_yaw = float(np.asarray(target_track["pose"], dtype=np.float64)[4])
            yaw_delta_deg = float((observation_yaw - send_yaw + 180.0) % 360.0 - 180.0)
            frame_time_residual = abs(
                float(row.get("observation_time"))
                - float(row.get("send_time"))
                - float(row.get("horizon"))
            )

            role_specs = (
                ("peer", source, str(row.get("source_track_id")), "forecast"),
                ("ego", receiver, str(row.get("receiver_local_track_id")), "ego_forecast"),
            )
            for role, raw_source, track_id, field in role_specs:
                key = (scene, raw_source, send_frame, track_id, role)
                raw_record = raw.get(key)
                attached_receiver = _json_xyz(row.get(field))
                if raw_record is None or attached_receiver is None:
                    counts[role + ".missing_raw_or_attached"] += 1
                    continue
                source_point = raw_record["points"].get(horizon)
                if source_point is None:
                    counts[role + ".missing_horizon"] += 1
                    continue
                forecast_frame = str(raw_record.get("forecast_frame", "source@send_time"))
                source_pose = raw_record.get("source_pose_at_send")
                if forecast_frame == "world":
                    pred_world = source_point.copy()
                else:
                    if not _is_finite_pose(source_pose):
                        counts[role + ".invalid_source_pose"] += 1
                        continue
                    if forecast_frame != "source@send_time":
                        counts[role + ".unsupported_forecast_frame"] += 1
                        continue
                    pred_world = transform_point(source_point, pose_to_world_matrix(source_pose))
                pred_receiver_from_world = transform_point(pred_world, world_to_receiver)
                pred_receiver_residual = float(np.linalg.norm(pred_receiver_from_world - attached_receiver))
                world_error_xy = float(np.linalg.norm(pred_world[:2] - target_world[:2]))
                receiver_error_xy = float(np.linalg.norm(attached_receiver[:2] - realized_receiver[:2]))
                planar_pred_receiver = transform_point(pred_world, planar_world_to_receiver)
                planar_target_receiver = transform_point(target_world, planar_world_to_receiver)
                planar_receiver_error_xy = float(
                    np.linalg.norm(planar_pred_receiver[:2] - planar_target_receiver[:2])
                )
                world_error_3d = float(np.linalg.norm(pred_world - target_world))
                receiver_error_3d = float(np.linalg.norm(attached_receiver - realized_receiver))
                world_receiver_xy_diff = abs(world_error_xy - receiver_error_xy)
                world_receiver_3d_diff = abs(world_error_3d - receiver_error_3d)
                source_track = tracks.get((scene, raw_source, send_frame, int(track_id)))
                source_pose_translation_residual = None
                source_pose_angle_residual_deg = None
                if source_track is not None and _is_finite_pose(source_track.get("pose")) and _is_finite_pose(source_pose):
                    source_pose_array = np.asarray(source_pose, dtype=np.float64)
                    track_pose_array = np.asarray(source_track["pose"], dtype=np.float64)
                    source_pose_translation_residual = float(
                        np.linalg.norm(source_pose_array[:3] - track_pose_array[:3])
                    )
                    source_pose_angle_residual_deg = float(
                        np.linalg.norm(source_pose_array[3:] - track_pose_array[3:])
                    )
                for name, value in (
                    ("pred_receiver_transform_residual_m", pred_receiver_residual),
                    ("target_receiver_transform_residual_m", target_frame_residual),
                    ("world_receiver_xy_error_abs_diff_m", world_receiver_xy_diff),
                    ("world_planar_receiver_xy_error_abs_diff_m", abs(world_error_xy - planar_receiver_error_xy)),
                    ("full_pose_vs_planar_receiver_xy_extra_m", abs(receiver_error_xy - planar_receiver_error_xy)),
                    ("world_receiver_3d_error_abs_diff_m", world_receiver_3d_diff),
                    ("observation_pose_shift_m", observation_pose_shift),
                    ("observation_pose_origin_shift_m", observation_pose_origin_shift),
                    ("observation_pose_origin_shift_xy_m", observation_pose_origin_shift_xy),
                    ("observation_yaw_delta_deg", abs(yaw_delta_deg)),
                    ("frame_time_residual_s", frame_time_residual),
                ):
                    by_horizon[horizon][role + "." + name].append(value)
                    by_role[role][name].append(value)
                    overall[role + "." + name].append(value)
                if source_pose_translation_residual is not None:
                    for name, value in (
                        ("source_pose_translation_residual_m", source_pose_translation_residual),
                        ("source_pose_angle_residual_deg", source_pose_angle_residual_deg),
                    ):
                        by_horizon[horizon][role + "." + name].append(value)
                        by_role[role][name].append(value)
                        overall[role + "." + name].append(value)
                by_horizon[horizon][role + ".world_error_xy_m"].append(world_error_xy)
                by_horizon[horizon][role + ".receiver_error_xy_m"].append(receiver_error_xy)
                by_horizon[horizon][role + ".planar_receiver_error_xy_m"].append(planar_receiver_error_xy)
                by_role[role]["world_error_xy_m"].append(world_error_xy)
                by_role[role]["receiver_error_xy_m"].append(receiver_error_xy)
                by_role[role]["planar_receiver_error_xy_m"].append(planar_receiver_error_xy)
                overall[role + ".world_error_xy_m"].append(world_error_xy)
                overall[role + ".receiver_error_xy_m"].append(receiver_error_xy)
                overall[role + ".planar_receiver_error_xy_m"].append(planar_receiver_error_xy)
                counts[role + ".compared"] += 1
                if world_receiver_xy_diff > 0.01:
                    xy_projection_examples.append(
                        {
                            "line": line_no,
                            "scene": scene,
                            "receiver": receiver,
                            "source": source,
                            "role": role,
                            "forecast_frame": forecast_frame,
                            "horizon": float(horizon),
                            "world_error_xy_m": world_error_xy,
                            "receiver_error_xy_m": receiver_error_xy,
                            "world_receiver_xy_error_abs_diff_m": world_receiver_xy_diff,
                            "planar_receiver_error_xy_m": planar_receiver_error_xy,
                            "world_planar_receiver_xy_error_abs_diff_m": abs(world_error_xy - planar_receiver_error_xy),
                            "full_pose_vs_planar_receiver_xy_extra_m": abs(receiver_error_xy - planar_receiver_error_xy),
                            "world_error_3d_m": world_error_3d,
                            "receiver_error_3d_m": receiver_error_3d,
                            "observation_pose_origin_shift_m": observation_pose_origin_shift,
                            "observation_pose_origin_shift_xy_m": observation_pose_origin_shift_xy,
                            "observation_yaw_delta_deg": yaw_delta_deg,
                        }
                    )
                if (
                    pred_receiver_residual > tolerance_m
                    or target_frame_residual > tolerance_m
                    or world_receiver_3d_diff > tolerance_m
                ) and len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {
                            "line": line_no,
                            "scene": scene,
                            "receiver": receiver,
                            "source": source,
                            "role": role,
                            "send_frame_idx": send_frame,
                            "observation_frame_idx": obs_frame,
                            "horizon": float(horizon),
                            "pred_receiver_transform_residual_m": pred_receiver_residual,
                            "target_receiver_transform_residual_m": target_frame_residual,
                            "world_error_xy_m": world_error_xy,
                            "receiver_error_xy_m": receiver_error_xy,
                            "world_receiver_xy_error_abs_diff_m": world_receiver_xy_diff,
                            "world_error_3d_m": world_error_3d,
                            "receiver_error_3d_m": receiver_error_3d,
                            "world_receiver_3d_error_abs_diff_m": world_receiver_3d_diff,
                            "observation_pose_shift_m": observation_pose_shift,
                            "observation_pose_origin_shift_m": observation_pose_origin_shift,
                            "observation_pose_origin_shift_xy_m": observation_pose_origin_shift_xy,
                            "observation_yaw_delta_deg": yaw_delta_deg,
                        }
                    )

    summary: Dict[str, Any] = {
        "status": "OK" if counts["peer.compared"] and counts["ego.compared"] else "INSUFFICIENT_ROWS",
        "input_ledger_rows": input_rows,
        "local_track_rows": len(tracks),
        "raw_mtr_records": len(raw),
        "counts": dict(counts),
        "tolerance_m": tolerance_m,
        "overall": {},
        "by_role": {},
        "by_horizon": {},
        "mismatch_examples": mismatch_examples,
        "xy_projection_examples_over_0.01m": sorted(
            xy_projection_examples,
            key=lambda item: float(item["world_receiver_xy_error_abs_diff_m"]),
            reverse=True,
        )[:20],
        "time_reference_checks": {
            "forecast_frame": "raw world -> receiver_pose_at_send inverse; legacy source@send_time additionally uses source_pose_at_send",
            "realized_frame": "future receiver local track at observation_frame_idx, whose center_world is transformed by receiver_pose_at_send inverse",
            "observation_time_definition": "observation_time = send_time + horizon; valid row also requires arrival_time <= observation_time",
            "observation_pose_control": "transforming the future target with receiver pose at observation time is intentionally reported as a wrong-frame control",
        },
    }
    for key, values in sorted(overall.items()):
        summary["overall"][key] = {
            "rows": len(values),
            "mean": _mean_stats(values),
            "max": float(np.max(values)) if values else None,
            "over_tol": int(np.sum(np.asarray(values) > tolerance_m)) if values else 0,
        }
    for role, values_by_name in sorted(by_role.items()):
        summary["by_role"][role] = {}
        for name, values in sorted(values_by_name.items()):
            summary["by_role"][role][name] = _stats(values, tolerance_m if "diff" in name or "residual" in name else float("inf"))
    for horizon, values_by_name in sorted(by_horizon.items()):
        summary["by_horizon"][horizon] = {}
        for name, values in sorted(values_by_name.items()):
            summary["by_horizon"][horizon][name] = _stats(values, tolerance_m if "diff" in name or "residual" in name else float("inf"))
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=root / "data/intermediate/gtfree_cpu/opv2v_3scene_mtr_3scene_eval/prediction_ledger.jsonl",
    )
    parser.add_argument(
        "--tracks",
        type=Path,
        default=root / "data/intermediate/gtfree_cpu/opv2v_3scene/local_tracks.jsonl",
    )
    parser.add_argument(
        "--raw-predictions",
        type=Path,
        default=root / "data/intermediate/gtfree_cpu/opv2v_3scene/mtr_predictions_gtfree_3scene.jsonl",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tolerance-m", type=float, default=1e-4)
    args = parser.parse_args()
    result = audit(args.ledger, args.tracks, args.raw_predictions, args.tolerance_m)
    output = args.output or args.ledger.with_name("mtr_coordinate_path_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print("output=%s" % output)
    print("status=%s" % result["status"])
    print("counts=%s" % result["counts"])
    for role in ("peer", "ego"):
        values = result["by_role"].get(role, {})
        print(
            "%s compared=%s world_xy_mean=%s receiver_xy_mean=%s xy_diff_max=%s 3d_diff_max=%s pred_transform_max=%s target_transform_max=%s observation_pose_shift_mean=%s"
            % (
                role,
                values.get("world_error_xy_m", {}).get("rows", 0),
                values.get("world_error_xy_m", {}).get("mean"),
                values.get("receiver_error_xy_m", {}).get("mean"),
                values.get("world_receiver_xy_error_abs_diff_m", {}).get("max"),
                values.get("world_receiver_3d_error_abs_diff_m", {}).get("max"),
                values.get("pred_receiver_transform_residual_m", {}).get("max"),
                values.get("target_receiver_transform_residual_m", {}).get("max"),
                values.get("observation_pose_shift_m", {}).get("mean"),
            )
        )


if __name__ == "__main__":
    main()
