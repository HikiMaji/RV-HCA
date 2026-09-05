#!/usr/bin/env python
"""Run the official CMP/MTR motion model on GT-free local tracks.

This adapter intentionally bypasses the native OPV2V MTR dataset/evaluator.
The native path carries GT object ids and future labels in its batch.  Here
the only input is the GT-free track history produced by the replay (history
positions are world-fixed; local IDs remain source-local).  The output is the
small normalized prediction contract consumed by
``scripts/ingest_mtr_predictions.py``.

The script is an inference/export adapter, not a reliability controller.  It
uses one source/frame group per forward pass because the upstream encoder's
lane indexing assumes that layout.  Use ``--role peer`` or ``--role ego`` with
the corresponding native detector-history replay when the two official
checkpoints require different input distributions; ``--role both`` is reserved
for a genuinely matched paired replay.  Use ``--max-frame``/``--max-groups``
for a smoke run before a longer GPU replay.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _track_source_model(tracks_path: Path, explicit: Optional[str]) -> str:
    """Recover detector provenance without looking at labels.

    New CPU replays put this in ``run_summary.json``.  The filename fallback
    keeps the pre-existing detector-cache naming convention useful, while an
    unknown source is intentionally not treated as compatible in strict mode.
    """
    if explicit:
        return str(explicit)
    summary_path = tracks_path.parent / "run_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            value = summary.get("config", {}).get("detector_source_model")
            if value:
                return str(value)
        except (OSError, ValueError, TypeError):
            pass
    name = tracks_path.parent.name.lower() + " " + tracks_path.name.lower()
    if "corpbevt" in name or "cobevt" in name:
        return "corpbevtlidar_delay_1_frame_aug_c256"
    if "point_pillar" in name or "pointpillar" in name:
        return "point_pillar_sinbevt"
    return "unknown"


def _history_family(config_path: Path) -> str:
    """Return the detector-history family declared by an MTR YAML."""
    text = config_path.read_text(encoding="utf-8").lower()
    if "tracking_trajs_corpbevtlidar_delay_1_frame_aug_c256" in text:
        return "corpbevtlidar_delay_1_frame_aug_c256"
    if "tracking_trajs_point_pillar_sinbevt" in text:
        return "point_pillar_sinbevt"
    return "unknown"


def _source_family(source_model: str) -> str:
    value = str(source_model).lower()
    if "corpbevt" in value or "cobevt" in value:
        return "corpbevtlidar_delay_1_frame_aug_c256"
    if "point_pillar" in value or "pointpillar" in value:
        return "point_pillar_sinbevt"
    return "unknown"


def _input_distribution_audit(
    tracks_path: Path,
    peer_config: Path,
    ego_config: Path,
    explicit_source_model: Optional[str],
    role: str = "both",
) -> Dict[str, Any]:
    source_model = _track_source_model(tracks_path, explicit_source_model)
    source_family = _source_family(source_model)
    peer_family = _history_family(peer_config)
    ego_family = _history_family(ego_config)
    issues = []
    if source_family == "unknown":
        issues.append("track provenance is unknown")
    if role in {"both", "peer"} and peer_family == "unknown":
        issues.append("peer checkpoint history family is unknown")
    if role in {"both", "ego"} and ego_family == "unknown":
        issues.append("ego checkpoint history family is unknown")
    if role in {"both", "peer"} and peer_family != "unknown" and source_family != "unknown" and source_family != peer_family:
        issues.append("peer checkpoint history family does not match track cache")
    if role in {"both", "ego"} and ego_family != "unknown" and source_family != "unknown" and source_family != ego_family:
        issues.append("ego checkpoint history family does not match track cache")
    return {
        "tracks": str(tracks_path),
        "declared_source_model": source_model,
        "track_history_family": source_family,
        "peer_config": str(peer_config),
        "peer_history_family": peer_family,
        "ego_config": str(ego_config),
        "ego_history_family": ego_family,
        "role": role,
        "issues": issues,
        "status": "PASS" if not issues else "MISMATCH",
    }


def _dims_lwh(row: Mapping[str, Any]) -> Tuple[float, float, float]:
    # The GT-free track contract stores dimensions as [height, width, length].
    h, w, l = [float(value) for value in row["dims_hwl"]]
    return l, w, h


def _build_group(
    rows: Sequence[Mapping[str, Any]],
    scene: str,
    source: str,
    frame: int,
    lane_root: Path,
    min_history: int = 11,
) -> Optional[Dict[str, Any]]:
    """Construct an MTR batch from source-local tracks only.

    ``center_indices`` are indexes into the local-track context array.  They
    are never GT ids and are not exported to the prediction contract.
    """

    source_rows = [row for row in rows if str(row["sequence_id"]) == scene and str(row["source"]) == source]
    by_frame: MutableMapping[int, List[Mapping[str, Any]]] = defaultdict(list)
    by_id: MutableMapping[int, MutableMapping[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in source_rows:
        cur_frame = int(row["frame_idx"])
        by_frame[cur_frame].append(row)
        by_id[int(row["local_track_id"])][cur_frame] = row
    current = by_frame.get(frame, [])
    if not current:
        return None

    history_frames = list(range(frame - min_history + 1, frame + 1))
    # Context contains every local ID observed in this history window; center
    # targets are restricted to tracks with a complete causal history.
    context_ids = sorted(
        {
            int(row["local_track_id"])
            for row in source_rows
            if frame - min_history + 1 <= int(row["frame_idx"]) <= frame
        }
    )
    current_by_id = {int(row["local_track_id"]): row for row in current}
    center_ids = [
        track_id
        for track_id in sorted(current_by_id)
        if all(history_frame in by_id[track_id] for history_frame in history_frames)
    ]
    if not center_ids:
        return None
    center_indices = np.asarray([context_ids.index(track_id) for track_id in center_ids], dtype=np.int64)

    num_objects = len(context_ids)
    num_centers = len(center_ids)
    past = np.zeros((num_objects, min_history, 8), dtype=np.float32)
    for object_index, track_id in enumerate(context_ids):
        for time_index, history_frame in enumerate(history_frames):
            row = by_id[track_id].get(history_frame)
            if row is None:
                continue
            if "center_world" not in row or "yaw_world" not in row:
                raise ValueError(
                    "track rows must contain center_world/yaw_world; regenerate GT-free replay with the fixed-world tracker"
                )
            length, width, height = _dims_lwh(row)
            past[object_index, time_index] = [
                float(row["center_world"][0]),
                float(row["center_world"][1]),
                float(row["center_world"][2]),
                length,
                width,
                height,
                float(row["yaw_world"]),
                1.0,
            ]

    centers = np.asarray([past[context_ids.index(track_id), -1] for track_id in center_ids], dtype=np.float32)
    obj_types = np.asarray(["TYPE_VEHICLE"] * num_objects)
    # The decoder is in eval mode, so future labels are not read.  A zero,
    # invalid placeholder lets us reuse the upstream feature builder without
    # importing its GT-bearing dataset/evaluator path.
    dummy_future = np.zeros((num_objects, 50, 8), dtype=np.float32)
    # Match the upstream OPV2V adapter exactly: its timestamp array is
    # ``0.1 * (0.1 * frame_idx)`` (0.01 per frame), and the selected past
    # window is an absolute slice rather than a -1..0 relative interval.
    timestamps = np.arange(frame - min_history + 1, frame + 1, dtype=np.float32) * 0.01

    # Import lazily so a CPU-only contract audit can still use the repository.
    from mtr.datasets.opv2v_multiego_dataset import OPV2VMultiEgoDataset

    obj_trajs, obj_mask, _, _ = OPV2VMultiEgoDataset.generate_centered_trajs_for_agents(
        center_objects=centers,
        obj_trajs_past=past,
        obj_types=obj_types,
        center_indices=center_indices,
        sdc_index=0,
        timestamps=timestamps,
        obj_trajs_future=dummy_future,
    )
    obj_mask = obj_mask > 0
    obj_last_pos = np.zeros((num_centers, num_objects, 3), dtype=np.float32)
    for time_index in range(min_history):
        valid = obj_mask[:, :, time_index]
        obj_last_pos[valid] = obj_trajs[:, :, time_index, :3][valid]

    timestamp_key = str(current_by_id[center_ids[0]]["timestamp_key"])
    lane_path = lane_root / scene / source / (timestamp_key + "_bev_lane.png")
    if not lane_path.exists():
        return None
    lane = np.asarray(Image.open(lane_path).convert("RGB"))
    lane = np.repeat(lane[None, ...], num_centers, axis=0)
    pose = list(current_by_id[center_ids[0]]["pose"])
    return {
        "scene": scene,
        "source": source,
        "frame": frame,
        "center_ids": center_ids,
        "center_world_xy": centers[:, 0:2].copy(),
        "center_z": centers[:, 2].copy(),
        "center_yaw_world": centers[:, 6].copy(),
        "history_reference_frame": "world",
        "source_pose": pose,
        "obj_trajs": obj_trajs.astype(np.float32, copy=False),
        "obj_mask": obj_mask,
        "obj_last_pos": obj_last_pos,
        "lane": lane,
        "center_indices": center_indices,
    }


def _make_batch(group: Mapping[str, Any]) -> Dict[str, Any]:
    center_count = len(group["center_ids"])
    input_dict = {
        "obj_trajs": torch.from_numpy(group["obj_trajs"]),
        "obj_trajs_mask": torch.from_numpy(group["obj_mask"]).bool(),
        "obj_trajs_last_pos": torch.from_numpy(group["obj_last_pos"]),
        "map_polylines": torch.from_numpy(group["lane"]),
        "track_index_to_predict": torch.from_numpy(group["center_indices"]).long(),
        "center_objects_type": np.asarray(["TYPE_VEHICLE"] * center_count),
    }
    return {"num_cavs": 1, "batch_sample_count": [center_count], "input_dict": input_dict}


def _load_model(config_path: Path, lane_encoder: Path, checkpoint: Path):
    from mtr.config import cfg, cfg_from_yaml_file
    from mtr.models_opv2v.multi_ego_mtr_model import MotionTransformerWithMultiEgoAggregation

    # cfg is process-global in the upstream code.  This process creates one
    # model instance and swaps only its weights, so loading the YAML once is
    # deterministic and avoids duplicating the 1.8 GB checkpoint in RAM.
    cfg_from_yaml_file(str(config_path), cfg)
    cfg.MODEL.CONTEXT_ENCODER.LANE_ENCODER = str(lane_encoder)
    model = MotionTransformerWithMultiEgoAggregation(cfg.MODEL)
    model.cuda().eval()
    state = torch.load(str(checkpoint), map_location="cuda:0")
    model.load_state_dict(state["model_state"], strict=True)
    del state
    gc.collect()
    torch.cuda.empty_cache()
    return model


def _swap_weights(model: torch.nn.Module, checkpoint: Path) -> None:
    state = torch.load(str(checkpoint), map_location="cuda:0")
    model.load_state_dict(state["model_state"], strict=True)
    del state
    gc.collect()
    torch.cuda.empty_cache()


def _records_for_group(
    model: torch.nn.Module,
    group: Mapping[str, Any],
    role: str,
    model_name: str,
    provenance: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    with torch.inference_mode():
        output = model(_make_batch(group))
    pred_xy = output["pred_trajs"][:, :, :, :2]
    scores = output["pred_scores"]
    # Upstream MTR predicts in each target's centered frame.  This wrapper
    # builds that frame from world-fixed histories, so the official inverse
    # transform returns points directly to global world XY.
    center_xy = torch.from_numpy(np.asarray(group["center_world_xy"], dtype=np.float32)).cuda()
    center_yaw = torch.from_numpy(np.asarray(group["center_yaw_world"], dtype=np.float32)).cuda()
    cos_yaw = torch.cos(center_yaw)[:, None, None]
    sin_yaw = torch.sin(center_yaw)[:, None, None]
    x = pred_xy[:, :, :, 0]
    y = pred_xy[:, :, :, 1]
    world_x = x * cos_yaw - y * sin_yaw + center_xy[:, None, None, 0]
    world_y = x * sin_yaw + y * cos_yaw + center_xy[:, None, None, 1]
    world_xy = torch.stack((world_x, world_y), dim=-1).cpu().numpy()
    # MTR's five-dimensional head contains planar position and GMM terms; it
    # does not predict a useful vertical coordinate.  Preserve the observed
    # local box height at send time rather than letting the contract's 2-D
    # fallback inject an artificial z=0 error.
    center_z = np.asarray(group["center_z"], dtype=np.float32)[:, None, None, None]
    world_xyz = np.concatenate(
        (world_xy, np.repeat(center_z, world_xy.shape[1], axis=1).repeat(world_xy.shape[2], axis=2)),
        axis=-1,
    )
    score_np = scores.detach().cpu().numpy()
    offsets = (np.arange(1, world_xy.shape[2] + 1, dtype=np.float64) * 0.1).tolist()
    records: List[Dict[str, Any]] = []
    for index, track_id in enumerate(group["center_ids"]):
        records.append(
            {
                "schema_version": "rvhca.mtr_prediction.v1",
                "sequence_id": str(group["scene"]),
                "source": str(group["source"]),
                "source_track_id": str(track_id),
                "send_frame_idx": int(group["frame"]),
                "send_time": float(group["frame"]) * 0.1,
                "role": role,
                "forecast_frame": "world",
                "pred_trajs": world_xyz[index].tolist(),
                "pred_scores": score_np[index].tolist(),
                "time_offsets_s": offsets,
                "source_pose_at_send": list(group["source_pose"]),
                "model": model_name,
                "history_reference_frame": str(group.get("history_reference_frame", "world")),
                "track_history_family": str(provenance["track_history_family"]),
                "checkpoint_history_family": str(provenance["checkpoint_history_family"]),
                "input_distribution_status": str(provenance["input_distribution_status"]),
                "uses_gt": False,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    default_replay = root / "data" / "intermediate" / "gtfree_cpu" / "opv2v_3scene"
    parser.add_argument("--replay-dir", type=Path, default=default_replay)
    parser.add_argument("--tracks", type=Path, default=None)
    parser.add_argument("--lane-root", type=Path, default=root / "data" / "raw" / "OPV2V" / "additional" / "test")
    parser.add_argument("--mtr-root", type=Path, default=root / "vendor" / "CMP-upstream" / "MTR")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--ego-config",
        type=Path,
        default=None,
        help="native no-coop YAML used only to audit the ego history distribution",
    )
    parser.add_argument(
        "--track-source-model",
        type=str,
        default=None,
        help="detector provenance written by the CPU replay; required when it cannot be inferred",
    )
    parser.add_argument(
        "--allow-input-mismatch",
        action="store_true",
        help="run a labelled smoke test despite checkpoint/history mismatch; never use for scientific results",
    )
    parser.add_argument("--lane-encoder", type=Path, default=root / "data" / "assets" / "checkpoints" / "CMP" / "pretrained" / "opv2v" / "swin-base-patch4-window7-224")
    parser.add_argument("--peer-checkpoint", type=Path, default=root / "data" / "assets" / "checkpoints" / "CMP" / "MTR" / "output" / "opv2v_multiego_cobevt_c256_no_agg" / "ckpt" / "best_model.pth")
    parser.add_argument("--ego-checkpoint", type=Path, default=root / "data" / "assets" / "checkpoints" / "CMP" / "MTR" / "output" / "opv2v_multiego_no_coop" / "ckpt" / "best_model.pth")
    parser.add_argument(
        "--role",
        choices=("both", "peer", "ego"),
        default="both",
        help="export one checkpoint-compatible role when building separate native-distribution replays",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--scene", action="append", default=None)
    parser.add_argument("--min-frame", type=int, default=10)
    parser.add_argument("--max-frame", type=int, default=None)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--max-centers", type=int, default=None)
    args = parser.parse_args()

    # Make the upstream model package importable even when the caller did not
    # pre-populate PYTHONPATH (the docs still set it explicitly for clarity).
    for import_root in (args.mtr_root, args.mtr_root.parent):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))

    tracks_path = args.tracks or (args.replay_dir / "local_tracks.jsonl")
    output = args.output or (args.replay_dir / "mtr_predictions_gtfree.jsonl")
    config = args.config or (args.mtr_root / "tools" / "cfgs" / "opv2v" / "opv2v_multiego_cobevt_c256_no_agg.yaml")
    ego_config = args.ego_config or (args.mtr_root / "tools" / "cfgs" / "opv2v" / "opv2v_multiego_no_coop.yaml")
    input_audit = _input_distribution_audit(
        tracks_path, config, ego_config, args.track_source_model, args.role
    )
    print("input_distribution_audit=%s" % json.dumps(input_audit, ensure_ascii=False, sort_keys=True), flush=True)
    if input_audit["issues"] and not args.allow_input_mismatch:
        raise SystemExit(
            "refusing MTR inference with an unverified checkpoint/history distribution; "
            "use a compatible GT-free track replay or --allow-input-mismatch for a non-scientific smoke test"
        )
    peer_provenance = {
        "track_history_family": input_audit["track_history_family"],
        "checkpoint_history_family": input_audit["peer_history_family"],
        "input_distribution_status": input_audit["status"],
    }
    ego_provenance = {
        "track_history_family": input_audit["track_history_family"],
        "checkpoint_history_family": input_audit["ego_history_family"],
        "input_distribution_status": input_audit["status"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(output.suffix + ".input_audit.json").open("w", encoding="utf-8") as audit_handle:
        json.dump(input_audit, audit_handle, ensure_ascii=False, indent=2, sort_keys=True)
    rows = _read_jsonl(tracks_path)
    scenes = set(str(value) for value in args.scene) if args.scene else None
    grouped_keys = sorted(
        {
            (str(row["sequence_id"]), str(row["source"]), int(row["frame_idx"]))
            for row in rows
            if int(row["frame_idx"]) >= args.min_frame
            and (args.max_frame is None or int(row["frame_idx"]) <= args.max_frame)
            and (scenes is None or str(row["sequence_id"]) in scenes)
        }
    )
    groups: List[Dict[str, Any]] = []
    skipped_no_history = 0
    skipped_no_lane = 0
    total_centers = 0
    for scene, source, frame in grouped_keys:
        group = _build_group(rows, scene, source, frame, args.lane_root)
        if group is None:
            # Distinguish missing lane from insufficient history for a useful
            # audit without importing any label data.
            has_current = any(
                str(row["sequence_id"]) == scene and str(row["source"]) == source and int(row["frame_idx"]) == frame
                for row in rows
            )
            lane_exists = any(
                str(row["sequence_id"]) == scene and str(row["source"]) == source and int(row["frame_idx"]) == frame
                and (args.lane_root / scene / source / (str(row["timestamp_key"]) + "_bev_lane.png")).exists()
                for row in rows
            )
            skipped_no_lane += int(has_current and not lane_exists)
            skipped_no_history += int(not lane_exists or has_current)
            continue
        if args.max_centers is not None and total_centers >= args.max_centers:
            break
        groups.append(group)
        total_centers += len(group["center_ids"])
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
        total_centers = sum(len(group["center_ids"]) for group in groups)
    if not groups:
        raise SystemExit("no GT-free source/frame groups with complete history and lane image")

    print(
        "groups=%d centers=%d skipped_no_history_or_lane=%d skipped_no_lane=%d"
        % (len(groups), total_centers, skipped_no_history, skipped_no_lane),
        flush=True,
    )
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model_config = ego_config if args.role == "ego" else config
    model_checkpoint = args.ego_checkpoint if args.role == "ego" else args.peer_checkpoint
    model = _load_model(model_config, args.lane_encoder, model_checkpoint)
    partial = output.with_suffix(output.suffix + ".partial")
    start = time.time()
    written = 0
    with partial.open("w", encoding="utf-8") as handle:
        if args.role in {"both", "peer"}:
            for index, group in enumerate(groups, start=1):
                for record in _records_for_group(
                    model, group, "peer", "cmp_mtr_no_agg", peer_provenance
                ):
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    written += 1
                if index == 1 or index % 10 == 0 or index == len(groups):
                    elapsed = max(time.time() - start, 1e-9)
                    print(
                        "peer_group=%d/%d records=%d rate=%.2f centers/s gpu_alloc=%.2fGB"
                        % (index, len(groups), written, total_centers * index / len(groups) / elapsed, torch.cuda.memory_allocated() / 1e9),
                        flush=True,
                    )
        if args.role == "both":
            _swap_weights(model, args.ego_checkpoint)
        if args.role in {"both", "ego"}:
            for index, group in enumerate(groups, start=1):
                for record in _records_for_group(
                    model, group, "ego", "cmp_mtr_no_coop", ego_provenance
                ):
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    written += 1
                if index == 1 or index % 10 == 0 or index == len(groups):
                    print("ego_group=%d/%d records=%d" % (index, len(groups), written), flush=True)
    os.replace(partial, output)
    print("output=%s" % output)
    print("records=%d" % written)
    print("elapsed_s=%.2f" % (time.time() - start))


if __name__ == "__main__":
    main()
