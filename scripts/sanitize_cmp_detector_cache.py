#!/usr/bin/env python
"""Create a GT-free detector cache from an official CMP detector cache.

The official CMP cache contains ``matched_car_id`` (an evaluator/GT-derived
field) and sometimes paths to fused features.  This utility uses a strict
allow-list and never reads those fields.  It is therefore suitable for
creating an online-input artifact; labels are only used later by the
independent offline evaluator.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


def _array(value: Any, dtype: Any = np.float32) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def sanitize(raw: Mapping[str, Any], source_model: str, mode: str) -> Dict[str, Any]:
    scenes: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    for sequence_id, source_map in sorted(raw.items(), key=lambda item: str(item[0])):
        scene_out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for source, frame_map in sorted(source_map.items(), key=lambda item: str(item[0])):
            frames_out: Dict[str, Dict[str, Any]] = {}
            for frame_key, payload in sorted(frame_map.items(), key=lambda item: int(item[0])):
                boxes_raw = _array(payload.get("pred_box3d_np", np.empty((0, 7))))
                if boxes_raw.size == 0:
                    boxes_raw = np.empty((0, 7), dtype=np.float32)
                if boxes_raw.ndim != 2 or boxes_raw.shape[1] != 7:
                    raise ValueError("%s/%s/%s pred_box3d_np has shape %s" % (sequence_id, source, frame_key, boxes_raw.shape))
                # Official CMP stores [x, y, z, l, w, h, yaw] despite an
                # outdated comment in the upstream exporter.  The GT-free
                # replay contract stores [x, y, z, h, w, l, yaw].
                boxes3d = boxes_raw[:, [0, 1, 2, 5, 4, 3, 6]].astype(np.float32, copy=False)
                boxes2d = _array(payload.get("standup_box_np", np.empty((0, 4))))
                scores = _array(payload.get("pred_score_np", np.empty((0,))))
                pose = _array(payload.get("cur_lidar_pose_np", np.zeros(6)), dtype=np.float32).reshape(-1)
                if pose.shape != (6,):
                    raise ValueError("%s/%s/%s pose has shape %s" % (sequence_id, source, frame_key, pose.shape))
                n = min(len(boxes3d), len(boxes2d), len(scores))
                boxes3d, boxes2d, scores = boxes3d[:n], boxes2d[:n], scores[:n]
                speeds = _array(payload.get("speeds", np.empty((n, 2))))
                if speeds.ndim == 1 and speeds.size == 0:
                    speeds = np.empty((0, 2), dtype=np.float32)
                if speeds.ndim != 2 or speeds.shape[1] != 2:
                    speeds = np.empty((n, 2), dtype=np.float32)
                frames_out[str(int(frame_key))] = {
                    "timestamp_idx": int(payload.get("timestamp_idx", frame_key)),
                    "timestamp_key": str(payload.get("timestamp_key", frame_key)),
                    "pose": pose.tolist(),
                    "boxes3d": boxes3d.tolist(),
                    "boxes2d": boxes2d.tolist(),
                    "scores": scores.tolist(),
                    "vel_xy": speeds[:n].tolist(),
                }
            scene_out[str(source)] = frames_out
        scenes[str(sequence_id)] = scene_out
    return {
        "schema_version": "rvhca.detection.v0",
        "dataset": "OPV2V",
        "split": "test",
        "mode": mode,
        "source_model": source_model,
        "pose_field": "pose",
        "scenes": scenes,
    }


def _summary(cache: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for scene, sources in cache["scenes"].items():
        out[scene] = {}
        for source, frames in sources.items():
            out[scene][source] = {
                "frames": len(frames),
                "detections": sum(len(frame["scores"]) for frame in frames.values()),
            }
    return out


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-model", type=str, default=None)
    parser.add_argument("--mode", type=str, default="official_cmp_raw_sanitized")
    args = parser.parse_args()
    with args.input.open("rb") as handle:
        raw = pickle.load(handle)
    source_model = args.source_model or args.input.stem
    cache = sanitize(raw, source_model, args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps({
        "schema_version": "rvhca.detection.v0",
        "input": str(args.input),
        "output": str(args.output),
        "source_model": source_model,
        "mode": args.mode,
        "scenes": _summary(cache),
    }, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print("output=%s" % args.output)
    print("metadata=%s" % metadata_path)
    print("scenes=%d" % len(cache["scenes"]))


if __name__ == "__main__":
    main()
