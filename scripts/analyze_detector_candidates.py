#!/usr/bin/env python
"""Offline detector-candidate coverage audit for the GT-free replay.

The sanitized detector cache is treated as an online artifact here.  Raw
OPV2V labels are read only by this evaluator to separate detector coverage
from subsequent AB3DMOT/receiver-track losses.  No result from this utility is
fed back into tracking or ledger construction.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from rvhca_cpu.evaluate import RawGTIndex, _match_centers, read_jsonl
from rvhca_cpu.online import pose_to_world_matrix, transform_point


DEFAULT_THRESHOLDS = (0.20, 0.25, 0.30, 0.35)


def _ratio(num: int, den: int) -> Optional[float]:
    return float(num) / float(den) if den else None


def _load_scene_allowlist(replay_dir: Optional[Path]) -> Optional[set]:
    if replay_dir is None:
        return None
    path = Path(replay_dir) / "local_tracks.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    scenes = set()
    for row in read_jsonl(path):
        scenes.add(str(row["sequence_id"]))
    return scenes


def _frame_gt(
    gt_index: RawGTIndex,
    scene: str,
    source: str,
    frame: Mapping[str, Any],
) -> Mapping[str, np.ndarray]:
    timestamp_key = str(frame.get("timestamp_key", ""))
    gt = gt_index.at(scene, source, timestamp_key)
    if gt:
        return gt
    return gt_index.by_frame(scene, source, int(frame.get("timestamp_idx", 0)))


def _audit(
    cache: Mapping[str, Any],
    raw_root: Path,
    thresholds: Sequence[float],
    scene_allowlist: Optional[set],
    gate_m: float,
    gt_index: Optional[RawGTIndex] = None,
) -> Dict[str, Any]:
    gt_index = gt_index or RawGTIndex(raw_root)
    overall: MutableMapping[float, Dict[str, int]] = defaultdict(
        lambda: {"candidates": 0, "matched_candidates": 0, "gt_instances": 0, "matched_gt_instances": 0, "frames": 0}
    )
    per_source: Dict[str, Dict[str, Any]] = {}
    for scene, sources in sorted(cache.get("scenes", {}).items()):
        if scene_allowlist is not None and scene not in scene_allowlist:
            continue
        for source, frame_map in sorted(sources.items(), key=lambda item: str(item[0])):
            counters: MutableMapping[float, Dict[str, int]] = defaultdict(
                lambda: {"candidates": 0, "matched_candidates": 0, "gt_instances": 0, "matched_gt_instances": 0, "frames": 0}
            )
            for frame_key in sorted(frame_map, key=lambda item: int(item)):
                frame = frame_map[frame_key]
                gt = _frame_gt(gt_index, str(scene), str(source), frame)
                boxes = np.asarray(frame.get("boxes3d", np.empty((0, 7))), dtype=np.float64)
                scores = np.asarray(frame.get("scores", np.empty((0,))), dtype=np.float64).reshape(-1)
                pose = np.asarray(frame["pose"], dtype=np.float64)
                to_world = pose_to_world_matrix(pose)
                centers = [transform_point(box[:3], to_world).tolist() for box in boxes if np.isfinite(box[:3]).all()]
                candidates = [{"center_world": center} for center in centers]
                for threshold in thresholds:
                    threshold = float(threshold)
                    selected = [row for row, score in zip(candidates, scores) if np.isfinite(score) and float(score) >= threshold]
                    matched = _match_centers(selected, gt, gate_m)
                    matched_detection = sum(value[0] is not None for value in matched.values())
                    matched_gt = {value[0] for value in matched.values() if value[0] is not None}
                    c = counters[threshold]
                    c["candidates"] += len(selected)
                    c["matched_candidates"] += matched_detection
                    c["gt_instances"] += len(gt)
                    c["matched_gt_instances"] += len(matched_gt)
                    c["frames"] += 1
                    o = overall[threshold]
                    o["candidates"] += len(selected)
                    o["matched_candidates"] += matched_detection
                    o["gt_instances"] += len(gt)
                    o["matched_gt_instances"] += len(matched_gt)
                    o["frames"] += 1
            per_source["%s/%s" % (scene, source)] = {
                "sequence_id": str(scene),
                "source": str(source),
                "thresholds": {
                    "%.2f" % threshold: {
                        **counts,
                        "candidate_precision": _ratio(counts["matched_candidates"], counts["candidates"]),
                        "candidate_recall": _ratio(counts["matched_gt_instances"], counts["gt_instances"]),
                    }
                    for threshold, counts in sorted(counters.items())
                },
            }
    return {
        "overall": {
            "%.2f" % threshold: {
                **counts,
                "candidate_precision": _ratio(counts["matched_candidates"], counts["candidates"]),
                "candidate_recall": _ratio(counts["matched_gt_instances"], counts["gt_instances"]),
            }
            for threshold, counts in sorted(overall.items())
        },
        "per_scene_source": per_source,
    }


def _markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Detector-candidate coverage audit",
        "",
        "This is an offline raw-label audit of the sanitized detector cache. It estimates how much recall is available before AB3DMOT and receiver-side association. Labels are not used by online replay.",
        "",
        "## Overall detector operating points",
        "",
        "| threshold | candidates | candidate precision | candidate recall |",
        "|---:|---:|---:|---:|",
    ]
    for threshold, value in result["overall"].items():
        lines.append(
            "| %s | %d | %.3f | %.3f |" % (
                threshold,
                value["candidates"],
                value["candidate_precision"],
                value["candidate_recall"],
            )
        )
    lines += ["", "## Interpretation", "", "Compare candidate recall with the formal local-track recall (0.722). If candidate recall is already low, a tracker-only change cannot reach the support gate; if candidate recall is high but track recall is low, AB3DMOT/score filtering is the immediate CPU-side target. Candidate precision is not trajectory-prediction quality.", ""]
    return "\n".join(lines)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=project_root / "data/intermediate/detections/opv2v_point_pillar_sinbevt_test_gtfree.pkl")
    parser.add_argument("--raw-root", type=Path, default=project_root / "data/raw/OPV2V/test")
    parser.add_argument("--replay-dir", type=Path, default=None, help="restrict scenes to those in local_tracks.jsonl")
    parser.add_argument("--thresholds", type=str, default="0.20,0.25,0.30,0.35")
    parser.add_argument("--gate-m", type=float, default=3.0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()
    thresholds = tuple(float(item) for item in args.thresholds.split(",") if item.strip())
    with args.input.open("rb") as handle:
        cache = pickle.load(handle)
    result = {
        "schema_version": "rvhca.detector_candidate_audit.v0",
        "input": str(args.input),
        "raw_root": str(args.raw_root),
        "replay_dir": str(args.replay_dir) if args.replay_dir else None,
        "gate_m": args.gate_m,
        "thresholds": list(thresholds),
        "metrics": _audit(cache, args.raw_root, thresholds, _load_scene_allowlist(args.replay_dir), args.gate_m),
    }
    output_json = args.output_json or (args.replay_dir / "detector_candidate_audit.json" if args.replay_dir else project_root / "data/intermediate/detector_candidate_audit.json")
    output_md = args.output_md or (args.replay_dir / "detector_candidate_audit.md" if args.replay_dir else project_root / "docs/detector_candidate_audit.md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(_markdown(result["metrics"]), encoding="utf-8")
    print("output_json=%s" % output_json)
    print("output_md=%s" % output_md)
    print(json.dumps(result["metrics"]["overall"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
