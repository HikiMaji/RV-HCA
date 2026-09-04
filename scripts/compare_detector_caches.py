#!/usr/bin/env python
"""Compare GT-free detector caches at candidate level on common scenes."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

from rvhca_cpu.evaluate import RawGTIndex

try:
    from scripts.analyze_detector_candidates import _audit, _load_scene_allowlist
except ModuleNotFoundError:
    # ``python scripts/compare_detector_caches.py`` puts the scripts directory
    # (rather than the project root) on sys.path.
    from analyze_detector_candidates import _audit, _load_scene_allowlist


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="GT-free detector cache; repeat")
    parser.add_argument("--raw-root", type=Path, default=project_root / "data/raw/OPV2V/test")
    parser.add_argument("--replay-dir", type=Path, default=project_root / "data/intermediate/gtfree_cpu/opv2v_3scene")
    parser.add_argument("--thresholds", type=str, default="0.20,0.25,0.30,0.35")
    parser.add_argument("--gate-m", type=float, default=3.0)
    parser.add_argument("--output-json", type=Path, default=project_root / "data/intermediate/detections/detector_cache_comparison.json")
    parser.add_argument("--output-md", type=Path, default=project_root / "docs/detector_cache_comparison.md")
    args = parser.parse_args()
    thresholds = tuple(float(item) for item in args.thresholds.split(",") if item.strip())
    scene_allowlist = _load_scene_allowlist(args.replay_dir)
    gt_index = RawGTIndex(args.raw_root)
    comparison: Dict[str, Any] = {}
    for input_path in args.input:
        with input_path.open("rb") as handle:
            cache = pickle.load(handle)
        result = _audit(cache, args.raw_root, thresholds, scene_allowlist, args.gate_m, gt_index=gt_index)
        comparison[input_path.stem] = result["overall"]
    output = {
        "schema_version": "rvhca.detector_cache_comparison.v0",
        "raw_root": str(args.raw_root),
        "replay_dir": str(args.replay_dir),
        "gate_m": args.gate_m,
        "thresholds": list(thresholds),
        "caches": comparison,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Detector cache comparison",
        "",
        "All values are candidate-level offline matches on the 3-scene formal replay allowlist. Raw labels are not used online.",
        "",
        "| cache | threshold | candidates | candidate precision | candidate recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for cache_name, values in sorted(comparison.items()):
        for threshold, value in sorted(values.items()):
            lines.append(
                "| %s | %s | %d | %.3f | %.3f |" % (
                    cache_name,
                    threshold,
                    value["candidates"],
                    value["candidate_precision"],
                    value["candidate_recall"],
                )
            )
    lines += [
        "",
        "The cache with the highest candidate recall at the lowest threshold is the only sensible source for the next GT-free replay. Candidate metrics are an offline upper-bound diagnostic, not trajectory metrics.",
        "",
    ]
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print("output_json=%s" % args.output_json)
    print("output_md=%s" % args.output_md)
    print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
