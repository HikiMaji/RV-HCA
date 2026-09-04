#!/usr/bin/env python
"""Run the CPU-only GT-free tracking/association/ledger replay."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from rvhca_cpu.online import DEFAULT_HORIZONS, build_replay, write_replay
from rvhca_cpu.baselines import summarize_baselines


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data/intermediate/detections/opv2v_point_pillar_sinbevt_test_gtfree.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data/intermediate/gtfree_cpu/test_probe",
    )
    parser.add_argument("--scene", action="append", dest="scenes", help="repeatable scene allowlist")
    parser.add_argument("--max-frames-per-source", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--association-gate-m", type=float, default=4.0)
    parser.add_argument("--target-gate-m", type=float, default=4.0)
    parser.add_argument("--target-max-miss", type=int, default=2)
    parser.add_argument("--peer-delay-s", type=float, default=0.1)
    parser.add_argument("--frame-period-s", type=float, default=0.1)
    parser.add_argument(
        "--horizons-s",
        type=str,
        default=",".join("%.3f" % value for value in DEFAULT_HORIZONS),
    )
    parser.add_argument(
        "--aggregate-mode",
        choices=("none", "mean_probe"),
        default="none",
        help="mean_probe is a contract smoke baseline and is not CMP output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("rb") as handle:
        detector_cache = pickle.load(handle)
    horizons = tuple(float(value) for value in args.horizons_s.split(",") if value.strip())
    replay = build_replay(
        detector_cache,
        frame_period_s=args.frame_period_s,
        min_score=args.min_score,
        association_gate_m=args.association_gate_m,
        target_gate_m=args.target_gate_m,
        target_max_miss=args.target_max_miss,
        peer_delay_s=args.peer_delay_s,
        horizons_s=horizons,
        aggregate_mode=args.aggregate_mode,
        scene_allowlist=args.scenes,
        max_frames_per_source=args.max_frames_per_source,
        detector_source_model=str(detector_cache.get("source_model", args.input.stem)),
        detector_cache_path=str(args.input),
    )
    write_replay(args.output_dir, replay)
    baseline_summary = summarize_baselines(replay["ledger"])
    with (args.output_dir / "baseline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(baseline_summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print("output_dir=%s" % args.output_dir)
    for key, value in sorted(replay["summary"].items()):
        print("%s=%s" % (key, value))
    print("baseline_summary=%s" % (args.output_dir / "baseline_summary.json"))


if __name__ == "__main__":
    main()
