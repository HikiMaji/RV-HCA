#!/usr/bin/env python
"""Run the independent raw-YAML evaluator for a GT-free replay."""

from __future__ import annotations

import argparse
from pathlib import Path

from rvhca_cpu.evaluate import evaluate_replay, write_evaluation


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir", type=Path)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=project_root / "data/raw/OPV2V/test",
    )
    parser.add_argument("--association-gate-m", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = evaluate_replay(args.replay_dir, args.raw_root, args.association_gate_m)
    output = args.output or (args.replay_dir / "offline_evaluation.json")
    write_evaluation(output, result)
    print("output=%s" % output)
    for group in ("tracking", "association", "ledger", "trajectory"):
        print("%s=%s" % (group, result[group]))


if __name__ == "__main__":
    main()

