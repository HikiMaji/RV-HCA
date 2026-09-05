#!/usr/bin/env python
"""Offline-only raw-GT audit for a finished cross-replay pairing artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from rvhca_cpu.cross_replay_evaluate import CrossReplayAuditError, evaluate_cross_replay_pairing


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _association_gate(replay_dir: Path) -> float:
    summary_path = Path(replay_dir) / "run_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return float(summary["config"]["association_gate_m"])
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise CrossReplayAuditError("cannot read canonical replay summary: %s" % summary_path) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical_replay", type=Path, help="PointPillar canonical replay directory")
    parser.add_argument("peer_replay", type=Path, help="CoBEVT peer replay directory")
    parser.add_argument("pairing", type=Path, help="cross_replay_pairing.jsonl")
    parser.add_argument("raw_root", type=Path, help="raw OPV2V root; read only for this audit")
    parser.add_argument("--association-gate-m", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        gate = args.association_gate_m if args.association_gate_m is not None else _association_gate(args.canonical_replay)
        summary = evaluate_cross_replay_pairing(
            _read_jsonl(args.pairing),
            _read_jsonl(args.canonical_replay / "local_tracks.jsonl"),
            _read_jsonl(args.peer_replay / "local_tracks.jsonl"),
            args.raw_root,
            association_gate_m=gate,
        )
    except (CrossReplayAuditError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
        return

    output = args.output or args.pairing.with_name(args.pairing.stem + "_offline_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("offline_audit=%s" % output)
    for key, value in sorted(summary.items()):
        print("%s=%s" % (key, value))


if __name__ == "__main__":
    main()
