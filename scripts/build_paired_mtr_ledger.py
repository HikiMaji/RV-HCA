#!/usr/bin/env python
"""Build the GT-free scientific paired MTR ledger.

The PointPillar replay supplies the canonical receiver target-state stream;
the cross-replay pairing supplies the sole arrival-time geometric match.  This
command never uses the legacy per-replay prediction ledger to create pairs:
send-time and future rows are recovered only through the same persisted
``receiver_target_id``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from rvhca_cpu.mtr_io import MTRContractError, load_prediction_records, write_jsonl
from rvhca_cpu.online import DEFAULT_HORIZONS
from rvhca_cpu.paired_mtr import build_paired_mtr_ledger


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _replay_config(replay_dir: Path) -> Mapping[str, Any]:
    summary_path = Path(replay_dir) / "run_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = summary["config"]
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise MTRContractError("cannot read canonical replay summary: %s" % summary_path) from exc
    if str(config.get("tracking_frame")) != "world_fixed":
        raise MTRContractError("canonical replay is not declared world_fixed")
    return config


def _parse_horizons(value: str) -> tuple[float, ...]:
    try:
        horizons = tuple(float(item) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("horizons must be comma-separated floats") from exc
    if not horizons or any(item <= 0.0 for item in horizons):
        raise argparse.ArgumentTypeError("horizons must be non-empty and positive")
    return horizons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical_replay", type=Path, help="PointPillar fixed-world canonical replay")
    parser.add_argument("pairing", type=Path, help="cross_replay_pairing.jsonl")
    parser.add_argument("peer_predictions", type=Path, help="CoBEVT role=peer normalized MTR export")
    parser.add_argument("ego_predictions", type=Path, help="PointPillar role=ego normalized MTR export")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--horizons-s", type=_parse_horizons,
                        default=tuple(float(value) for value in DEFAULT_HORIZONS))
    parser.add_argument("--frame-period-s", type=float, default=None)
    parser.add_argument("--peer-delay-s", type=float, default=None)
    args = parser.parse_args()

    try:
        config = _replay_config(args.canonical_replay)
        frame_period = (
            float(args.frame_period_s)
            if args.frame_period_s is not None
            else float(config["frame_period_s"])
        )
        peer_delay = (
            float(args.peer_delay_s)
            if args.peer_delay_s is not None
            else float(config["peer_delay_s"])
        )
        rows, summary = build_paired_mtr_ledger(
            _read_jsonl(args.pairing),
            _read_jsonl(args.canonical_replay / "receiver_target_states.jsonl"),
            load_prediction_records([args.peer_predictions]),
            load_prediction_records([args.ego_predictions]),
            horizons_s=args.horizons_s,
            frame_period_s=frame_period,
            peer_delay_s=peer_delay,
        )
    except (MTRContractError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
        return

    output = args.output or (args.canonical_replay / "paired_mtr_ledger.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, rows)
    summary_path = args.summary or output.with_name(output.stem + "_summary.json")
    summary.update(
        {
            "canonical_replay": str(args.canonical_replay),
            "pairing": str(args.pairing),
            "peer_predictions": str(args.peer_predictions),
            "ego_predictions": str(args.ego_predictions),
            "horizons_s": list(args.horizons_s),
            "frame_period_s": frame_period,
            "peer_delay_s": peer_delay,
            "uses_gt": False,
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("paired_ledger=%s" % output)
    print("summary=%s" % summary_path)
    for key, value in sorted(summary.items()):
        print("%s=%s" % (key, value))


if __name__ == "__main__":
    main()
