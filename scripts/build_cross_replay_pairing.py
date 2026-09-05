#!/usr/bin/env python
"""Build GT-free CoBEVT-to-PointPillar receiver target pairings.

The PointPillar replay is the canonical receiver state stream.  This command
never reads GT, MTR forecasts, or a local ID from one replay as an ID in the
other replay.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from rvhca_cpu.cross_replay import CrossReplayContractError, build_cross_replay_pairings


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _config(summary_path: Path, label: str) -> Mapping[str, Any]:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = summary["config"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CrossReplayContractError("cannot read %s run summary: %s" % (label, summary_path)) from exc
    if str(config.get("tracking_frame")) != "world_fixed":
        raise CrossReplayContractError("%s replay is not declared world_fixed" % label)
    return config


def _validate_target_state_coverage(states: List[Mapping[str, Any]], tracks: List[Mapping[str, Any]]) -> None:
    state_keys = {
        (
            str(row["sequence_id"]),
            str(row["receiver"]),
            int(row["frame_idx"]),
            int(row["receiver_local_track_id"]),
        )
        for row in states
    }
    track_keys = {
        (str(row["sequence_id"]), str(row["source"]), int(row["frame_idx"]), int(row["local_track_id"]))
        for row in tracks
    }
    if state_keys != track_keys:
        raise CrossReplayContractError(
            "canonical receiver target states must cover exactly all canonical local-track rows "
            "(states=%d tracks=%d)" % (len(state_keys), len(track_keys))
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical_replay", type=Path, help="PointPillar fixed-world replay directory")
    parser.add_argument("peer_replay", type=Path, help="CoBEVT fixed-world replay directory")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--frame-period-s", type=float, default=None)
    parser.add_argument("--peer-delay-s", type=float, default=None)
    parser.add_argument("--association-gate-m", type=float, default=None)
    args = parser.parse_args()

    canonical_config = _config(args.canonical_replay / "run_summary.json", "canonical")
    peer_config = _config(args.peer_replay / "run_summary.json", "peer")
    canonical_period = float(canonical_config["frame_period_s"])
    peer_period = float(peer_config["frame_period_s"])
    if not math.isclose(canonical_period, peer_period, rel_tol=0.0, abs_tol=1e-9):
        parser.error("canonical and peer replay frame periods differ")
    frame_period = args.frame_period_s if args.frame_period_s is not None else canonical_period
    peer_delay = args.peer_delay_s if args.peer_delay_s is not None else float(canonical_config["peer_delay_s"])
    association_gate = (
        args.association_gate_m
        if args.association_gate_m is not None
        else float(canonical_config["association_gate_m"])
    )

    try:
        states = _read_jsonl(args.canonical_replay / "receiver_target_states.jsonl")
        canonical_tracks = _read_jsonl(args.canonical_replay / "local_tracks.jsonl")
        peer_tracks = _read_jsonl(args.peer_replay / "local_tracks.jsonl")
        _validate_target_state_coverage(states, canonical_tracks)
        rows, summary = build_cross_replay_pairings(
            states,
            peer_tracks,
            frame_period_s=frame_period,
            peer_delay_s=peer_delay,
            association_gate_m=association_gate,
        )
    except (CrossReplayContractError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
        return

    output = args.output or (args.canonical_replay / "cross_replay_pairing.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, rows)
    summary_path = args.summary or output.with_name(output.stem + "_summary.json")
    summary.update(
        {
            "canonical_replay": str(args.canonical_replay),
            "peer_replay": str(args.peer_replay),
            "canonical_target_state_rows": len(states),
            "canonical_track_rows": len(canonical_tracks),
            "peer_track_rows": len(peer_tracks),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("pairing=%s" % output)
    print("summary=%s" % summary_path)
    for key, value in sorted(summary.items()):
        if key != "association_cost_m":
            print("%s=%s" % (key, value))


if __name__ == "__main__":
    main()
