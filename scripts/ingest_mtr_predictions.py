#!/usr/bin/env python
"""Attach normalized real MTR forecasts to a derived GT-free ledger.

The base ledger is immutable.  This command writes a sibling JSONL with MTR
peer/ego/aggregate payloads and a small attachment audit.  Official CMP/MTR
``generate_prediction_dicts`` files are intentionally rejected when they still
contain ``object_id``/GT fields or lack a source-local track-id sidecar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rvhca_cpu.mtr_io import (
    MTRContractError,
    iter_prediction_records,
    merge_predictions_into_ledger_stream,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir", type=Path, help="GT-free replay directory")
    parser.add_argument(
        "predictions",
        type=Path,
        nargs="+",
        help="normalized MTR JSON/JSONL/Pickle file(s) or directories",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    replay_dir = args.replay_dir
    output_dir = args.output_dir or replay_dir.with_name(replay_dir.name + "_mtr")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_ledger = output_dir / "prediction_ledger.jsonl"
    try:
        summary = merge_predictions_into_ledger_stream(
            replay_dir / "prediction_ledger.jsonl",
            iter_prediction_records(args.predictions),
            output_ledger,
        )
    except (MTRContractError, ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
        return

    with (output_dir / "mtr_attachment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print("output_dir=%s" % output_dir)
    print("ledger=%s" % (output_dir / "prediction_ledger.jsonl"))
    print("summary=%s" % (output_dir / "mtr_attachment_summary.json"))
    for key, value in sorted(summary.items()):
        print("%s=%s" % (key, value))


if __name__ == "__main__":
    main()
