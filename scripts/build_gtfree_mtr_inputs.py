#!/usr/bin/env python
"""Build a GT-free source-side MTR input manifest from local tracker rows.

This is deliberately only the identity/time contract.  A GPU wrapper may use
the manifest to construct the official MTR tensors and must emit normalized
prediction records carrying the same ``source_track_id``.  No future labels,
GT IDs, or cross-source matching are read here.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_records(
    rows: Sequence[Mapping[str, Any]],
    past_frames: int = 10,
    min_history_valid_ratio: float = 1.0,
) -> Iterable[Dict[str, Any]]:
    grouped: MutableMapping[Tuple[str, str, int], Mapping[str, Any]] = {}
    frame_ids: MutableMapping[Tuple[str, str], List[int]] = defaultdict(list)
    for row in rows:
        key = (str(row["sequence_id"]), str(row["source"]), int(row["frame_idx"]))
        # One local ID can have at most one emitted row per source/frame.  If a
        # malformed artifact violates this, fail rather than choosing silently.
        track_key = key + (int(row["local_track_id"]),)
        if track_key in grouped:
            raise ValueError("duplicate local track row for %s" % (track_key,))
        grouped[track_key] = row
        if key not in frame_ids[(key[0], key[1])]:
            frame_ids[(key[0], key[1])].append(key[2])
    for key in frame_ids:
        frame_ids[key].sort()

    for (scene, source), frames in sorted(frame_ids.items()):
        local_ids = sorted({track_id for (s, so, _, track_id) in grouped if s == scene and so == source})
        for frame_idx in frames:
            history_indices = list(range(frame_idx - int(past_frames), frame_idx + 1))
            for local_id in local_ids:
                history = []
                for history_frame in history_indices:
                    item = grouped.get((scene, source, history_frame, local_id))
                    if item is None:
                        history.append({"frame_idx": history_frame, "valid": False})
                    else:
                        history.append(
                            {
                                "frame_idx": history_frame,
                                "valid": True,
                                "center_local": list(item["center_local"]),
                                "dims_hwl": list(item["dims_hwl"]),
                                "yaw_local": float(item["yaw_local"]),
                                "score": float(item.get("score", 0.0)),
                            }
                        )
                if not history or not history[-1].get("valid"):
                    continue
                valid_count = sum(bool(item.get("valid")) for item in history)
                if valid_count / len(history) < float(min_history_valid_ratio):
                    continue
                yield {
                    "schema_version": "rvhca.mtr_input.v0",
                    "sequence_id": scene,
                    "source": source,
                    "source_track_id": str(local_id),
                    "send_frame_idx": frame_idx,
                    "send_time": frame_idx * 0.1,
                    "past_frames": int(past_frames),
                    "history_valid_ratio": valid_count / len(history),
                    "history": history,
                    "source_pose_at_send": list(grouped[(scene, source, frame_idx, local_id)]["pose"]),
                    "uses_gt": False,
                }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("local_tracks", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--past-frames", type=int, default=10)
    parser.add_argument("--min-history-valid-ratio", type=float, default=1.0)
    args = parser.parse_args()
    rows = read_jsonl(args.local_tracks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for record in iter_records(rows, args.past_frames, args.min_history_valid_ratio):
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    print("output=%s" % args.output)
    print("input_track_rows=%d" % len(rows))
    print("mtr_input_records=%d" % count)


if __name__ == "__main__":
    main()
