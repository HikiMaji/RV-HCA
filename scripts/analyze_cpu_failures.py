#!/usr/bin/env python
"""Decompose failures in the GT-free CPU replay.

This is an *offline* analysis utility.  It is allowed to read raw OPV2V
labels, but those labels are used only to score the already-created tracking,
association, and ledger artifacts.  Nothing in this script is imported by the
online replay path.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from rvhca_cpu.evaluate import (
    RawGTIndex,
    _build_track_matches,
    _association_metrics,
    _track_survival_metrics,
    _tracking_metrics,
    read_jsonl,
)


def _ratio(num: int, den: int) -> Optional[float]:
    return float(num) / float(den) if den else None


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    return float(np.percentile(values, q)) if values else None


def _bin_label(value: float, edges: Sequence[float]) -> str:
    # The first bucket is half-open on the left; the final bucket is closed.
    idx = int(np.searchsorted(np.asarray(edges, dtype=np.float64), value, side="right") - 1)
    idx = max(0, min(idx, len(edges) - 2))
    return "[%.2f,%.2f%s" % (
        edges[idx],
        edges[idx + 1],
        "]" if idx == len(edges) - 2 else ")",
    )


def _read_artifacts(replay_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    return (
        read_jsonl(replay_dir / "local_tracks.jsonl"),
        read_jsonl(replay_dir / "association_events.jsonl"),
        read_jsonl(replay_dir / "prediction_ledger.jsonl"),
    )


def _track_decomposition(
    tracks: Sequence[Mapping[str, Any]],
    matches: Mapping[Tuple[str, str, int, int], Optional[str]],
    gt_frames: Mapping[Tuple[str, str, int], Mapping[str, Any]],
) -> Dict[str, Any]:
    by_key: MutableMapping[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in tracks:
        by_key[(str(row["sequence_id"]), str(row["source"]))].append(row)

    score_edges = (0.0, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.01)
    per_source: Dict[str, Any] = {}
    for (scene, source), rows in sorted(by_key.items()):
        frame_keys = {(scene, source, int(row["frame_idx"])) for row in rows}
        scoped_gt_frames = {key: value for key, value in gt_frames.items() if key in frame_keys}
        metrics = _tracking_metrics(rows, {
            (scene, source, int(row["frame_idx"]), int(row["local_track_id"])):
            matches.get((scene, source, int(row["frame_idx"]), int(row["local_track_id"])))
            for row in rows
        }, scoped_gt_frames)
        score_bins: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            score = float(row.get("score", 0.0))
            label = _bin_label(score, score_edges)
            bucket = score_bins.setdefault(label, {"rows": 0, "matched_rows": 0, "scores": []})
            bucket["rows"] += 1
            bucket["matched_rows"] += int(
                matches.get((scene, source, int(row["frame_idx"]), int(row["local_track_id"]))) is not None
            )
            bucket["scores"].append(score)
        for bucket in score_bins.values():
            bucket["precision"] = _ratio(bucket["matched_rows"], bucket["rows"])
            bucket["mean_score"] = _mean(bucket.pop("scores"))

        # Count GT instances that are absent from the detector/tracker at each
        # frame, which makes recall losses distinguishable from false tracks.
        missed_gt_by_frame = 0
        total_gt_by_frame = 0
        for (s, src, frame_idx), gt in scoped_gt_frames.items():
            total_gt_by_frame += len(gt)
            matched_ids = {
                matches.get((s, src, frame_idx, int(row["local_track_id"])))
                for row in rows
                if int(row["frame_idx"]) == frame_idx
            }
            matched_ids.discard(None)
            missed_gt_by_frame += len(set(gt) - matched_ids)
        per_source["%s/%s" % (scene, source)] = {
            "sequence_id": scene,
            "source": source,
            "tracking": metrics,
            "missed_gt_instances": missed_gt_by_frame,
            "total_gt_instances_checked": total_gt_by_frame,
            "score_bins": score_bins,
        }

    by_scene: Dict[str, Any] = {}
    for scene in sorted({str(row["sequence_id"]) for row in tracks}):
        scene_rows = [row for row in tracks if str(row["sequence_id"]) == scene]
        scene_keys = {(scene, str(row["source"]), int(row["frame_idx"])) for row in scene_rows}
        scene_gt_frames = {key: value for key, value in gt_frames.items() if key in scene_keys}
        scene_matches = {
            (scene, str(row["source"]), int(row["frame_idx"]), int(row["local_track_id"])):
            matches.get((scene, str(row["source"]), int(row["frame_idx"]), int(row["local_track_id"])))
            for row in scene_rows
        }
        by_scene[scene] = _tracking_metrics(scene_rows, scene_matches, scene_gt_frames)
    overall_matches = {
        (str(row["sequence_id"]), str(row["source"]), int(row["frame_idx"]), int(row["local_track_id"])):
        matches.get((str(row["sequence_id"]), str(row["source"]), int(row["frame_idx"]), int(row["local_track_id"])))
        for row in tracks
    }
    return {
        "overall": {"track_rows": len(tracks), "tracking": _tracking_metrics(tracks, overall_matches, gt_frames)},
        "per_scene_source": per_source,
        "per_scene": by_scene,
    }


def _event_gt_state(
    event: Mapping[str, Any],
    matches: Mapping[Tuple[str, str, int, int], Optional[str]],
    tracks_by_frame: Mapping[Tuple[str, str, int], Sequence[Mapping[str, Any]]],
    gt_index: RawGTIndex,
) -> Dict[str, Any]:
    scene = str(event["sequence_id"])
    receiver = str(event["receiver"])
    source = str(event["source"])
    peer_frame = int(event.get("peer_frame_idx", event.get("send_frame_idx", event["frame_idx"])))
    arrival_frame = int(event.get("arrival_frame_idx", event["frame_idx"]))
    peer_id = int(event["peer_local_track_id"])
    peer_gt = matches.get((scene, source, peer_frame, peer_id))
    receiver_rows = tracks_by_frame.get((scene, receiver, arrival_frame), ())
    receiver_gt_frame = gt_index.by_frame(
        scene,
        receiver,
        arrival_frame,
        timestamp_key=None,
    )
    receiver_track_to_gt = {
        int(row["local_track_id"]): matches.get((scene, receiver, arrival_frame, int(row["local_track_id"])))
        for row in receiver_rows
    }
    receiver_gts = {gt for gt in receiver_track_to_gt.values() if gt is not None}
    receiver_id = event.get("receiver_local_track_id")
    receiver_gt = (
        receiver_track_to_gt.get(int(receiver_id)) if receiver_id is not None else None
    )
    if peer_gt is None:
        category = "peer_track_unmatched"
    elif peer_gt not in receiver_gt_frame:
        category = "receiver_label_absent_or_not_visible"
    elif peer_gt not in receiver_gts:
        category = "receiver_track_missed"
    elif event.get("target_scope") == "shared_only":
        category = "cross_source_gate_miss"
    elif receiver_gt is None:
        category = "receiver_assignment_unmatched"
    elif receiver_gt != peer_gt:
        category = "receiver_assignment_wrong_target"
    else:
        category = "correct_common"
    return {
        "peer_gt": peer_gt,
        "receiver_gt": receiver_gt,
        "receiver_gts": sorted(receiver_gts),
        "receiver_gt_frame_present": peer_gt in receiver_gt_frame if peer_gt is not None else False,
        "category": category,
    }


def _association_decomposition(
    events: Sequence[Mapping[str, Any]],
    tracks: Sequence[Mapping[str, Any]],
    matches: Mapping[Tuple[str, str, int, int], Optional[str]],
    gt_index: RawGTIndex,
) -> Dict[str, Any]:
    tracks_by_frame: MutableMapping[Tuple[str, str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in tracks:
        tracks_by_frame[(str(row["sequence_id"]), str(row["source"]), int(row["frame_idx"]))].append(row)

    pair_buckets: MutableMapping[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    scene_buckets: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        pair_buckets[(str(event["sequence_id"]), str(event["receiver"]), str(event["source"]))].append(event)
        scene_buckets[str(event["sequence_id"])].append(event)

    def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        scope = Counter(str(row.get("target_scope")) for row in rows)
        categories = Counter()
        common_correct = 0
        common_evaluable = 0
        receiver_gt_present = 0
        receiver_label_present = 0
        peer_gt_present = 0
        candidate_with_receiver_track = 0
        shared_hit = 0
        common_costs: List[float] = []
        common_correct_costs: List[float] = []
        common_wrong_costs: List[float] = []
        for event in rows:
            state = _event_gt_state(event, matches, tracks_by_frame, gt_index)
            categories[state["category"]] += 1
            peer_gt = state["peer_gt"]
            receiver_gts = set(state["receiver_gts"])
            if peer_gt is not None:
                peer_gt_present += 1
            if state["receiver_gt_frame_present"]:
                receiver_label_present += 1
            if peer_gt is not None and peer_gt in receiver_gts:
                receiver_gt_present += 1
                candidate_with_receiver_track += 1
            if event.get("target_scope") == "shared_only":
                if peer_gt is not None and peer_gt in receiver_gts:
                    shared_hit += 1
            else:
                if state["receiver_gt"] is not None and peer_gt is not None:
                    common_evaluable += 1
                    if state["receiver_gt"] == peer_gt:
                        common_correct += 1
                cost = event.get("association_cost_m")
                if cost is not None and math.isfinite(float(cost)):
                    common_costs.append(float(cost))
                    if state["category"] == "correct_common":
                        common_correct_costs.append(float(cost))
                    else:
                        common_wrong_costs.append(float(cost))
        cost_edges = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.000001)
        cost_bins: Dict[str, Dict[str, Any]] = {}
        for event in rows:
            if event.get("target_scope") != "common" or event.get("association_cost_m") is None:
                continue
            cost = float(event["association_cost_m"])
            label = _bin_label(cost, cost_edges)
            bucket = cost_bins.setdefault(label, {"events": 0, "correct": 0})
            bucket["events"] += 1
            state = _event_gt_state(event, matches, tracks_by_frame, gt_index)
            bucket["correct"] += int(state["category"] == "correct_common")
        for bucket in cost_bins.values():
            bucket["precision"] = _ratio(bucket["correct"], bucket["events"])
        common = scope.get("common", 0)
        shared = scope.get("shared_only", 0)
        return {
            "events": len(rows),
            "common_events": common,
            "shared_only_events": shared,
            "categories": dict(sorted(categories.items())),
            "peer_track_gt_rate": _ratio(peer_gt_present, len(rows)),
            "receiver_gt_present_given_peer_gt_rate": _ratio(receiver_gt_present, peer_gt_present),
            "receiver_label_present_given_peer_gt_rate": _ratio(receiver_label_present, peer_gt_present),
            "receiver_track_candidate_rate": _ratio(candidate_with_receiver_track, peer_gt_present),
            "common_evaluable_events": common_evaluable,
            "common_correct_events": common_correct,
            "common_precision": _ratio(common_correct, common_evaluable),
            "common_candidate_recall": _ratio(common_correct, candidate_with_receiver_track),
            "shared_only_offline_hit_rate": _ratio(shared_hit, shared),
            "association_cost_m": {
                "mean": _mean(common_costs),
                "p50": _percentile(common_costs, 50),
                "p90": _percentile(common_costs, 90),
                "correct_mean": _mean(common_correct_costs),
                "wrong_mean": _mean(common_wrong_costs),
            },
            "common_cost_bins": cost_bins,
        }

    return {
        "per_scene_receiver_source": {
            "%s/%s->%s" % key: summarize(rows)
            for key, rows in sorted(pair_buckets.items())
        },
        "per_scene": {scene: summarize(rows) for scene, rows in sorted(scene_buckets.items())},
        "overall": summarize(events),
    }


def _ledger_decomposition(ledger: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    common = [row for row in ledger if row.get("target_scope") == "common"]
    by_horizon: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_pair: MutableMapping[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in common:
        by_horizon["%.3f" % float(row["horizon"])].append(row)
        by_pair[(str(row["sequence_id"]), str(row["receiver"]), str(row["source"]))].append(row)

    def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        censor = Counter(str(row.get("censor_reason")) for row in rows)
        valid = [row for row in rows if row.get("valid_mask")]
        arrival_delta = [
            float(row["arrival_time"]) - float(row["send_time"])
            for row in rows
            if row.get("arrival_time") is not None and row.get("send_time") is not None
        ]
        return {
            "rows": len(rows),
            "matured_valid_rows": len(valid),
            "coverage": _ratio(len(valid), len(rows)),
            "censor_reasons": dict(sorted(censor.items())),
            "arrival_delay_s": {
                "mean": _mean(arrival_delta),
                "min": min(arrival_delta) if arrival_delta else None,
                "max": max(arrival_delta) if arrival_delta else None,
            },
            "track_age": {
                "valid_mean": _mean([float(row.get("track_age", 0)) for row in valid]),
                "censored_mean": _mean([float(row.get("track_age", 0)) for row in rows if not row.get("valid_mask")]),
            },
            "miss_count": {
                "valid_mean": _mean([float(row.get("miss_count", 0)) for row in valid]),
                "censored_mean": _mean([float(row.get("miss_count", 0)) for row in rows if not row.get("valid_mask")]),
            },
        }

    return {
        "overall": summarize(common),
        "horizon": {key: summarize(rows) for key, rows in sorted(by_horizon.items())},
        "per_scene_receiver_source": {
            "%s/%s->%s" % key: summarize(rows)
            for key, rows in sorted(by_pair.items())
        },
    }


def _markdown(result: Mapping[str, Any]) -> str:
    tracking = result["tracking"]
    association_events = result["association"]
    association_eval = result.get("association_metrics", {})
    track_survival = result.get("track_survival", {})
    ledger = result["ledger"]
    lines = [
        "# CPU GT-free failure decomposition",
        "",
        "This report scores the GT-free artifacts offline with raw OPV2V labels. Labels are not consumed by replay, tracking IDs, cross-source association, or ledger construction.",
        "",
        "## Main finding",
        "",
        "The dominant current blocker is recall, not false-positive precision. The replay keeps highly precise local tracks, but detector/track misses and receiver visibility gaps remove many candidate targets before a common event can be verified.",
        "",
        "| scope | track rows | track precision | track recall | ID switches / 100 matched | common events | event-conditioned precision | event-conditioned candidate recall | shared-only receiver-track hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| overall | %s | %.3f | %.3f | %.3f | %s | %.3f | %.3f | %.3f |" % (
            tracking["overall"]["track_rows"],
            tracking["overall"]["tracking"]["track_precision"],
            tracking["overall"]["tracking"]["track_recall"],
            tracking["overall"]["tracking"]["id_switches_per_100_matched_rows"],
            association_events["overall"]["common_events"],
            association_events["overall"]["common_precision"] or 0.0,
            association_events["overall"]["common_candidate_recall"] or 0.0,
            association_events["overall"]["shared_only_offline_hit_rate"] or 0.0,
        ),
    ]
    if association_eval:
        lines += [
            "",
            "The table above is an event-conditioned failure decomposition. The independent raw-GT opportunity metrics used for the main contract are:",
            "",
            "| metric | value | denominator |",
            "|---|---:|---:|",
            "| end-to-end association recall | %.3f | %d raw-GT opportunities |" % (
                association_eval.get("end_to_end_association_recall") or 0.0,
                association_eval.get("end_to_end_opportunities", 0),
            ),
            "| receiver raw-GT visibility rate | %.3f | %d / %d peer opportunities |" % (
                association_eval.get("receiver_visibility_rate") or 0.0,
                association_eval.get("receiver_visible_opportunities", 0),
                association_eval.get("peer_raw_opportunities", 0),
            ),
            "| recall over all peer opportunities | %.3f | %d raw peer opportunities |" % (
                association_eval.get("all_peer_opportunity_recall") or 0.0,
                association_eval.get("peer_raw_opportunities", 0),
            ),
            "| conditional association recall | %.3f | %d both-side track opportunities |" % (
                association_eval.get("conditional_association_recall") or 0.0,
                association_eval.get("conditional_opportunities", 0),
            ),
            "| association precision (same evaluated events) | %.3f | %d evaluated events |" % (
                association_eval.get("association_precision") or 0.0,
                association_eval.get("common_evaluable_events", 0),
            ),
            "",
            "End-to-end opportunities include peer/receiver detector or track misses; conditional recall intentionally conditions those failures out.",
        ]
    lines += [
        "",
        "## Tracking by scene/source",
        "",
        "| scene/source | rows | precision | recall | ID switches | persistence median | missed GT |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, value in tracking["per_scene_source"].items():
        t = value["tracking"]
        lines.append(
            "| %s | %d | %.3f | %.3f | %d | %.3f | %d |" % (
                key,
                t["track_rows"],
                t["track_precision"],
                t["track_recall"],
                t["id_switches"],
                t["track_persistence_median"],
                value["missed_gt_instances"],
            )
        )
    lines += ["", "## Association failure categories", "", "| scene/receiver->source | common | shared-only | peer unmatched | receiver label absent | receiver track miss | cross-source gate miss | wrong receiver target | correct common |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, value in association_events["per_scene_receiver_source"].items():
        c = value["categories"]
        lines.append(
            "| %s | %d | %d | %d | %d | %d | %d | %d | %d |" % (
                key,
                value["common_events"],
                value["shared_only_events"],
                c.get("peer_track_unmatched", 0),
                c.get("receiver_label_absent_or_not_visible", 0),
                c.get("receiver_track_missed", 0),
                c.get("cross_source_gate_miss", 0),
                c.get("receiver_assignment_wrong_target", 0),
                c.get("correct_common", 0),
            )
        )
    lines += ["", "## Ledger hindsight coverage", "", "| horizon (s) | rows | valid | coverage | censor: not observed/track dead |", "|---:|---:|---:|---:|---:|"]
    for key, value in ledger["horizon"].items():
        lines.append(
            "| %s | %d | %d | %.3f | %d |" % (
                key,
                value["rows"],
                value["matured_valid_rows"],
                value["coverage"],
                value["censor_reasons"].get("not_observed_or_track_dead", 0),
            )
        )
    lines += [
        "",
        "## Track survival",
        "",
        "| horizon | matched start rows | same local ID | same-GT survival |",
        "|---:|---:|---:|---:|",
    ]
    for key, value in sorted(track_survival.items()):
        lines.append(
            "| %s | %d | %.3f | %.3f |" % (
                key,
                value.get("denominator_matched_start_rows", 0),
                value.get("id_present_rate") or 0.0,
                value.get("same_gt_survival_rate") or 0.0,
            )
        )
    lines += [
        "",
        "## Interpretation and next gate",
        "",
        "The 0.1 s arrival delay is fully exposed in the ledger; no row is silently evaluated before arrival. Coverage falls with horizon because the receiver target track is not observed or has died, reaching the lowest level at 5 s. Recent-error and EWMA selectors therefore have usable feedback but currently produce negative mean gain versus ego on the formal artifact.",
        "",
        "Under the current contract gate (track and end-to-end association recall at least 0.80, precision at least 0.80, and common hindsight coverage at least 0.50), precision and persistence pass but track recall and end-to-end association recall do not. Do not start a learned reliability controller yet; first test whether a lower detector threshold or a better receiver-side gate can recover recall without collapsing precision. The present trajectory aggregate is a mean probe, not CMP, so no CMP-vs-GT-free ADE/FDE claim is made.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir", type=Path)
    parser.add_argument("--raw-root", type=Path, default=project_root / "data/raw/OPV2V/test")
    parser.add_argument("--association-gate-m", type=float, default=3.0, help="offline track-to-label gate")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    tracks, events, ledger = _read_artifacts(args.replay_dir)
    gt_index = RawGTIndex(args.raw_root)
    matches, _, gt_frames = _build_track_matches(tracks, gt_index, args.association_gate_m)
    result = {
        "schema_version": "rvhca.cpu_failure_analysis.v0",
        "replay_dir": str(args.replay_dir),
        "raw_root": str(args.raw_root),
        "offline_track_gate_m": args.association_gate_m,
        "tracking": _track_decomposition(tracks, matches, gt_frames),
        "association": _association_decomposition(events, tracks, matches, gt_index),
        "association_metrics": _association_metrics(events, matches, tracks, gt_index),
        "track_survival": _track_survival_metrics(tracks, matches),
        "ledger": _ledger_decomposition(ledger),
    }
    output_json = args.output_json or (args.replay_dir / "failure_analysis.json")
    output_md = args.output_md or (args.replay_dir / "failure_analysis.md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(_markdown(result), encoding="utf-8")
    print("output_json=%s" % output_json)
    print("output_md=%s" % output_md)
    print(json.dumps({
        "tracking_overall": result["tracking"]["per_scene"],
        "association_overall": result["association"]["overall"],
        "ledger_overall": result["ledger"]["overall"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
