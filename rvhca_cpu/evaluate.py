"""Independent offline evaluator for the GT-free CPU artifacts.

Only this module reads raw OPV2V labels.  It is intentionally not imported by
``rvhca_cpu.online`` or the online replay script.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from rvhca_cpu.online import pose_to_world_matrix, transform_point


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Stream JSONL rows for large derived ledgers."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class RawGTIndex:
    """Lazy raw-YAML index used only by the offline evaluator."""

    def __init__(self, raw_root: Path) -> None:
        self.raw_root = Path(raw_root)
        self._cache: Dict[Tuple[str, str, str], Dict[str, np.ndarray]] = {}
        self._files: Dict[Tuple[str, str], List[Path]] = {}

    def files(self, sequence_id: str, source: str) -> List[Path]:
        key = (sequence_id, str(source))
        if key not in self._files:
            directory = self.raw_root / sequence_id / str(source)
            self._files[key] = sorted(directory.glob("*.yaml"), key=lambda p: p.stem)
        return self._files[key]

    def at(self, sequence_id: str, source: str, timestamp_key: str) -> Dict[str, np.ndarray]:
        key = (sequence_id, str(source), str(timestamp_key))
        if key in self._cache:
            return self._cache[key]
        path = self.raw_root / sequence_id / str(source) / (str(timestamp_key) + ".yaml")
        if not path.exists():
            self._cache[key] = {}
            return self._cache[key]
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        vehicles = payload.get("vehicles", {}) or {}
        result: Dict[str, np.ndarray] = {}
        for vehicle_id, vehicle in vehicles.items():
            if not isinstance(vehicle, Mapping):
                continue
            location = np.asarray(vehicle.get("location", [np.nan] * 3), dtype=np.float64)
            center = np.asarray(vehicle.get("center", [0.0] * 3), dtype=np.float64)
            if location.size != 3 or center.size != 3 or not np.isfinite(location + center).all():
                continue
            result[str(vehicle_id)] = location + center
        self._cache[key] = result
        return result

    def by_frame(self, sequence_id: str, source: str, frame_idx: int, timestamp_key: Optional[str] = None) -> Dict[str, np.ndarray]:
        if timestamp_key is None:
            files = self.files(sequence_id, source)
            if frame_idx < 0 or frame_idx >= len(files):
                return {}
            timestamp_key = files[frame_idx].stem
        return self.at(sequence_id, source, str(timestamp_key))


def _match_centers(
    tracks: Sequence[Mapping[str, Any]],
    gt: Mapping[str, np.ndarray],
    gate_m: float,
) -> Dict[int, Tuple[Optional[str], Optional[float]]]:
    if not tracks or not gt:
        return {index: (None, None) for index in range(len(tracks))}
    gt_ids = sorted(gt)
    track_centers = np.asarray([row["center_world"] for row in tracks], dtype=np.float64)
    gt_centers = np.asarray([gt[item] for item in gt_ids], dtype=np.float64)
    cost = np.linalg.norm(track_centers[:, None, :] - gt_centers[None, :, :], axis=2)
    rr, cc = linear_sum_assignment(cost)
    result: Dict[int, Tuple[Optional[str], Optional[float]]] = {
        index: (None, None) for index in range(len(tracks))
    }
    for r, c in zip(rr.tolist(), cc.tolist()):
        distance = float(cost[r, c])
        if distance <= gate_m:
            result[r] = (gt_ids[c], distance)
    return result


def _group_tracks(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str, int], List[Mapping[str, Any]]]:
    grouped: MutableMapping[Tuple[str, str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["sequence_id"]), str(row["source"]), int(row["frame_idx"]))].append(row)
    return dict(grouped)


def _build_track_matches(
    tracks: Sequence[Mapping[str, Any]],
    gt_index: RawGTIndex,
    gate_m: float,
) -> Tuple[
    Dict[Tuple[str, str, int, int], Optional[str]],
    Dict[Tuple[str, str, int, int], Optional[float]],
    Dict[Tuple[str, str, int], Dict[str, np.ndarray]],
]:
    grouped = _group_tracks(tracks)
    matches: Dict[Tuple[str, str, int, int], Optional[str]] = {}
    distances: Dict[Tuple[str, str, int, int], Optional[float]] = {}
    gt_frames: Dict[Tuple[str, str, int], Dict[str, np.ndarray]] = {}
    for (sequence_id, source, frame_idx), frame_tracks in grouped.items():
        timestamp_key = str(frame_tracks[0].get("timestamp_key", ""))
        gt = gt_index.at(sequence_id, source, timestamp_key)
        if not gt:
            gt = gt_index.by_frame(sequence_id, source, frame_idx, timestamp_key=None)
        gt_frames[(sequence_id, source, frame_idx)] = gt
        row_matches = _match_centers(frame_tracks, gt, gate_m)
        for local_row_index, row in enumerate(frame_tracks):
            gt_id, distance = row_matches[local_row_index]
            key = (sequence_id, source, frame_idx, int(row["local_track_id"]))
            matches[key] = gt_id
            distances[key] = distance
    return matches, distances, gt_frames


def _tracking_metrics(
    tracks: Sequence[Mapping[str, Any]],
    matches: Mapping[Tuple[str, str, int, int], Optional[str]],
    gt_frames: Mapping[Tuple[str, str, int], Mapping[str, np.ndarray]],
) -> Dict[str, Any]:
    total_track_rows = len(tracks)
    matched_rows = 0
    total_gt_instances = sum(len(frame) for frame in gt_frames.values())
    matched_gt_instances: set = set()
    gt_to_tracks: MutableMapping[Tuple[str, str, str], List[Tuple[int, int]]] = defaultdict(list)
    track_span: MutableMapping[Tuple[str, str, int], List[int]] = defaultdict(list)
    for row in tracks:
        scene, source, frame_idx, local_id = (
            str(row["sequence_id"]),
            str(row["source"]),
            int(row["frame_idx"]),
            int(row["local_track_id"]),
        )
        key = (scene, source, frame_idx, local_id)
        gt_id = matches.get(key)
        track_span[(scene, source, local_id)].append(frame_idx)
        if gt_id is not None:
            matched_rows += 1
            matched_gt_instances.add((scene, source, frame_idx, gt_id))
            gt_to_tracks[(scene, source, gt_id)].append((frame_idx, local_id))
    id_switches = 0
    for assignments in gt_to_tracks.values():
        assignments.sort()
        previous_id: Optional[int] = None
        for _, local_id in assignments:
            if previous_id is not None and local_id != previous_id:
                id_switches += 1
            previous_id = local_id
    persistence = []
    for frame_list in track_span.values():
        unique = sorted(set(frame_list))
        span = max(unique) - min(unique) + 1
        persistence.append(len(unique) / max(span, 1))
    return {
        "track_rows": total_track_rows,
        "matched_track_rows": matched_rows,
        "track_precision": matched_rows / total_track_rows if total_track_rows else 0.0,
        "gt_instances": total_gt_instances,
        "matched_gt_instances": len(matched_gt_instances),
        "track_recall": len(matched_gt_instances) / total_gt_instances if total_gt_instances else 0.0,
        "id_switches": id_switches,
        "id_switches_per_100_matched_rows": id_switches * 100.0 / matched_rows if matched_rows else 0.0,
        "track_persistence_mean": float(np.mean(persistence)) if persistence else None,
        "track_persistence_median": float(np.median(persistence)) if persistence else None,
        "track_persistence_p10": float(np.percentile(persistence, 10)) if persistence else None,
    }


def _track_survival_metrics(
    tracks: Sequence[Mapping[str, Any]],
    matches: Mapping[Tuple[str, str, int, int], Optional[str]],
    frame_period_s: float = 0.1,
    horizons_s: Sequence[float] = (1.0, 2.0, 3.0, 5.0),
) -> Dict[str, Any]:
    """Measure exact-frame survival of a matched local track.

    A denominator is a track row with a valid offline GT match for which the
    same source/sequence has a frame at ``t + horizon``.  ``id_present`` only
    asks whether the same local tracker ID is emitted then; ``same_gt`` also
    requires that it still matches the original GT object.  The latter exposes
    ID-switches and accidental track recycling.  Objects leaving the recorded
    sequence or sensor range are not hidden: they remain failures whenever a
    future frame exists in the sequence.
    """

    by_key: MutableMapping[Tuple[str, str, int], List[Mapping[str, Any]]] = defaultdict(list)
    frame_sets: MutableMapping[Tuple[str, str], set] = defaultdict(set)
    for row in tracks:
        scene = str(row["sequence_id"])
        source = str(row["source"])
        frame = int(row["frame_idx"])
        by_key[(scene, source, frame)].append(row)
        frame_sets[(scene, source)].add(frame)

    result: Dict[str, Any] = {}
    for horizon_s in horizons_s:
        horizon = float(horizon_s)
        key = "%.1fs" % horizon
        offset = int(round(horizon / frame_period_s))
        denominator = 0
        id_present = 0
        same_gt = 0
        for (scene, source, frame), rows in by_key.items():
            future_frame = frame + offset
            if future_frame not in frame_sets[(scene, source)]:
                continue
            for row in rows:
                local_id = int(row["local_track_id"])
                base_gt = matches.get((scene, source, frame, local_id))
                if base_gt is None:
                    continue
                denominator += 1
                future_rows = [
                    item
                    for item in by_key.get((scene, source, future_frame), [])
                    if int(item["local_track_id"]) == local_id
                ]
                if future_rows:
                    id_present += 1
                    future_gt = matches.get((scene, source, future_frame, local_id))
                    if future_gt == base_gt:
                        same_gt += 1
        result[key] = {
            "horizon_s": horizon,
            "denominator_matched_start_rows": denominator,
            "same_local_id_rows": id_present,
            "same_gt_rows": same_gt,
            "id_present_rate": id_present / denominator if denominator else None,
            "same_gt_survival_rate": same_gt / denominator if denominator else None,
            "definition": "matched start row -> exact frame t+h; same_gt additionally preserves offline GT match",
        }
    return result


def _association_metrics(
    events: Sequence[Mapping[str, Any]],
    matches: Mapping[Tuple[str, str, int, int], Optional[str]],
    tracks: Sequence[Mapping[str, Any]],
    gt_index: RawGTIndex,
) -> Dict[str, Any]:
    by_frame: MutableMapping[Tuple[str, str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in tracks:
        by_frame[(str(row["sequence_id"]), str(row["source"]), int(row["frame_idx"]))].append(row)
    common = [event for event in events if event.get("target_scope") == "common"]
    shared_only = [event for event in events if event.get("target_scope") == "shared_only"]
    correct_keys = set()
    conditional_keys = set()
    shared_with_receiver_gt = 0
    receiver_gt_unavailable = 0

    # Enumerate raw-GT opportunities independently of detector/tracker rows.
    # This is the important distinction between end-to-end recall and the
    # conditional metric below: a peer detector miss must remain in the
    # denominator.  We use the replay's observed frame range so a short probe
    # does not accidentally include the remainder of the raw scene.
    sequence_ids = {str(row["sequence_id"]) for row in tracks}
    sequence_ids.update(str(event["sequence_id"]) for event in events)
    max_frame: Dict[str, int] = defaultdict(lambda: -1)
    for row in tracks:
        max_frame[str(row["sequence_id"])] = max(max_frame[str(row["sequence_id"])], int(row["frame_idx"]))
    for event in events:
        scene = str(event["sequence_id"])
        max_frame[scene] = max(max_frame[scene], int(event.get("send_frame_idx", event.get("frame_idx", -1))))
    sources_by_scene: MutableMapping[str, set] = defaultdict(set)
    for row in tracks:
        sources_by_scene[str(row["sequence_id"])].add(str(row["source"]))
    for event in events:
        sources_by_scene[str(event["sequence_id"])].update(
            {str(event["receiver"]), str(event["source"])}
        )
    for scene in sorted(sequence_ids):
        scene_dir = gt_index.raw_root / scene
        if scene_dir.exists():
            sources_by_scene[scene].update(
                item.name for item in scene_dir.iterdir() if item.is_dir()
            )
    offset_values = [
        int(event.get("arrival_frame_idx", event.get("frame_idx", 0))) - int(event.get("send_frame_idx", 0))
        for event in events
        if event.get("arrival_frame_idx") is not None or event.get("frame_idx") is not None
    ]
    arrival_offset = int(round(float(np.median(offset_values)))) if offset_values else 1
    all_peer_opportunities = set()
    end_to_end_opportunities = set()
    receiver_invisible_opportunities = set()
    peer_track_misses = 0
    receiver_track_misses = 0
    for scene in sorted(sequence_ids):
        for receiver in sorted(sources_by_scene[scene]):
            receiver_dir = gt_index.raw_root / scene / receiver
            if not receiver_dir.exists():
                continue
            for source in sorted(sources_by_scene[scene]):
                if source == receiver:
                    continue
                for send_frame in range(max_frame.get(scene, -1) + 1):
                    peer_gt_all = gt_index.by_frame(scene, source, send_frame)
                    receiver_gt_all = gt_index.by_frame(scene, receiver, send_frame + arrival_offset)
                    if not peer_gt_all:
                        continue
                    receiver_gt_ids = set(receiver_gt_all)
                    for peer_gt in sorted(set(peer_gt_all)):
                        opportunity_key = (scene, receiver, source, send_frame, peer_gt)
                        all_peer_opportunities.add(opportunity_key)
                        if peer_gt not in receiver_gt_ids:
                            receiver_invisible_opportunities.add(opportunity_key)
                            continue
                        opportunity = (scene, receiver, source, send_frame, peer_gt)
                        end_to_end_opportunities.add(opportunity)
                        peer_rows = [
                            row
                            for row in by_frame.get((scene, source, send_frame), [])
                            if matches.get((scene, source, send_frame, int(row["local_track_id"]))) == peer_gt
                        ]
                        receiver_rows = [
                            row
                            for row in by_frame.get((scene, receiver, send_frame + arrival_offset), [])
                            if matches.get((scene, receiver, send_frame + arrival_offset, int(row["local_track_id"]))) == peer_gt
                        ]
                        if not peer_rows:
                            peer_track_misses += 1
                        if not receiver_rows:
                            receiver_track_misses += 1

    for event in events:
        scene = str(event["sequence_id"])
        receiver = str(event["receiver"])
        source = str(event["source"])
        frame = int(event["frame_idx"])
        peer_frame = int(event.get("peer_frame_idx", frame))
        receiver_frame = int(event.get("arrival_frame_idx", frame))
        peer_id = int(event["peer_local_track_id"])
        peer_gt = matches.get((scene, source, peer_frame, peer_id))
        receiver_gt_all = gt_index.by_frame(scene, receiver, receiver_frame)
        receiver_gts = {
            matches.get((scene, receiver, receiver_frame, int(row["local_track_id"])))
            for row in by_frame.get((scene, receiver, receiver_frame), [])
        }
        receiver_gts.discard(None)
        if peer_gt is not None and peer_gt in receiver_gt_all:
            if event.get("target_scope") == "shared_only":
                shared_with_receiver_gt += 1
        elif peer_gt is not None:
            receiver_gt_unavailable += 1
        if event.get("target_scope") != "common":
            continue
        receiver_id = event.get("receiver_local_track_id")
        receiver_gt = (
            matches.get((scene, receiver, receiver_frame, int(receiver_id))) if receiver_id is not None else None
        )
        if peer_gt is not None and receiver_gt is not None:
            key = (scene, receiver, source, int(event.get("send_frame_idx", peer_frame)), peer_gt)
            conditional_keys.add(key)
            if peer_gt == receiver_gt:
                correct_keys.add(key)
    end_to_end_denominator = len(end_to_end_opportunities)
    conditional_denominator = len(conditional_keys)
    correct = len(correct_keys)
    evaluable = conditional_denominator
    return {
        "common_events": len(common),
        "common_evaluable_events": evaluable,
        "association_precision": correct / evaluable if evaluable else None,
        "association_recall": correct / end_to_end_denominator if end_to_end_denominator else None,
        "end_to_end_association_recall": (
            correct / end_to_end_denominator if end_to_end_denominator else None
        ),
        "conditional_association_recall": (
            correct / conditional_denominator if conditional_denominator else None
        ),
        "correct_common_events": correct,
        "peer_raw_opportunities": len(all_peer_opportunities),
        "receiver_visible_opportunities": len(end_to_end_opportunities),
        "receiver_invisible_opportunities": len(receiver_invisible_opportunities),
        "receiver_visibility_rate": (
            len(end_to_end_opportunities) / len(all_peer_opportunities)
            if all_peer_opportunities else None
        ),
        "all_peer_opportunity_recall": (
            correct / len(all_peer_opportunities) if all_peer_opportunities else None
        ),
        "offline_common_candidate_denominator": end_to_end_denominator,
        "end_to_end_opportunities": end_to_end_denominator,
        "conditional_opportunities": conditional_denominator,
        "receiver_track_misses_on_opportunity": receiver_track_misses,
        "peer_track_misses_on_opportunity": peer_track_misses,
        "receiver_gt_unavailable_for_peer": receiver_gt_unavailable,
        "shared_only_events": len(shared_only),
        "shared_only_with_receiver_gt": shared_with_receiver_gt,
        "shared_only_offline_hit_rate": (
            shared_with_receiver_gt / len(shared_only) if shared_only else None
        ),
        "end_to_end_definition": (
            "correct common association / all raw-GT opportunities where the peer "
            "object is present in the receiver at packet arrival (receiver-visible "
            "opportunity; raw peer targets invisible at the receiver are reported "
            "separately)"
        ),
        "conditional_definition": (
            "correct common association / opportunities where peer and receiver "
            "local tracks both have independent offline GT matches"
        ),
        "shared_only_definition": (
            "peer forecast rows whose target is present in receiver raw GT at "
            "arrival; this is coverage only and is not a common-target recall"
        ),
    }


def _position_from_forecast(value: Any, horizon_s: Optional[float] = None) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if value.get("position_xyz") is not None:
            value = value.get("position_xyz")
        elif value.get("trajectories_xyz") is not None:
            trajectories = np.asarray(value.get("trajectories_xyz"), dtype=np.float64)
            if trajectories.ndim != 3 or trajectories.shape[0] == 0 or trajectories.shape[1] == 0:
                return None
            scores = np.asarray(value.get("scores", np.ones(trajectories.shape[0])), dtype=np.float64).reshape(-1)
            mode = int(np.argmax(scores[: trajectories.shape[0]])) if len(scores) else 0
            offsets = np.asarray(value.get("time_offsets_s", []), dtype=np.float64).reshape(-1)
            timestep = trajectories.shape[1] - 1
            if horizon_s is not None and len(offsets) == trajectories.shape[1]:
                timestep = int(np.argmin(np.abs(offsets - float(horizon_s))))
            value = trajectories[mode, timestep]
        else:
            value = None
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (3,) and np.isfinite(array).all():
        return array
    if array.ndim == 2 and array.shape[1] == 3 and len(array) > 0 and np.isfinite(array[-1]).all():
        return array[-1]
    return None


def _mtr_trajectory_payload(value: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return full MTR trajectories and normalized mode scores when present."""
    if not isinstance(value, Mapping) or value.get("trajectories_xyz") is None:
        return None
    trajectories = np.asarray(value.get("trajectories_xyz"), dtype=np.float64)
    if trajectories.ndim != 3 or trajectories.shape[0] < 1 or trajectories.shape[1] < 1 or trajectories.shape[2] < 3:
        return None
    trajectories = trajectories[:, :, :3]
    scores = np.asarray(value.get("scores", np.ones(trajectories.shape[0])), dtype=np.float64).reshape(-1)
    if len(scores) != trajectories.shape[0] or not np.isfinite(scores).all():
        scores = np.ones(trajectories.shape[0], dtype=np.float64)
    return trajectories, scores


def _future_gt_trajectory(
    row: Mapping[str, Any],
    gt_id: str,
    gt_index: RawGTIndex,
    offsets_s: Sequence[float],
) -> Optional[np.ndarray]:
    """Build the GT target path in the receiver-at-send coordinate frame."""
    pose = row.get("receiver_pose_at_send")
    if pose is None:
        return None
    try:
        receiver_inverse = np.linalg.inv(pose_to_world_matrix(pose))
    except (ValueError, np.linalg.LinAlgError):
        return None
    scene = str(row["sequence_id"])
    receiver = str(row["receiver"])
    send_frame = int(row["send_frame_idx"])
    points: List[np.ndarray] = []
    for offset in offsets_s:
        future_frame = send_frame + int(round(float(offset) / 0.1))
        gt = gt_index.by_frame(scene, receiver, future_frame)
        if gt_id not in gt:
            return None
        points.append(transform_point(gt[gt_id], receiver_inverse))
    return np.asarray(points, dtype=np.float64)


def _trajectory_metrics(
    ledger: Sequence[Mapping[str, Any]],
    matches: Mapping[Tuple[str, str, int, int], Optional[str]],
    gt_index: RawGTIndex,
) -> Dict[str, Any]:
    errors: MutableMapping[str, List[float]] = defaultdict(list)
    model_errors: MutableMapping[Tuple[str, str], List[float]] = defaultdict(list)
    mtr_model_errors: MutableMapping[Tuple[str, str, str], List[float]] = defaultdict(list)
    mtr_horizon_errors: MutableMapping[Tuple[str, str, str], List[float]] = defaultdict(list)
    # Per-horizon action-space diagnostic for the two available predictions:
    # choose ego or peer (oracle), or take their arithmetic mean.  These are
    # offline GT metrics only; no choice is exposed to the online ledger.
    mtr_oracle_horizon_errors: MutableMapping[str, List[Tuple[float, float, float, float]]] = defaultdict(list)
    mtr_oracle_scene_errors: MutableMapping[Tuple[str, str], List[Tuple[float, float, float, float]]] = defaultdict(list)
    native_errors: MutableMapping[Tuple[str, str], List[Tuple[float, float, float, float]]] = defaultdict(list)
    native_scene_errors: MutableMapping[Tuple[str, str, str], List[Tuple[float, float, float, float]]] = defaultdict(list)
    native_seen: set = set()
    rows_used = 0
    mtr_rows_used = 0
    harm_probe: List[bool] = []
    harm_cmp: List[bool] = []
    model_counts: MutableMapping[str, int] = defaultdict(int)
    for row in ledger:
        if row.get("target_scope") != "common":
            continue
        scene = str(row["sequence_id"])
        receiver = str(row["receiver"])
        send_frame = int(row["send_frame_idx"])
        local_id = int(row["receiver_local_track_id"])
        gt_id = matches.get((scene, receiver, send_frame, local_id))
        if gt_id is None:
            continue
        send_pose = row.get("receiver_pose_at_send")
        if send_pose is None:
            continue
        future_frame = int(row["observation_frame_idx"])
        gt = gt_index.by_frame(scene, receiver, future_frame)
        if gt_id not in gt:
            continue
        receiver_from_world = np.linalg.inv(pose_to_world_matrix(send_pose))
        target = transform_point(gt[gt_id], receiver_from_world)
        horizon_s = float(row.get("horizon", 0.0))
        peer = _position_from_forecast(row.get("forecast"), horizon_s)
        ego = _position_from_forecast(row.get("ego_forecast"), horizon_s)
        aggregate = _position_from_forecast(row.get("aggregate_forecast"), horizon_s)
        cmp_aggregate = _position_from_forecast(row.get("cmp_aggregate_forecast"), horizon_s)
        if peer is None or ego is None:
            continue
        rows_used += 1
        peer_error = float(np.linalg.norm(peer - target))
        ego_error = float(np.linalg.norm(ego - target))
        errors["peer"].append(peer_error)
        errors["ego_only"].append(ego_error)
        peer_model = str((row.get("forecast") or {}).get("model", "unknown"))
        ego_model = str((row.get("ego_forecast") or {}).get("model", "unknown"))
        horizon_key = "%.1f" % horizon_s
        if ("mtr" in peer_model.lower() or "cmp" in peer_model.lower()) and (
            "mtr" in ego_model.lower() or "cmp" in ego_model.lower()
        ):
            mtr_rows_used += 1
            mtr_model_errors[(peer_model, "peer", "all")].append(peer_error)
            mtr_model_errors[(ego_model, "ego_only", "all")].append(ego_error)
            mtr_horizon_errors[(peer_model, "peer", horizon_key)].append(peer_error)
            mtr_horizon_errors[(ego_model, "ego_only", horizon_key)].append(ego_error)
            oracle_error = min(peer_error, ego_error)
            mean_fusion = 0.5 * (peer + ego)
            mean_fusion_error = float(np.linalg.norm(mean_fusion - target))
            oracle_values = (peer_error, ego_error, oracle_error, mean_fusion_error)
            mtr_oracle_horizon_errors[horizon_key].append(oracle_values)
            mtr_oracle_scene_errors[(scene, horizon_key)].append(oracle_values)

        # Native-style six-mode metrics use the complete MTR trajectory once
        # per source/target/send-time, rather than repeating it for all seven
        # ledger horizons.  This block is offline-only: ``gt_id`` and the
        # future labels are obtained from the independent evaluator above.
        native_key_base = (
            scene,
            receiver,
            str(row.get("source")),
            str(row.get("receiver_target_id")),
            send_frame,
        )
        for field, model, role in (("forecast", peer_model, "peer"), ("ego_forecast", ego_model, "ego_only")):
            if "mtr" not in model.lower() and "cmp" not in model.lower():
                continue
            native_key = native_key_base + (role,)
            if native_key in native_seen:
                continue
            payload = _mtr_trajectory_payload(row.get(field))
            if payload is None:
                continue
            trajectories, scores = payload
            offsets = np.asarray(row[field].get("time_offsets_s", []), dtype=np.float64).reshape(-1)
            if len(offsets) != trajectories.shape[1] or (offsets <= 0).any() or np.any(np.diff(offsets) <= 0):
                continue
            target_path = _future_gt_trajectory(row, gt_id, gt_index, offsets.tolist())
            if target_path is None:
                continue
            native_seen.add(native_key)
            errors_by_mode = np.linalg.norm(trajectories - target_path[None, :, :], axis=2)
            mode_ade = errors_by_mode.mean(axis=1)
            mode_fde = errors_by_mode[:, -1]
            top_mode = int(np.argmax(scores))
            values = (
                float(np.min(mode_ade)),
                float(np.min(mode_fde)),
                float(mode_ade[top_mode]),
                float(mode_fde[top_mode]),
            )
            native_errors[(model, role)].append(values)
            native_scene_errors[(scene, model, role)].append(values)
        if aggregate is not None:
            aggregate_error = float(np.linalg.norm(aggregate - target))
            errors["aggregate"].append(aggregate_error)
            model = str((row.get("aggregate_forecast") or {}).get("model", "unknown"))
            model_counts[model] += 1
            model_errors[(model, "aggregate")].append(aggregate_error)
            model_errors[(model, "ego")].append(ego_error)
            harm_probe.append(aggregate_error > ego_error + 0.1)
            if (model == "cmp" or model.startswith("cmp_")) and cmp_aggregate is None:
                harm_cmp.append(aggregate_error > ego_error + 0.1)
        if cmp_aggregate is not None:
            cmp_error = float(np.linalg.norm(cmp_aggregate - target))
            model_errors[("cmp_aggregate", "aggregate")].append(cmp_error)
            harm_cmp.append(cmp_error > ego_error + 0.1)
    metrics: Dict[str, Any] = {"rows_used": rows_used, "model_counts": dict(model_counts)}
    metrics["mtr_rows_used"] = mtr_rows_used
    for name in ("ego_only", "peer", "aggregate"):
        values = errors.get(name, [])
        metrics[name] = {
            "rows": len(values),
            "ADE": float(np.mean(values)) if values else None,
            "FDE": float(np.mean(values)) if values else None,
        }
    cmp_values = model_errors.get(("cmp_aggregate", "aggregate"), [])
    if not cmp_values:
        cmp_values = [
            value
            for model in model_counts
            if model == "cmp" or model.startswith("cmp_")
            for value in model_errors.get((model, "aggregate"), [])
        ]
    metrics["cmp_aggregate"] = (
        {
            "rows": len(cmp_values),
            "harm_rate_eps_0.1m": float(np.mean(harm_cmp)) if harm_cmp else None,
            "ADE": float(np.mean(cmp_values)) if cmp_values else None,
            "FDE": float(np.mean(cmp_values)) if cmp_values else None,
        }
        if cmp_values
        else None
    )
    metrics["probe_aggregate_harm_rate_eps_0.1m"] = (
        float(np.mean(harm_probe)) if harm_probe else None
    )
    metrics["mtr_by_model"] = {}
    for (model, role, _), values in sorted(mtr_model_errors.items()):
        metrics["mtr_by_model"].setdefault(model, {})[role] = {
            "rows": len(values),
            "ADE": float(np.mean(values)) if values else None,
            "FDE": float(np.mean(values)) if values else None,
        }
    metrics["mtr_by_model_horizon"] = {}
    for (model, role, horizon), values in sorted(mtr_horizon_errors.items()):
        metrics["mtr_by_model_horizon"].setdefault(model, {}).setdefault(horizon, {})[role] = {
            "rows": len(values),
            "ADE": float(np.mean(values)) if values else None,
            "FDE": float(np.mean(values)) if values else None,
        }
    def _oracle_summary(values: Sequence[Tuple[float, float, float, float]]) -> Dict[str, Any]:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 4 or not len(array):
            return {"rows": 0}
        peer_mean, ego_mean, oracle_mean, fusion_mean = array.mean(axis=0).tolist()
        headroom = ego_mean - oracle_mean
        return {
            "rows": int(len(array)),
            "ego_error": float(ego_mean),
            "peer_error": float(peer_mean),
            "oracle_error": float(oracle_mean),
            "oracle_headroom_m": float(headroom),
            "oracle_headroom_relative_to_ego": float(headroom / ego_mean) if ego_mean else None,
            "naive_mean_fusion_error": float(fusion_mean),
            "naive_mean_fusion_minus_ego_m": float(fusion_mean - ego_mean),
            "harm_rate_peer_gt_ego_plus_0.1m": float(np.mean(array[:, 0] > array[:, 1] + 0.1)),
        }
    metrics["mtr_ego_peer_oracle_and_naive_mean"] = {
        "definition": (
            "On paired real MTR top-score positions at each ledger horizon, "
            "oracle_error=min(peer_error, ego_error); naive_mean_fusion is the "
            "arithmetic mean of the peer and ego positions. Both are offline GT "
            "diagnostics and are not an online controller."
        ),
        "overall": _oracle_summary([value for values in mtr_oracle_horizon_errors.values() for value in values]),
        "by_horizon": {
            horizon: _oracle_summary(values)
            for horizon, values in sorted(mtr_oracle_horizon_errors.items())
        },
        "by_scene_horizon": {
            "%s/%s" % (scene, horizon): _oracle_summary(values)
            for (scene, horizon), values in sorted(mtr_oracle_scene_errors.items())
        },
    }
    metrics["mtr_top_score_error_definition"] = (
        "For each ledger horizon, mean 3D Euclidean GT error of the selected-score "
        "MTR position at that single horizon; the legacy ADE/FDE fields above are "
        "point-error aliases, not full-trajectory ADE/FDE."
    )
    metrics["mtr_native_minADE6_minFDE6"] = {}
    for (model, role), values in sorted(native_errors.items()):
        native = np.asarray(values, dtype=np.float64)
        metrics["mtr_native_minADE6_minFDE6"].setdefault(model, {})[role] = {
            "rows": len(values),
            "minADE6": float(native[:, 0].mean()) if len(values) else None,
            "minFDE6": float(native[:, 1].mean()) if len(values) else None,
            "top_score_full_ADE": float(native[:, 2].mean()) if len(values) else None,
            "top_score_full_FDE": float(native[:, 3].mean()) if len(values) else None,
            "definition": "six MTR modes over the complete 0.1--5.0s trajectory; one row per source/target/send-time",
        }
    metrics["mtr_native_by_scene"] = {}
    for (scene, model, role), values in sorted(native_scene_errors.items()):
        native = np.asarray(values, dtype=np.float64)
        metrics["mtr_native_by_scene"].setdefault(scene, {}).setdefault(model, {})[role] = {
            "rows": len(values),
            "minADE6": float(native[:, 0].mean()) if len(values) else None,
            "minFDE6": float(native[:, 1].mean()) if len(values) else None,
            "top_score_full_ADE": float(native[:, 2].mean()) if len(values) else None,
            "top_score_full_FDE": float(native[:, 3].mean()) if len(values) else None,
        }
    metrics["harm_rate_definition"] = (
        "relative harm = peer_error > ego_error + 0.1m for the peer-vs-ego "
        "comparison; legacy probe harm fields retain their aggregate definition"
    )
    return metrics


def evaluate_replay(
    replay_dir: Path,
    raw_root: Path,
    association_gate_m: float = 3.0,
) -> Dict[str, Any]:
    replay_dir = Path(replay_dir)
    tracks = read_jsonl(replay_dir / "local_tracks.jsonl")
    events = read_jsonl(replay_dir / "association_events.jsonl")
    ledger_path = replay_dir / "prediction_ledger.jsonl"
    gt_index = RawGTIndex(Path(raw_root))
    matches, distances, gt_frames = _build_track_matches(tracks, gt_index, association_gate_m)
    return {
        "schema_version": "rvhca.offline_eval.v1",
        "replay_dir": str(replay_dir),
        "raw_root": str(raw_root),
        "association_gate_m": association_gate_m,
        "tracking": _tracking_metrics(tracks, matches, gt_frames),
        "association": _association_metrics(events, matches, tracks, gt_index),
        "track_survival": _track_survival_metrics(tracks, matches),
        # Replay the derived ledger twice as a stream.  Forecast payloads are
        # large multimodal arrays; materialising all rows here can exceed the
        # small container memory limit even though each metric is scalar.
        "ledger": _ledger_metrics(iter_jsonl(ledger_path)),
        "trajectory": _trajectory_metrics(iter_jsonl(ledger_path), matches, gt_index),
    }


def _ledger_metrics(ledger: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    common_count = 0
    valid_count = 0
    by_horizon: MutableMapping[str, MutableMapping[str, int]] = defaultdict(
        lambda: {"rows": 0, "matured_valid_rows": 0}
    )
    for row in ledger:
        if row.get("target_scope") != "common":
            continue
        common_count += 1
        valid = bool(row.get("valid_mask"))
        valid_count += int(valid)
        bucket = by_horizon["%.3f" % float(row["horizon"])]
        bucket["rows"] += 1
        bucket["matured_valid_rows"] += int(valid)
    return {
        "common_rows": common_count,
        "matured_valid_rows": valid_count,
        "row_hindsight_coverage": valid_count / common_count if common_count else 0.0,
        "horizon": {
            key: {
                "rows": counts["rows"],
                "matured_valid_rows": counts["matured_valid_rows"],
                "coverage": counts["matured_valid_rows"] / counts["rows"] if counts["rows"] else 0.0,
            }
            for key, counts in sorted(by_horizon.items())
        },
    }


def write_evaluation(path: Path, result: Mapping[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
