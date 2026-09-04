"""GT-free, CPU-only tracking, association, and ledger replay.

This module deliberately accepts only detector outputs, measured poses, and
timestamps.  Offline labels are handled by ``rvhca_cpu.evaluate`` and are not
imported here.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


DEFAULT_HORIZONS = (0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0)


def _as_float_array(value: Any, shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if shape is not None and arr.shape != shape:
        raise ValueError("expected shape %s, got %s" % (shape, arr.shape))
    return arr


def pose_to_world_matrix(pose: Sequence[float]) -> np.ndarray:
    """Return the OPV2V lidar-to-world matrix.

    OPV2V stores ``[x, y, z, roll, yaw, pitch]`` with angles in degrees.
    This follows the OpenCOOD convention without importing the model stack.
    """

    p = _as_float_array(pose)
    if p.size != 6:
        raise ValueError("pose must have six values")
    x, y, z, roll, yaw, pitch = p.tolist()
    roll, yaw, pitch = np.radians([roll, yaw, pitch])
    c_y, s_y = math.cos(yaw), math.sin(yaw)
    c_r, s_r = math.cos(roll), math.sin(roll)
    c_p, s_p = math.cos(pitch), math.sin(pitch)
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    matrix[:3, 3] = [x, y, z]
    return matrix


def transform_point(point: Sequence[float], matrix: np.ndarray) -> np.ndarray:
    p = np.asarray(point, dtype=np.float64).reshape(3)
    return (matrix @ np.r_[p, 1.0])[:3]


def transform_vector(vector: Sequence[float], matrix: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64).reshape(3)
    return matrix[:3, :3] @ v


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _ensure_ab3dmot_import() -> Any:
    """Load AB3DMOT without exposing its ``io.py`` as a top-level module."""

    project_root = Path(__file__).resolve().parents[1]
    ab3d_root = project_root / "vendor" / "CMP-upstream" / "AB3Dmot"
    toolbox_root = ab3d_root / "Xinshuo_PyToolbox"
    for item in (str(ab3d_root), str(toolbox_root)):
        if item not in sys.path:
            sys.path.insert(0, item)
    from AB3DMOT_libs.model import AB3DMOT  # type: ignore

    return AB3DMOT


def detector_to_ab3dmot(
    boxes3d: np.ndarray,
    boxes2d: np.ndarray,
    scores: np.ndarray,
    min_score: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert sanitized detector rows to AB3DMOT inputs.

    The sanitized artifact has the OpenCOOD tracking layout
    ``[x, y, z, h, w, l, yaw]``.  AB3DMOT consumes ``[h, w, l, x, y, z,
    yaw]``.  No identity or label field is accepted by this function.
    """

    boxes = np.asarray(boxes3d, dtype=np.float64)
    boxes_2d = np.asarray(boxes2d, dtype=np.float64)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if boxes.size == 0:
        boxes = np.empty((0, 7), dtype=np.float64)
    if boxes_2d.size == 0:
        boxes_2d = np.empty((0, 4), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError("boxes3d must have shape (N, 7)")
    if boxes_2d.ndim != 2 or boxes_2d.shape[1] != 4:
        raise ValueError("boxes2d must have shape (N, 4)")
    n = min(len(boxes), len(boxes_2d), len(score))
    boxes, boxes_2d, score = boxes[:n], boxes_2d[:n], score[:n]
    valid = (
        np.isfinite(boxes).all(axis=1)
        & np.isfinite(boxes_2d).all(axis=1)
        & np.isfinite(score)
        & (score >= float(min_score))
        & (boxes[:, 3:6] > 1e-3).all(axis=1)
        & np.isfinite(boxes[:, 6])
    )
    boxes, boxes_2d, score = boxes[valid], boxes_2d[valid], score[valid]
    dets = boxes[:, [3, 4, 5, 0, 1, 2, 6]] if len(boxes) else np.empty((0, 7))
    if len(boxes):
        # AB3DMOT's info payload is bookkeeping only for this path.  Keep the
        # detector score and a finite 2-D box in the source artifact.
        info = np.stack(
            [
                boxes[:, 6],
                np.full(len(boxes), 2.0),
                boxes_2d[:, 0],
                boxes_2d[:, 1],
                boxes_2d[:, 2],
                boxes_2d[:, 3],
                score,
            ],
            axis=1,
        )
    else:
        info = np.empty((0, 7), dtype=np.float64)
    return dets, info, score


@dataclass
class TrackRow:
    sequence_id: str
    source: str
    frame_idx: int
    timestamp_key: str
    time_s: float
    local_track_id: int
    center_local: List[float]
    center_world: List[float]
    dims_hwl: List[float]
    yaw_local: float
    yaw_world: float
    score: float
    pose: List[float]
    velocity_world: List[float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "source": self.source,
            "frame_idx": self.frame_idx,
            "timestamp_key": self.timestamp_key,
            "time_s": self.time_s,
            "local_track_id": self.local_track_id,
            "center_local": self.center_local,
            "center_world": self.center_world,
            "dims_hwl": self.dims_hwl,
            "yaw_local": self.yaw_local,
            "yaw_world": self.yaw_world,
            "score": self.score,
            "pose": self.pose,
            "velocity_world": self.velocity_world,
        }


def _frame_items(frame_map: Mapping[str, Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [frame_map[k] for k in sorted(frame_map, key=lambda item: int(item))]


def _tracker_config() -> Any:
    from easydict import EasyDict as edict

    return edict(
        {
            "dataset": "opv2v",
            "det_name": "pointpillar-CoBEVT-nocompression",
            "score_threshold": -10000,
            "num_hypo": 1,
            "ego_com": False,
            "vis": False,
            "affi_pro": True,
        }
    )


def track_source(
    sequence_id: str,
    source: str,
    frame_map: Mapping[str, Mapping[str, Any]],
    frame_period_s: float = 0.1,
    min_score: float = 0.25,
) -> List[TrackRow]:
    """Run source-local AB3DMOT in timestamp order and emit GT-free rows."""

    AB3DMOT = _ensure_ab3dmot_import()
    import contextlib

    tracker = AB3DMOT(
        _tracker_config(),
        "Car",
        calib=None,
        oxts=None,
        img_dir=None,
        vis_dir=None,
        hw={"image": (375, 1242), "lidar": (720, 1920)},
        log=open(os.devnull, "w"),
        ID_init=1,
    )
    frames = _frame_items(frame_map)
    rows: List[TrackRow] = []
    previous_world: Dict[int, Tuple[int, np.ndarray]] = {}
    for ordinal, frame in enumerate(frames):
        frame_idx = int(frame.get("timestamp_idx", ordinal))
        timestamp_key = str(frame.get("timestamp_key", frame_idx))
        pose = _as_float_array(frame["pose"], (6,))
        boxes = _as_float_array(frame.get("boxes3d", np.empty((0, 7))))
        boxes2d = _as_float_array(frame.get("boxes2d", np.empty((0, 4))))
        scores = _as_float_array(frame.get("scores", np.empty((0,))))
        dets, info, _ = detector_to_ab3dmot(boxes, boxes2d, scores, min_score)
        out, _ = tracker.track(
            {"dets": dets, "info": info, "cav_id": np.zeros(len(dets), dtype=np.int64)},
            frame_idx,
            "%s_%s" % (sequence_id, source),
        )
        output = out[0]
        to_world = pose_to_world_matrix(pose)
        yaw_offset = math.radians(float(pose[4]))
        for item in output:
            # AB3DMOT output: h,w,l,x,y,z,theta,id,info(7),cav_id.
            center_local = np.asarray(item[3:6], dtype=np.float64)
            local_id = int(round(float(item[7])))
            center_world = transform_point(center_local, to_world)
            yaw_local = float(item[6])
            yaw_world = wrap_angle(yaw_local + yaw_offset)
            score = float(item[14]) if len(item) > 14 and np.isfinite(item[14]) else 0.0
            velocity = np.zeros(3, dtype=np.float64)
            if local_id in previous_world:
                prev_frame, prev_center = previous_world[local_id]
                dt = max((frame_idx - prev_frame) * frame_period_s, 1e-6)
                velocity = (center_world - prev_center) / dt
            previous_world[local_id] = (frame_idx, center_world.copy())
            rows.append(
                TrackRow(
                    sequence_id=sequence_id,
                    source=str(source),
                    frame_idx=frame_idx,
                    timestamp_key=timestamp_key,
                    time_s=frame_idx * frame_period_s,
                    local_track_id=local_id,
                    center_local=center_local.tolist(),
                    center_world=center_world.tolist(),
                    dims_hwl=np.asarray(item[:3], dtype=np.float64).tolist(),
                    yaw_local=yaw_local,
                    yaw_world=yaw_world,
                    score=score,
                    pose=pose.tolist(),
                    velocity_world=velocity.tolist(),
                )
            )
    return rows


class ReceiverTargetManager:
    """Position-only receiver target IDs, independent of source IDs."""

    def __init__(self, gate_m: float = 4.0, max_miss: int = 2, frame_period_s: float = 0.1) -> None:
        self.gate_m = float(gate_m)
        self.max_miss = int(max_miss)
        self.frame_period_s = float(frame_period_s)
        self.next_id = 1
        self.active: Dict[int, Dict[str, Any]] = {}

    def update(self, observations: Sequence[TrackRow], frame_idx: int) -> Dict[int, int]:
        ids = sorted(self.active)
        if ids and observations:
            cost = np.full((len(ids), len(observations)), 1e9, dtype=np.float64)
            for i, target_id in enumerate(ids):
                state = self.active[target_id]
                previous = np.asarray(state["center_world"], dtype=np.float64)
                velocity = np.asarray(state.get("velocity_world", [0.0, 0.0, 0.0]), dtype=np.float64)
                dt = max((frame_idx - int(state["frame_idx"])) * self.frame_period_s, 0.0)
                predicted = previous + velocity * dt
                for j, row in enumerate(observations):
                    cost[i, j] = float(np.linalg.norm(predicted - np.asarray(row.center_world)))
            rr, cc = linear_sum_assignment(cost)
        else:
            rr, cc = np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
            cost = np.empty((0, len(observations)), dtype=np.float64)

        assignment: Dict[int, int] = {}
        matched_targets = set()
        matched_obs = set()
        for r, c in zip(rr.tolist(), cc.tolist()):
            if cost[r, c] > self.gate_m:
                continue
            target_id = ids[r]
            row = observations[c]
            self.active[target_id] = {
                "center_world": list(row.center_world),
                "velocity_world": list(row.velocity_world),
                "frame_idx": frame_idx,
                "miss_count": 0,
                "age": int(self.active[target_id].get("age", 0)) + 1,
            }
            assignment[row.local_track_id] = target_id
            matched_targets.add(target_id)
            matched_obs.add(c)

        for target_id in list(self.active):
            if target_id in matched_targets:
                continue
            self.active[target_id]["miss_count"] = int(self.active[target_id].get("miss_count", 0)) + 1
            if self.active[target_id]["miss_count"] > self.max_miss:
                del self.active[target_id]

        for index, row in enumerate(observations):
            if index in matched_obs:
                continue
            target_id = self.next_id
            self.next_id += 1
            self.active[target_id] = {
                "center_world": list(row.center_world),
                "velocity_world": list(row.velocity_world),
                "frame_idx": frame_idx,
                "miss_count": 0,
                "age": 1,
            }
            assignment[row.local_track_id] = target_id
        return assignment


def _associate_peer(
    receiver_rows: Sequence[TrackRow],
    peer_rows: Sequence[TrackRow],
    receiver_pose: Sequence[float],
    gate_m: float,
    peer_offset_s: float = 0.0,
) -> List[Dict[str, Any]]:
    """Associate one peer source to receiver rows in receiver coordinates."""

    receiver_to_local = np.linalg.inv(pose_to_world_matrix(receiver_pose))
    ego_points = [transform_point(row.center_world, receiver_to_local) for row in receiver_rows]
    peer_points = [
        transform_point(
            np.asarray(row.center_world, dtype=np.float64)
            + np.asarray(row.velocity_world, dtype=np.float64) * float(peer_offset_s),
            receiver_to_local,
        )
        for row in peer_rows
    ]
    if ego_points and peer_points:
        cost = np.linalg.norm(
            np.asarray(peer_points)[:, None, :] - np.asarray(ego_points)[None, :, :], axis=2
        )
        rr, cc = linear_sum_assignment(cost)
    else:
        cost = np.empty((len(peer_rows), len(receiver_rows)), dtype=np.float64)
        rr, cc = np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    matched_peer = set()
    events: List[Dict[str, Any]] = []
    for r, c in zip(rr.tolist(), cc.tolist()):
        d = float(cost[r, c])
        if d > gate_m:
            continue
        matched_peer.add(r)
        events.append(
            {
                "receiver_local_track_id": int(receiver_rows[c].local_track_id),
                "peer_local_track_id": int(peer_rows[r].local_track_id),
                "target_scope": "common",
                "association_cost_m": d,
                "association_gate_m": float(gate_m),
                "association_confidence": float(math.exp(-d / max(gate_m, 1e-6))),
                "peer_center_receiver": peer_points[r].tolist(),
                "receiver_center_receiver": ego_points[c].tolist(),
            }
        )
    for index, row in enumerate(peer_rows):
        if index in matched_peer:
            continue
        events.append(
            {
                "receiver_local_track_id": None,
                "peer_local_track_id": int(row.local_track_id),
                "target_scope": "shared_only",
                "association_cost_m": None,
                "association_gate_m": float(gate_m),
                "association_confidence": 0.0,
                "peer_center_receiver": peer_points[index].tolist(),
                "receiver_center_receiver": None,
            }
        )
    return events


def _row_index(rows: Sequence[TrackRow]) -> Dict[Tuple[str, int], TrackRow]:
    return {(row.source, row.frame_idx): row for row in rows}


def _group_tracks(rows: Sequence[TrackRow]) -> Dict[Tuple[str, int], List[TrackRow]]:
    grouped: Dict[Tuple[str, int], List[TrackRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.source, row.frame_idx)].append(row)
    return grouped


def _forecast_in_receiver_frame(
    row: TrackRow,
    receiver_pose: Sequence[float],
    horizon_s: float,
) -> List[float]:
    world = np.asarray(row.center_world, dtype=np.float64) + np.asarray(row.velocity_world, dtype=np.float64) * float(horizon_s)
    to_receiver = np.linalg.inv(pose_to_world_matrix(receiver_pose))
    return transform_point(world, to_receiver).tolist()


def _realized_in_send_frame(
    observation: TrackRow,
    send_pose: Sequence[float],
) -> List[float]:
    return transform_point(observation.center_world, np.linalg.inv(pose_to_world_matrix(send_pose))).tolist()


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def build_replay(
    detection_cache: Mapping[str, Any],
    frame_period_s: float = 0.1,
    min_score: float = 0.25,
    association_gate_m: float = 4.0,
    target_gate_m: float = 4.0,
    target_max_miss: int = 2,
    peer_delay_s: float = 0.1,
    horizons_s: Sequence[float] = DEFAULT_HORIZONS,
    aggregate_mode: str = "none",
    scene_allowlist: Optional[Sequence[str]] = None,
    max_frames_per_source: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the GT-free CPU pipeline and return JSON-serializable artifacts."""

    allowed = set(scene_allowlist) if scene_allowlist else None
    all_tracks: List[TrackRow] = []
    tracking_failures: List[Dict[str, str]] = []
    pose_index: Dict[Tuple[str, str, int], List[float]] = {}
    scene_sources = detection_cache.get("scenes", {})
    for sequence_id in sorted(scene_sources):
        if allowed is not None and sequence_id not in allowed:
            continue
        for source in sorted(scene_sources[sequence_id], key=str):
            frame_map = scene_sources[sequence_id][source]
            if max_frames_per_source is not None:
                keys = sorted(frame_map, key=lambda item: int(item))[: int(max_frames_per_source)]
                frame_map = {key: frame_map[key] for key in keys}
            for frame in _frame_items(frame_map):
                pose_index[(sequence_id, str(source), int(frame.get("timestamp_idx", 0)))] = _as_float_array(
                    frame["pose"], (6,)
                ).tolist()
            try:
                all_tracks.extend(
                    track_source(
                        sequence_id,
                        str(source),
                        frame_map,
                        frame_period_s=frame_period_s,
                        min_score=min_score,
                    )
                )
            except Exception as exc:  # keep other sources auditable
                tracking_failures.append(
                    {"sequence_id": sequence_id, "source": str(source), "error": repr(exc)}
                )

    tracks_by_scene_source_frame = _group_tracks(all_tracks)
    scene_frame_indices: Dict[str, List[int]] = defaultdict(list)
    for sequence_id, _, frame_idx in pose_index:
        if frame_idx not in scene_frame_indices[sequence_id]:
            scene_frame_indices[sequence_id].append(frame_idx)
    poses: Dict[Tuple[str, str, int], List[float]] = dict(pose_index)
    for row in all_tracks:
        key = (row.sequence_id, row.source, row.frame_idx)
        poses[key] = row.pose
        if row.frame_idx not in scene_frame_indices[row.sequence_id]:
            scene_frame_indices[row.sequence_id].append(row.frame_idx)
    for sequence_id in scene_frame_indices:
        scene_frame_indices[sequence_id].sort()

    target_assignments: Dict[Tuple[str, str, int, int], int] = {}
    target_metadata: Dict[Tuple[str, str, int, int], Dict[str, int]] = {}
    observations_by_target: Dict[Tuple[str, str, int, int], TrackRow] = {}
    association_events: List[Dict[str, Any]] = []
    ledger: List[Dict[str, Any]] = []

    for sequence_id in sorted(scene_frame_indices):
        sources = sorted(
            {row.source for row in all_tracks if row.sequence_id == sequence_id}
            | {source for scene, source, _ in pose_index if scene == sequence_id}
        )
        by_source_frame = {
            (source, frame_idx): tracks_by_scene_source_frame.get((source, frame_idx), [])
            for source in sources
            for frame_idx in scene_frame_indices[sequence_id]
        }
        # First assign receiver-local target IDs for the complete sequence.
        # Ledger rows are created only afterwards so a future observation is
        # available for hindsight lookup (no future value is used to assign an
        # ID or an association).
        managers: Dict[str, ReceiverTargetManager] = {
            source: ReceiverTargetManager(target_gate_m, target_max_miss, frame_period_s)
            for source in sources
        }
        for frame_idx in scene_frame_indices[sequence_id]:
            for receiver in sources:
                receiver_rows = by_source_frame[(receiver, frame_idx)]
                receiver_assignment = managers[receiver].update(receiver_rows, frame_idx)
                for row in receiver_rows:
                    target_id = receiver_assignment[row.local_track_id]
                    target_assignments[(sequence_id, receiver, frame_idx, row.local_track_id)] = target_id
                    observations_by_target[(sequence_id, receiver, frame_idx, target_id)] = row
                    state = managers[receiver].active.get(target_id, {})
                    target_metadata[(sequence_id, receiver, frame_idx, target_id)] = {
                        "track_age": int(state.get("age", 1)),
                        "miss_count": int(state.get("miss_count", 0)),
                    }

        # Cross-source association occurs when a peer packet arrives.  The
        # peer state is issued at ``send_frame_idx`` and propagated to the
        # arrival frame for the geometry gate; the receiver target ID is the
        # ID assigned at that arrival frame.  Forecast evaluation remains in
        # the receiver frame at send time.
        for send_frame_idx in scene_frame_indices[sequence_id]:
            for receiver in sources:
                send_receiver_rows = by_source_frame[(receiver, send_frame_idx)]
                send_pose = poses.get((sequence_id, receiver, send_frame_idx))
                if send_pose is None:
                    continue
                arrival_frame_idx = send_frame_idx + int(round(peer_delay_s / frame_period_s))
                arrival_time = send_frame_idx * frame_period_s + float(peer_delay_s)
                arrival_receiver_rows = by_source_frame.get((receiver, arrival_frame_idx), [])
                arrival_pose = poses.get((sequence_id, receiver, arrival_frame_idx), send_pose)
                arrival_in_sequence = (sequence_id, receiver, arrival_frame_idx) in poses
                for source in sources:
                    if source == receiver:
                        continue
                    peer_rows = by_source_frame[(source, send_frame_idx)]
                    events = _associate_peer(
                        arrival_receiver_rows,
                        peer_rows,
                        arrival_pose,
                        association_gate_m,
                        peer_offset_s=(arrival_frame_idx - send_frame_idx) * frame_period_s,
                    )
                    for event in events:
                        event.update(
                            {
                                "sequence_id": sequence_id,
                                "receiver": receiver,
                                "source": source,
                                "frame_idx": arrival_frame_idx,
                                "send_frame_idx": send_frame_idx,
                                "peer_frame_idx": send_frame_idx,
                                "send_time": send_frame_idx * frame_period_s,
                                "arrival_time": arrival_time,
                                "arrival_frame_idx": arrival_frame_idx,
                            }
                        )
                        receiver_local_id: Optional[int] = None
                        receiver_send_row: Optional[TrackRow] = None
                        if event["target_scope"] == "common":
                            receiver_local_id = int(event["receiver_local_track_id"])
                            target_id = target_assignments[(sequence_id, receiver, arrival_frame_idx, receiver_local_id)]
                            event["receiver_target_id"] = target_id
                            # Main task is common targets: require a receiver
                            # observation at issue time as the ego baseline.
                            receiver_send_row = next(
                                (
                                    row
                                    for row in send_receiver_rows
                                    if target_assignments.get(
                                        (sequence_id, receiver, send_frame_idx, row.local_track_id)
                                    )
                                    == target_id
                                ),
                                None,
                            )
                            if receiver_send_row is None:
                                event["target_scope"] = "shared_only"
                                event["receiver_target_id"] = None
                                event["censor_reason"] = "receiver_not_observed_at_send"
                        else:
                            event["receiver_target_id"] = None
                            if not arrival_in_sequence:
                                event["censor_reason"] = "arrival_out_of_sequence"
                        association_events.append(event)

                        if event["target_scope"] != "common" or receiver_local_id is None or receiver_send_row is None:
                            continue
                        peer_local_id = int(event["peer_local_track_id"])
                        peer_row = next(row for row in peer_rows if row.local_track_id == peer_local_id)
                        target_id = int(event["receiver_target_id"])
                        for horizon_s in horizons_s:
                            horizon = float(horizon_s)
                            observation_frame = send_frame_idx + int(round(horizon / frame_period_s))
                            observation_time = observation_frame * frame_period_s
                            peer_forecast = _forecast_in_receiver_frame(peer_row, send_pose, horizon)
                            ego_forecast = _forecast_in_receiver_frame(receiver_send_row, send_pose, horizon)
                            aggregate_forecast = None
                            if aggregate_mode == "mean_probe":
                                aggregate_forecast = (
                                    (np.asarray(peer_forecast) + np.asarray(ego_forecast)) / 2.0
                                ).tolist()
                            elif aggregate_mode not in ("none", "mean_probe"):
                                raise ValueError("unknown aggregate_mode: %s" % aggregate_mode)
                            row = {
                                "sequence_id": sequence_id,
                                "receiver": receiver,
                                "source": source,
                                "source_track_id": peer_local_id,
                                "receiver_local_track_id": int(receiver_send_row.local_track_id),
                                "association_receiver_local_track_id": receiver_local_id,
                                "receiver_target_id": target_id,
                                "send_frame_idx": send_frame_idx,
                                "send_timestamp_key": str(receiver_send_row.timestamp_key),
                                "send_time": send_frame_idx * frame_period_s,
                                "arrival_time": arrival_time,
                                "arrival_frame_idx": arrival_frame_idx,
                                "horizon": horizon,
                                "forecast": {
                                    "frame": "receiver@send_time",
                                    "position_xyz": peer_forecast,
                                    "model": "constant_velocity_probe",
                                },
                                "ego_forecast": {
                                    "frame": "receiver@send_time",
                                    "position_xyz": ego_forecast,
                                    "model": "constant_velocity_probe",
                                },
                                "aggregate_forecast": (
                                    {
                                        "frame": "receiver@send_time",
                                        "position_xyz": aggregate_forecast,
                                        "model": "mean_probe_not_cmp",
                                    }
                                    if aggregate_forecast is not None
                                    else None
                                ),
                                "cmp_aggregate_forecast": None,
                                "receiver_pose_at_send": list(send_pose),
                                "track_age": target_metadata.get(
                                    (sequence_id, receiver, arrival_frame_idx, target_id), {}
                                ).get("track_age", 1),
                                "miss_count": target_metadata.get(
                                    (sequence_id, receiver, arrival_frame_idx, target_id), {}
                                ).get("miss_count", 0),
                                "association_confidence": event["association_confidence"],
                                "target_scope": "common",
                                "observation_frame_idx": observation_frame,
                                "observation_time": observation_time,
                                "matured": False,
                                "valid_mask": False,
                                "matched_receiver_track_id": None,
                                "realized_state": None,
                                "peer_realized_error": None,
                                "ego_only_error": None,
                                "aggregate_error": None,
                                "cmp_aggregate_error": None,
                                "error_type": "none",
                                "censor_reason": None,
                            }
                            if arrival_time > observation_time + 1e-6:
                                row["censor_reason"] = "arrival_after_horizon"
                            else:
                                observed = observations_by_target.get(
                                    (sequence_id, receiver, observation_frame, target_id)
                                )
                                if observed is None:
                                    row["censor_reason"] = "not_observed_or_track_dead"
                                else:
                                    realized = _realized_in_send_frame(observed, send_pose)
                                    row["matured"] = True
                                    row["valid_mask"] = True
                                    row["matched_receiver_track_id"] = observed.local_track_id
                                    row["realized_state"] = realized
                                    row["peer_realized_error"] = _distance(peer_forecast, realized)
                                    row["ego_only_error"] = _distance(ego_forecast, realized)
                                    if aggregate_forecast is not None:
                                        row["aggregate_error"] = _distance(aggregate_forecast, realized)
                                    row["error_type"] = "displacement"
                            ledger.append(row)

    summary = summarize_replay(all_tracks, association_events, ledger, tracking_failures)
    return {
        "schema_version": "rvhca.cpu_replay.v0",
        "config": {
            "frame_period_s": frame_period_s,
            "min_score": min_score,
            "association_gate_m": association_gate_m,
            "target_gate_m": target_gate_m,
            "target_max_miss": target_max_miss,
            "peer_delay_s": peer_delay_s,
            "horizons_s": [float(x) for x in horizons_s],
            "aggregate_mode": aggregate_mode,
        },
        "tracks": [row.as_dict() for row in all_tracks],
        "association_events": association_events,
        "ledger": ledger,
        "summary": summary,
    }


def summarize_replay(
    tracks: Sequence[TrackRow],
    association_events: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    tracking_failures: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    common = [event for event in association_events if event.get("target_scope") == "common"]
    shared_only = [event for event in association_events if event.get("target_scope") == "shared_only"]
    valid = [row for row in ledger if row.get("valid_mask")]
    aggregate_rows = [row for row in valid if row.get("aggregate_error") is not None]
    out: Dict[str, Any] = {
        "track_rows": len(tracks),
        "association_events": len(association_events),
        "common_events": len(common),
        "shared_only_events": len(shared_only),
        "ledger_rows": len(ledger),
        "ledger_matured_rows": len(valid),
        "ledger_hindsight_coverage": float(len(valid) / len(ledger)) if ledger else 0.0,
        "horizon_coverage": {},
        "mean_peer_realized_error": None,
        "mean_ego_only_error": None,
        "mean_aggregate_error": None,
        "harm_rate_eps_0.1m": None,
        "tracking_failures": list(tracking_failures),
    }
    if valid:
        out["mean_peer_realized_error"] = float(
            np.mean([row["peer_realized_error"] for row in valid if row.get("peer_realized_error") is not None])
        )
        out["mean_ego_only_error"] = float(
            np.mean([row["ego_only_error"] for row in valid if row.get("ego_only_error") is not None])
        )
    if aggregate_rows:
        out["mean_aggregate_error"] = float(np.mean([row["aggregate_error"] for row in aggregate_rows]))
        out["harm_rate_eps_0.1m"] = float(
            np.mean(
                [
                    row["aggregate_error"] > row["ego_only_error"] + 0.1
                    for row in aggregate_rows
                ]
            )
        )
    by_horizon: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in ledger:
        by_horizon["%.3f" % float(row["horizon"])].append(row)
    for horizon, rows in sorted(by_horizon.items()):
        matured = sum(bool(row.get("valid_mask")) for row in rows)
        out["horizon_coverage"][horizon] = {
            "rows": len(rows),
            "matured_rows": matured,
            "coverage": float(matured / len(rows)) if rows else 0.0,
        }
    return out


def write_replay(output_dir: Path, replay: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, key in (
        ("local_tracks.jsonl", "tracks"),
        ("association_events.jsonl", "association_events"),
        ("prediction_ledger.jsonl", "ledger"),
    ):
        with (output_dir / name).open("w", encoding="utf-8") as handle:
            for row in replay[key]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"schema_version": replay["schema_version"], "config": replay["config"], "summary": replay["summary"]},
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
