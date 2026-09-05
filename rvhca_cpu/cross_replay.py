"""GT-free association between independent detector/tracker replays.

The CoBEVT peer replay and PointPillar canonical receiver replay have
independent AB3DMOT identifiers.  Geometry is used exactly once, at packet
arrival, to select a canonical receiver target.  All send-time and future
lookups subsequently use that receiver_target_id without another association.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


PAIRING_SCHEMA_VERSION = "rvhca.cross_replay_pairing.v1"


class CrossReplayContractError(ValueError):
    """Raised when an input cannot support a GT-free cross-replay pairing."""


def _vector(row: Mapping[str, Any], field: str) -> np.ndarray:
    try:
        value = np.asarray(row[field], dtype=np.float64).reshape(3)
    except (KeyError, TypeError, ValueError) as exc:
        raise CrossReplayContractError("%s must be a finite three-vector" % field) from exc
    if not np.isfinite(value).all():
        raise CrossReplayContractError("%s must be finite" % field)
    return value


def _require_gt_free(row: Mapping[str, Any], kind: str) -> None:
    if bool(row.get("uses_gt", False)):
        raise CrossReplayContractError("%s declares uses_gt=true" % kind)
    if str(row.get("tracking_frame", "world_fixed")) != "world_fixed":
        raise CrossReplayContractError("%s is not in world_fixed tracking frame" % kind)


def _summary_costs(costs: Sequence[float]) -> Dict[str, Any]:
    if not costs:
        return {"count": 0, "mean_m": None, "median_m": None, "p95_m": None, "max_m": None}
    values = np.asarray(costs, dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean_m": float(values.mean()),
        "median_m": float(np.median(values)),
        "p95_m": float(np.percentile(values, 95)),
        "max_m": float(values.max()),
    }


def build_cross_replay_pairings(
    canonical_target_states: Sequence[Mapping[str, Any]],
    peer_tracks: Sequence[Mapping[str, Any]],
    *,
    frame_period_s: float,
    peer_delay_s: float,
    association_gate_m: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build causal CoBEVT-to-canonical target pairings.

    ``canonical_target_states`` must originate from the PointPillar receiver
    replay.  A source track is propagated from send to arrival using its
    current velocity and Hungarian-matched once to arrival states.  The target
    ID returned by that operation is then looked up at send only by identity.
    """

    if frame_period_s <= 0.0 or peer_delay_s < 0.0 or association_gate_m <= 0.0:
        raise CrossReplayContractError("frame period, delay, and association gate must be positive")
    arrival_offset = int(round(float(peer_delay_s) / float(frame_period_s)))
    if not canonical_target_states:
        raise CrossReplayContractError("canonical target states are empty")

    states_at_frame: MutableMapping[Tuple[str, str, int], List[Mapping[str, Any]]] = defaultdict(list)
    state_at_target: Dict[Tuple[str, str, int, int], Mapping[str, Any]] = {}
    receivers_by_scene: MutableMapping[str, set] = defaultdict(set)
    for state in canonical_target_states:
        _require_gt_free(state, "canonical target state")
        scene = str(state["sequence_id"])
        receiver = str(state["receiver"])
        frame = int(state["frame_idx"])
        target_id = int(state["receiver_target_id"])
        local_id = int(state["receiver_local_track_id"])
        _vector(state, "center_world")
        _vector(state, "velocity_world")
        key = (scene, receiver, frame, target_id)
        if key in state_at_target:
            raise CrossReplayContractError(
                "duplicate canonical target state for %s/%s/frame=%d/target=%d"
                % (scene, receiver, frame, target_id)
            )
        states_at_frame[(scene, receiver, frame)].append(state)
        state_at_target[key] = state
        receivers_by_scene[scene].add(receiver)
        if local_id < 0:
            raise CrossReplayContractError("receiver_local_track_id must be non-negative")

    peers_by_frame: MutableMapping[Tuple[str, str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for peer in peer_tracks:
        _require_gt_free(peer, "peer track")
        scene = str(peer["sequence_id"])
        source = str(peer["source"])
        frame = int(peer["frame_idx"])
        int(peer["local_track_id"])
        _vector(peer, "center_world")
        _vector(peer, "velocity_world")
        peers_by_frame[(scene, source, frame)].append(peer)

    output: List[Dict[str, Any]] = []
    costs: List[float] = []
    summary_counts = {
        "receiver_expanded_candidates": 0,
        "common_pairs": 0,
        "shared_only_pairs": 0,
        "rejected_pairs": 0,
        "unmatched_geometry_pairs": 0,
    }
    for (scene, source, send_frame), peers in sorted(peers_by_frame.items()):
        arrival_frame = send_frame + arrival_offset
        for receiver in sorted(receivers_by_scene.get(scene, set())):
            if receiver == source:
                continue
            summary_counts["receiver_expanded_candidates"] += len(peers)
            arrival_states = states_at_frame.get((scene, receiver, arrival_frame), [])
            if not arrival_states:
                for peer in peers:
                    output.append(
                        _pairing_row(
                            scene=scene,
                            receiver=receiver,
                            source=source,
                            send_frame=send_frame,
                            arrival_frame=arrival_frame,
                            peer=peer,
                            association_gate_m=association_gate_m,
                            target_scope="rejected",
                            pairing_status="rejected",
                            censor_reason="arrival_canonical_state_unavailable",
                        )
                    )
                    summary_counts["rejected_pairs"] += 1
                continue

            peer_points = np.asarray(
                [
                    _vector(peer, "center_world")
                    + _vector(peer, "velocity_world") * float(peer_delay_s)
                    for peer in peers
                ],
                dtype=np.float64,
            )
            receiver_points = np.asarray([_vector(state, "center_world") for state in arrival_states], dtype=np.float64)
            matrix = np.linalg.norm(peer_points[:, None, :] - receiver_points[None, :, :], axis=2)
            rr, cc = linear_sum_assignment(matrix)
            matches = {
                int(peer_index): (arrival_states[int(state_index)], float(matrix[peer_index, state_index]))
                for peer_index, state_index in zip(rr.tolist(), cc.tolist())
                if float(matrix[peer_index, state_index]) <= float(association_gate_m)
            }
            for peer_index, peer in enumerate(peers):
                matched = matches.get(peer_index)
                if matched is None:
                    output.append(
                        _pairing_row(
                            scene=scene,
                            receiver=receiver,
                            source=source,
                            send_frame=send_frame,
                            arrival_frame=arrival_frame,
                            peer=peer,
                            association_gate_m=association_gate_m,
                            target_scope="rejected",
                            pairing_status="unmatched_geometry",
                            censor_reason="no_arrival_geometry_match",
                            peer_center_world_at_arrival=peer_points[peer_index].tolist(),
                        )
                    )
                    summary_counts["rejected_pairs"] += 1
                    summary_counts["unmatched_geometry_pairs"] += 1
                    continue

                arrival_state, cost = matched
                target_id = int(arrival_state["receiver_target_id"])
                send_state = state_at_target.get((scene, receiver, send_frame, target_id))
                if send_state is None:
                    target_scope = "shared_only"
                    pairing_status = "matched_arrival_target_not_observed_at_send"
                    censor_reason = "receiver_target_not_observed_at_send"
                    summary_counts["shared_only_pairs"] += 1
                else:
                    target_scope = "common"
                    pairing_status = "common"
                    censor_reason = None
                    summary_counts["common_pairs"] += 1
                output.append(
                    _pairing_row(
                        scene=scene,
                        receiver=receiver,
                        source=source,
                        send_frame=send_frame,
                        arrival_frame=arrival_frame,
                        peer=peer,
                        association_gate_m=association_gate_m,
                        target_scope=target_scope,
                        pairing_status=pairing_status,
                        censor_reason=censor_reason,
                        target_id=target_id,
                        arrival_state=arrival_state,
                        send_state=send_state,
                        association_cost_m=cost,
                        peer_center_world_at_arrival=peer_points[peer_index].tolist(),
                    )
                )
                costs.append(cost)

    candidates = summary_counts["receiver_expanded_candidates"]
    common_coverage = float(summary_counts["common_pairs"] / candidates) if candidates else 0.0
    arrival_coverage = (
        float((summary_counts["common_pairs"] + summary_counts["shared_only_pairs"]) / candidates)
        if candidates
        else 0.0
    )
    summary: Dict[str, Any] = {
        "schema_version": PAIRING_SCHEMA_VERSION,
        "frame_period_s": float(frame_period_s),
        "peer_delay_s": float(peer_delay_s),
        "arrival_frame_offset": arrival_offset,
        "association_gate_m": float(association_gate_m),
        **summary_counts,
        "pairing_coverage": common_coverage,
        "common_pairing_coverage": common_coverage,
        "arrival_pairing_coverage": arrival_coverage,
        "pairing_coverage_definition": "common pairs / receiver-expanded candidates",
        "arrival_pairing_coverage_definition": (
            "common plus shared-only arrival geometry matches / receiver-expanded candidates"
        ),
        "association_cost_m": _summary_costs(costs),
        "uses_gt": False,
    }
    return output, summary


def _pairing_row(
    *,
    scene: str,
    receiver: str,
    source: str,
    send_frame: int,
    arrival_frame: int,
    peer: Mapping[str, Any],
    association_gate_m: float,
    target_scope: str,
    pairing_status: str,
    censor_reason: Any,
    target_id: Any = None,
    arrival_state: Any = None,
    send_state: Any = None,
    association_cost_m: Any = None,
    peer_center_world_at_arrival: Any = None,
) -> Dict[str, Any]:
    return {
        "schema_version": PAIRING_SCHEMA_VERSION,
        "sequence_id": scene,
        "receiver": receiver,
        "source": source,
        "send_frame_idx": int(send_frame),
        "arrival_frame_idx": int(arrival_frame),
        "peer_replay_source_track_id": str(peer["local_track_id"]),
        "canonical_receiver_target_id": int(target_id) if target_id is not None else None,
        "arrival_receiver_local_track_id": (
            int(arrival_state["receiver_local_track_id"]) if arrival_state is not None else None
        ),
        "send_receiver_local_track_id": (
            int(send_state["receiver_local_track_id"]) if send_state is not None else None
        ),
        "association_cost_m": association_cost_m,
        "association_gate_m": float(association_gate_m),
        "association_confidence": (
            float(math.exp(-float(association_cost_m) / float(association_gate_m)))
            if association_cost_m is not None
            else 0.0
        ),
        "peer_center_world_at_send": _vector(peer, "center_world").tolist(),
        "peer_center_world_at_arrival": peer_center_world_at_arrival,
        "receiver_center_world_at_arrival": (
            _vector(arrival_state, "center_world").tolist() if arrival_state is not None else None
        ),
        "target_scope": target_scope,
        "pairing_status": pairing_status,
        "censor_reason": censor_reason,
        "tracking_frame": "world_fixed",
        "uses_gt": False,
    }
