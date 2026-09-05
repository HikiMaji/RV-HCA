"""Build a GT-free scientific MTR ledger from explicit cross-replay pairings."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from rvhca_cpu.mtr_io import MTRContractError, _forecast_payload, normalize_prediction_record
from rvhca_cpu.online import pose_to_world_matrix, transform_point


PAIRED_LEDGER_SCHEMA_VERSION = "rvhca.paired_mtr_ledger.v1"
_PROVENANCE_FIELDS = (
    "track_history_family",
    "checkpoint_history_family",
    "input_distribution_status",
    "history_reference_frame",
)


def _target_state_index(
    target_states: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str, int, int], Mapping[str, Any]]:
    index: Dict[Tuple[str, str, int, int], Mapping[str, Any]] = {}
    for state in target_states:
        if bool(state.get("uses_gt", False)):
            raise MTRContractError("canonical target state declares uses_gt=true")
        if str(state.get("tracking_frame", "world_fixed")) != "world_fixed":
            raise MTRContractError("canonical target state is not world_fixed")
        key = (
            str(state["sequence_id"]),
            str(state["receiver"]),
            int(state["frame_idx"]),
            int(state["receiver_target_id"]),
        )
        if key in index:
            raise MTRContractError(
                "duplicate canonical target state for %s/%s/frame=%d/target=%d" % key
            )
        center = np.asarray(state["center_world"], dtype=np.float64).reshape(-1)
        pose = np.asarray(state["pose"], dtype=np.float64).reshape(-1)
        if center.size != 3 or pose.size != 6 or not np.isfinite(center).all() or not np.isfinite(pose).all():
            raise MTRContractError("canonical target state has invalid world center or pose")
        index[key] = state
    return index


def _scientific_record_index(
    records: Sequence[Mapping[str, Any]], expected_role: str
) -> Dict[Tuple[str, str, int, str], Mapping[str, Any]]:
    index: Dict[Tuple[str, str, int, str], Mapping[str, Any]] = {}
    for raw_record in records:
        record = normalize_prediction_record(raw_record)
        if str(record["role"]) != expected_role:
            raise MTRContractError("expected %s record, got %s" % (expected_role, record["role"]))
        missing = [field for field in _PROVENANCE_FIELDS if field not in record]
        if missing:
            raise MTRContractError(
                "scientific %s record lacks provenance: %s" % (expected_role, ", ".join(missing))
            )
        if str(record["input_distribution_status"]) != "PASS":
            raise MTRContractError("scientific %s record has non-PASS input distribution" % expected_role)
        if str(record["history_reference_frame"]) != "world":
            raise MTRContractError("scientific %s record must use world history" % expected_role)
        if str(record["track_history_family"]) != str(record["checkpoint_history_family"]):
            raise MTRContractError("scientific %s record has mismatched history provenance" % expected_role)
        key = (
            str(record["sequence_id"]),
            str(record["source"]),
            int(record["send_frame_idx"]),
            str(record["source_track_id"]),
        )
        if key in index:
            raise MTRContractError("ambiguous scientific %s prediction key" % expected_role)
        index[key] = record
    return index


def _provenance(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {field: record[field] for field in _PROVENANCE_FIELDS}


def _realized_in_receiver_send_frame(state: Mapping[str, Any], send_pose: Sequence[float]) -> List[float]:
    receiver_from_world = np.linalg.inv(pose_to_world_matrix(send_pose))
    return transform_point(state["center_world"], receiver_from_world).tolist()


def build_paired_mtr_ledger(
    pairing_rows: Sequence[Mapping[str, Any]],
    canonical_target_states: Sequence[Mapping[str, Any]],
    peer_records: Sequence[Mapping[str, Any]],
    ego_records: Sequence[Mapping[str, Any]],
    *,
    horizons_s: Sequence[float],
    frame_period_s: float,
    peer_delay_s: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Join peer and ego MTR records through canonical temporal target IDs.

    The pairing input supplies the only arrival-time geometric association.
    Send and future observations are retrieved by the same target ID, never by
    a second nearest-neighbour operation or cross-replay local-ID comparison.
    """

    if frame_period_s <= 0.0 or peer_delay_s < 0.0:
        raise MTRContractError("frame period must be positive and peer delay non-negative")
    horizons = [float(value) for value in horizons_s]
    if not horizons or any(value <= 0.0 for value in horizons):
        raise MTRContractError("horizons must be non-empty and positive")
    target_states = _target_state_index(canonical_target_states)
    peer_index = _scientific_record_index(peer_records, "peer")
    ego_index = _scientific_record_index(ego_records, "ego")

    output: List[Dict[str, Any]] = []
    unmatched_peer_keys = set()
    unmatched_ego_keys = set()
    missing_send_target_keys = set()
    common_pairing_rows = 0
    censored_rows = 0
    matured_rows = 0
    for pairing in pairing_rows:
        if bool(pairing.get("uses_gt", False)):
            raise MTRContractError("pairing record declares uses_gt=true")
        if str(pairing.get("tracking_frame", "world_fixed")) != "world_fixed":
            raise MTRContractError("pairing record is not in world_fixed tracking frame")
        if str(pairing.get("target_scope")) != "common":
            continue
        common_pairing_rows += 1
        scene = str(pairing["sequence_id"])
        receiver = str(pairing["receiver"])
        source = str(pairing["source"])
        send_frame = int(pairing["send_frame_idx"])
        arrival_frame = int(pairing["arrival_frame_idx"])
        target_id = int(pairing["canonical_receiver_target_id"])
        peer_track_id = str(pairing["peer_replay_source_track_id"])
        send_local_id = int(pairing["send_receiver_local_track_id"])
        arrival_local_id = int(pairing["arrival_receiver_local_track_id"])
        arrival_state_key = (scene, receiver, arrival_frame, target_id)
        arrival_state = target_states.get(arrival_state_key)
        if arrival_state is None:
            raise MTRContractError("pairing arrival target is absent from canonical target states")
        if int(arrival_state["receiver_local_track_id"]) != arrival_local_id:
            raise MTRContractError("pairing arrival local ID disagrees with canonical target continuity")
        send_state_key = (scene, receiver, send_frame, target_id)
        send_state = target_states.get(send_state_key)
        if send_state is None:
            missing_send_target_keys.add(send_state_key)
            continue
        if int(send_state["receiver_local_track_id"]) != send_local_id:
            raise MTRContractError("pairing send local ID disagrees with canonical target continuity")

        peer_key = (scene, source, send_frame, peer_track_id)
        peer_record = peer_index.get(peer_key)
        if peer_record is None:
            unmatched_peer_keys.add(peer_key)
            continue
        ego_key = (scene, receiver, send_frame, str(send_local_id))
        ego_record = ego_index.get(ego_key)
        if ego_record is None:
            unmatched_ego_keys.add(ego_key)
            continue

        send_pose = list(send_state["pose"])
        arrival_time = send_frame * float(frame_period_s) + float(peer_delay_s)
        for horizon in horizons:
            observation_frame = send_frame + int(round(horizon / float(frame_period_s)))
            observation_time = observation_frame * float(frame_period_s)
            row: Dict[str, Any] = {
                "schema_version": PAIRED_LEDGER_SCHEMA_VERSION,
                "tracking_frame": "world_fixed",
                "uses_gt": False,
                "sequence_id": scene,
                "receiver": receiver,
                "source": source,
                "source_track_id": peer_track_id,
                "peer_replay_source_track_id": peer_track_id,
                "receiver_local_track_id": send_local_id,
                "association_receiver_local_track_id": int(pairing["arrival_receiver_local_track_id"]),
                "receiver_target_id": target_id,
                "send_frame_idx": send_frame,
                "send_timestamp_key": str(send_state["timestamp_key"]),
                "send_time": send_frame * float(frame_period_s),
                "arrival_time": arrival_time,
                "arrival_frame_idx": arrival_frame,
                "horizon": horizon,
                "receiver_pose_at_send": send_pose,
                "track_age": int(send_state.get("track_age", 1)),
                "miss_count": int(send_state.get("miss_count", 0)),
                "association_cost_m": pairing.get("association_cost_m"),
                "association_gate_m": pairing.get("association_gate_m"),
                "association_confidence": pairing.get("association_confidence"),
                "pairing_status": pairing.get("pairing_status", "common"),
                "target_scope": "common",
                "peer_prediction_provenance": _provenance(peer_record),
                "ego_prediction_provenance": _provenance(ego_record),
                "observation_frame_idx": observation_frame,
                "observation_time": observation_time,
                "matured": False,
                "valid_mask": False,
                "matched_receiver_track_id": None,
                "realized_state": None,
                "peer_realized_error": None,
                "ego_only_error": None,
                "peer_realized_error_3d": None,
                "ego_only_error_3d": None,
                "aggregate_error": None,
                "aggregate_error_3d": None,
                "cmp_aggregate_error": None,
                "error_type": "none",
                "censor_reason": None,
            }
            row["forecast"] = _forecast_payload(peer_record, row)
            row["ego_forecast"] = _forecast_payload(ego_record, row)
            future_state = target_states.get((scene, receiver, observation_frame, target_id))
            if arrival_time > observation_time + 1e-6:
                row["censor_reason"] = "arrival_after_horizon"
                censored_rows += 1
            elif future_state is None:
                row["censor_reason"] = "not_observed_or_track_dead"
                censored_rows += 1
            else:
                row["matured"] = True
                row["valid_mask"] = True
                row["matched_receiver_track_id"] = int(future_state["receiver_local_track_id"])
                row["realized_state"] = _realized_in_receiver_send_frame(future_state, send_pose)
                matured_rows += 1
            output.append(row)

    summary = {
        "schema_version": PAIRED_LEDGER_SCHEMA_VERSION,
        "common_pairing_rows": common_pairing_rows,
        "paired_rows": len(output),
        "matured_rows": matured_rows,
        "censored_rows": censored_rows,
        "unmatched_peer_prediction_keys": len(unmatched_peer_keys),
        "unmatched_ego_prediction_keys": len(unmatched_ego_keys),
        "missing_canonical_send_target_keys": len(missing_send_target_keys),
        "scientific_provenance_required": "PASS",
        "uses_gt": False,
    }
    return output, summary
