"""GT-free ingestion of real CMP/MTR multimodal predictions.

The official CMP/MTR evaluator writes convenience fields such as ``object_id``
and ``gt_trajs``.  Those fields are useful for its native metric code, but are
not legal inputs to the RV-HCA ledger.  This module therefore accepts only a
small normalized prediction contract and fails closed when a prediction file
still contains GT-derived identity fields.

Prediction records are model outputs, not labels.  A record must carry a
source-local track id that was emitted by the GT-free MTR input adapter (or by
an equivalent source-side inference wrapper).  Mapping an official output's
array index to ``object_id`` is deliberately not supported.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from rvhca_cpu.online import pose_to_world_matrix, transform_point


SCHEMA_VERSION = "rvhca.mtr_prediction.v0"

# Exact keys used by the official MTR/CMP output and data loader.  Keep this
# list explicit so benign online names such as receiver_target_id remain legal.
FORBIDDEN_GT_KEYS = frozenset(
    {
        "object_id",
        "object_ids",
        "gt_object_id",
        "gt_object_ids",
        "matched_car_id",
        "gt_trajs",
        "center_gt_trajs",
        "center_gt_trajs_src",
        "center_gt_trajs_mask",
        "center_gt_final_valid_idx",
        "obj_trajs_future_state",
        "obj_trajs_future_mask",
        "track_index_to_predict",
        "gt_ids",
    }
)


class MTRContractError(ValueError):
    """Raised when a prediction artifact cannot be used without GT leakage."""


def _scan_forbidden(value: Any, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_GT_KEYS:
                raise MTRContractError("forbidden GT-derived key at %s.%s" % (path, key_text))
            _scan_forbidden(child, "%s.%s" % (path, key_text))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan_forbidden(child, "%s[%d]" % (path, index))


def _finite_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise MTRContractError("%s must be non-empty and finite" % name)
    return array


def _record_list(payload: Any, path: Path) -> List[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        if "records" in payload:
            payload = payload["records"]
        elif "predictions" in payload:
            payload = payload["predictions"]
        else:
            payload = [payload]
    if not isinstance(payload, (list, tuple)):
        raise MTRContractError("%s must contain a record or a list of records" % path)
    records = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise MTRContractError("%s[%d] is not a mapping" % (path, index))
        records.append(item)
    return records


def load_prediction_records(paths: Sequence[Path]) -> List[Mapping[str, Any]]:
    """Load normalized JSON/JSONL/Pickle records without stripping fields."""

    output: List[Mapping[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            children = sorted(
                child
                for child in path.iterdir()
                if child.suffix.lower() in {".json", ".jsonl", ".pkl", ".pickle"}
            )
            output.extend(load_prediction_records(children))
            continue
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise MTRContractError("invalid JSON at %s:%d" % (path, line_no)) from exc
                    output.extend(_record_list(item, path))
        elif suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                output.extend(_record_list(json.load(handle), path))
        elif suffix in {".pkl", ".pickle"}:
            with path.open("rb") as handle:
                output.extend(_record_list(pickle.load(handle), path))
        else:
            raise MTRContractError("unsupported prediction extension: %s" % path)
    return output


def iter_prediction_records(paths: Sequence[Path]) -> Iterable[Mapping[str, Any]]:
    """Yield normalized-input records without retaining the whole file.

    This is the scalable counterpart to :func:`load_prediction_records` for
    JSONL inference exports.  It keeps the contract checks in the merge step
    while avoiding a second in-memory copy of multimodal trajectories.
    """
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            children = sorted(
                child
                for child in path.iterdir()
                if child.suffix.lower() in {".json", ".jsonl", ".pkl", ".pickle"}
            )
            yield from iter_prediction_records(children)
            continue
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise MTRContractError("invalid JSON at %s:%d" % (path, line_no)) from exc
                    for record in _record_list(item, path):
                        yield record
        elif suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                for record in _record_list(json.load(handle), path):
                    yield record
        elif suffix in {".pkl", ".pickle"}:
            with path.open("rb") as handle:
                for record in _record_list(pickle.load(handle), path):
                    yield record
        else:
            raise MTRContractError("unsupported prediction extension: %s" % path)


def normalize_prediction_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize one source/aggregate MTR record.

    Required identity is source-local: ``source_track_id`` for peer/ego
    records, or ``receiver_target_id`` for a receiver-side CMP aggregate.
    ``object_id`` is never used as a fallback.
    """

    _scan_forbidden(record)
    if bool(record.get("uses_gt", False)):
        raise MTRContractError("prediction record declares uses_gt=true")
    required = ("sequence_id", "send_frame_idx", "send_time", "pred_trajs")
    missing = [key for key in required if key not in record]
    if missing:
        raise MTRContractError("missing required fields: %s" % ", ".join(missing))

    role = str(record.get("role", "peer"))
    if role not in {"peer", "ego", "aggregate"}:
        raise MTRContractError("role must be peer, ego, or aggregate")
    if role in {"peer", "ego"} and "source" not in record:
        raise MTRContractError("source is required for %s prediction" % role)
    if role in {"peer", "ego"} and "source_track_id" not in record:
        raise MTRContractError("source_track_id is required; array-index/object_id mapping is forbidden")
    if role == "aggregate" and "receiver_target_id" not in record:
        raise MTRContractError("receiver_target_id is required for aggregate prediction")

    trajectories = _finite_array(record["pred_trajs"], "pred_trajs")
    if trajectories.ndim != 3 or trajectories.shape[0] < 1 or trajectories.shape[1] < 1:
        raise MTRContractError("pred_trajs must have shape (modes, timesteps, coordinates)")
    if trajectories.shape[2] not in {2, 3, 5, 7}:
        raise MTRContractError("pred_trajs last dimension must be 2, 3, 5, or 7")

    scores_value = record.get("pred_scores", record.get("scores"))
    if scores_value is None:
        scores = np.full((trajectories.shape[0],), 1.0 / trajectories.shape[0], dtype=np.float64)
    else:
        scores = _finite_array(scores_value, "pred_scores").reshape(-1)
        if len(scores) != trajectories.shape[0]:
            raise MTRContractError("pred_scores length does not match number of modes")
        if (scores < 0).any():
            raise MTRContractError("pred_scores must be non-negative")
        score_sum = float(scores.sum())
        scores = scores / score_sum if score_sum > 0 else np.full_like(scores, 1.0 / len(scores))

    offsets_value = record.get("time_offsets_s")
    if offsets_value is None:
        offsets = np.arange(1, trajectories.shape[1] + 1, dtype=np.float64) * 0.1
    else:
        offsets = _finite_array(offsets_value, "time_offsets_s").reshape(-1)
        if len(offsets) != trajectories.shape[1] or (offsets <= 0).any() or np.any(np.diff(offsets) <= 0):
            raise MTRContractError("time_offsets_s must be strictly increasing and match timesteps")

    normalized: Dict[str, Any] = {
        "schema_version": str(record.get("schema_version", SCHEMA_VERSION)),
        "sequence_id": str(record["sequence_id"]),
        "send_frame_idx": int(record["send_frame_idx"]),
        "send_time": float(record["send_time"]),
        "role": role,
        "forecast_frame": str(record.get("forecast_frame", "world")),
        "pred_trajs": trajectories.tolist(),
        "pred_scores": scores.tolist(),
        "time_offsets_s": offsets.tolist(),
        "model": str(record.get("model", "cmp_mtr")),
    }
    if normalized["forecast_frame"] not in {"world", "source@send_time", "receiver@send_time"}:
        raise MTRContractError("unsupported forecast_frame: %s" % normalized["forecast_frame"])
    if role in {"peer", "ego"}:
        normalized["source"] = str(record["source"])
        normalized["source_track_id"] = str(record["source_track_id"])
    if role == "aggregate":
        normalized["receiver"] = str(record.get("receiver", ""))
        normalized["receiver_target_id"] = str(record["receiver_target_id"])
    if "receiver" in record:
        normalized["receiver"] = str(record["receiver"])
    if "source_pose_at_send" in record:
        pose = _finite_array(record["source_pose_at_send"], "source_pose_at_send").reshape(-1)
        if len(pose) != 6:
            raise MTRContractError("source_pose_at_send must have six values")
        normalized["source_pose_at_send"] = pose.tolist()
    return normalized


def normalize_records(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [normalize_prediction_record(record) for record in records]


def _xyz(trajectory_point: Sequence[float]) -> np.ndarray:
    point = np.asarray(trajectory_point, dtype=np.float64).reshape(-1)
    if len(point) >= 3:
        return point[:3]
    return np.array([point[0], point[1], 0.0], dtype=np.float64)


def _transform_trajectory_to_receiver(
    trajectory: np.ndarray,
    frame: str,
    ledger_row: Mapping[str, Any],
    record: Mapping[str, Any],
) -> np.ndarray:
    if frame == "receiver@send_time":
        return np.stack([_xyz(point) for point in trajectory.reshape(-1, trajectory.shape[-1])]).reshape(
            trajectory.shape[0], trajectory.shape[1], 3
        )
    receiver_pose = ledger_row.get("receiver_pose_at_send")
    if receiver_pose is None:
        raise MTRContractError("ledger row has no receiver_pose_at_send for world/source forecast")
    receiver_inverse = np.linalg.inv(pose_to_world_matrix(receiver_pose))
    if frame == "world":
        return np.stack(
            [transform_point(_xyz(point), receiver_inverse) for point in trajectory.reshape(-1, trajectory.shape[-1])]
        ).reshape(trajectory.shape[0], trajectory.shape[1], 3)
    source_pose = record.get("source_pose_at_send")
    if source_pose is None:
        raise MTRContractError("source@send_time requires source_pose_at_send")
    source_to_world = pose_to_world_matrix(source_pose)
    return np.stack(
        [
            transform_point(transform_point(_xyz(point), source_to_world), receiver_inverse)
            for point in trajectory.reshape(-1, trajectory.shape[-1])
        ]
    ).reshape(trajectory.shape[0], trajectory.shape[1], 3)


def _forecast_payload(record: Mapping[str, Any], row: Mapping[str, Any]) -> Dict[str, Any]:
    trajectories = np.asarray(record["pred_trajs"], dtype=np.float64)
    trajectories_receiver = _transform_trajectory_to_receiver(
        trajectories, str(record["forecast_frame"]), row, record
    )
    offsets = np.asarray(record["time_offsets_s"], dtype=np.float64)
    horizon = float(row["horizon"])
    timestep = int(np.argmin(np.abs(offsets - horizon)))
    mode = int(np.argmax(np.asarray(record["pred_scores"], dtype=np.float64)))
    position = trajectories_receiver[mode, timestep].tolist()
    return {
        "frame": "receiver@send_time",
        "position_xyz": position,
        "trajectories_xyz": trajectories_receiver.tolist(),
        "scores": np.asarray(record["pred_scores"], dtype=np.float64).tolist(),
        "time_offsets_s": np.asarray(record["time_offsets_s"], dtype=np.float64).tolist(),
        "selected_mode": mode,
        "model": str(record["model"]),
        "source_prediction_role": str(record["role"]),
    }


def merge_predictions_into_ledger(
    ledger_rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return a derived ledger with peer/ego/aggregate MTR payloads attached.

    The input ledger is never modified.  Peer records match by source-local
    identity and send frame.  Aggregate records match by receiver target and
    send frame.  No GT field is read or inferred in either case.
    """

    normalized = normalize_records(records)
    peer_index: MutableMapping[Tuple[str, str, int, str], List[Mapping[str, Any]]] = {}
    aggregate_index: MutableMapping[Tuple[str, str, int, str], List[Mapping[str, Any]]] = {}
    for record in normalized:
        role = str(record["role"])
        if role in {"peer", "ego"}:
            key = (
                str(record["sequence_id"]),
                str(record["source"]),
                int(record["send_frame_idx"]),
                str(record["source_track_id"]),
            )
            peer_index.setdefault(key, []).append(record)
        else:
            key = (
                str(record["sequence_id"]),
                str(record.get("receiver", "")),
                int(record["send_frame_idx"]),
                str(record["receiver_target_id"]),
            )
            aggregate_index.setdefault(key, []).append(record)

    output: List[Dict[str, Any]] = []
    peer_attached = 0
    aggregate_attached = 0
    peer_rows = 0
    aggregate_rows = 0
    ego_attached = 0
    matched_peer_keys = set()
    matched_ego_keys = set()
    for original in ledger_rows:
        row = dict(original)
        sequence = str(row.get("sequence_id"))
        source = str(row.get("source"))
        frame = int(row.get("send_frame_idx", -1))
        source_track = str(row.get("source_track_id"))
        # A source-local key can legitimately have one peer record and one ego
        # record (when the receiver is also exporting its own forecast).  The
        # two roles share the same identity tuple but are attached to different
        # ledger fields, so filter before checking ambiguity.
        peer_records = [
            record
            for record in peer_index.get((sequence, source, frame, source_track), [])
            if str(record.get("role")) == "peer"
        ]
        if source != str(row.get("receiver")) and peer_records:
            peer_rows += 1
            if len(peer_records) != 1:
                raise MTRContractError(
                    "ambiguous peer records for %s/%s/frame=%d/track=%s"
                    % (sequence, source, frame, source_track)
                )
            row["forecast"] = _forecast_payload(peer_records[0], row)
            peer_attached += 1
            matched_peer_keys.add((sequence, source, frame, source_track))
        ego_records = peer_index.get(
            (sequence, str(row.get("receiver")), frame, str(row.get("receiver_local_track_id"))), []
        )
        ego_records = [record for record in ego_records if str(record.get("role")) == "ego"]
        if ego_records:
            if len(ego_records) != 1:
                raise MTRContractError("ambiguous ego records for %s/frame=%d" % (sequence, frame))
            row["ego_forecast"] = _forecast_payload(ego_records[0], row)
            ego_attached += 1
            matched_ego_keys.add(
                (sequence, str(row.get("receiver")), frame, str(row.get("receiver_local_track_id")))
            )
        aggregate_records = aggregate_index.get(
            (sequence, str(row.get("receiver")), frame, str(row.get("receiver_target_id"))), []
        )
        if aggregate_records:
            aggregate_rows += 1
            if len(aggregate_records) != 1:
                raise MTRContractError("ambiguous aggregate records for %s/frame=%d" % (sequence, frame))
            row["cmp_aggregate_forecast"] = _forecast_payload(aggregate_records[0], row)
            # Keep a generic aggregate alias for existing evaluators.
            row["aggregate_forecast"] = dict(row["cmp_aggregate_forecast"])
            aggregate_attached += 1
        output.append(row)

    peer_key_count = sum(
        any(str(record.get("role")) == "peer" for record in values)
        for values in peer_index.values()
    )
    ego_key_count = sum(
        any(str(record.get("role")) == "ego" for record in values)
        for values in peer_index.values()
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "ledger_rows": len(output),
        "normalized_records": len(normalized),
        "peer_record_keys": len(peer_index),
        "aggregate_record_keys": len(aggregate_index),
        "peer_ledger_rows_with_match": peer_rows,
        "peer_attached_rows": peer_attached,
        "ego_attached_rows": ego_attached,
        "aggregate_ledger_rows_with_match": aggregate_rows,
        "aggregate_attached_rows": aggregate_attached,
        "unmatched_peer_keys": max(0, peer_key_count - len(matched_peer_keys)),
        "unmatched_ego_keys": max(0, ego_key_count - len(matched_ego_keys)),
    }
    return output, summary


def merge_predictions_into_ledger_stream(
    ledger_path: Path,
    records: Iterable[Mapping[str, Any]],
    output_path: Path,
) -> Dict[str, Any]:
    """Attach predictions while streaming the ledger JSONL.

    Large MTR exports contain multimodal trajectories on every attached row.
    The original list-returning merge remains useful for small unit tests, but
    a multi-scene replay can exceed a small container's memory limit if both
    the base ledger and derived rows are materialized.  This variant keeps a
    compact float32 prediction index and writes each derived row immediately.
    """
    peer_index: MutableMapping[Tuple[str, str, int, str], List[Mapping[str, Any]]] = {}
    aggregate_index: MutableMapping[Tuple[str, str, int, str], List[Mapping[str, Any]]] = {}
    normalized_count = 0
    for raw_record in records:
        normalized = normalize_prediction_record(raw_record)
        # Arrays are only needed for geometric transformation during attach;
        # float32 storage avoids retaining thousands of Python float objects.
        normalized["pred_trajs"] = np.asarray(normalized["pred_trajs"], dtype=np.float32)
        normalized["pred_scores"] = np.asarray(normalized["pred_scores"], dtype=np.float32)
        normalized["time_offsets_s"] = np.asarray(normalized["time_offsets_s"], dtype=np.float32)
        normalized_count += 1
        role = str(normalized["role"])
        if role in {"peer", "ego"}:
            key = (
                str(normalized["sequence_id"]),
                str(normalized["source"]),
                int(normalized["send_frame_idx"]),
                str(normalized["source_track_id"]),
            )
            peer_index.setdefault(key, []).append(normalized)
        else:
            key = (
                str(normalized["sequence_id"]),
                str(normalized.get("receiver", "")),
                int(normalized["send_frame_idx"]),
                str(normalized["receiver_target_id"]),
            )
            aggregate_index.setdefault(key, []).append(normalized)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_rows = 0
    peer_attached = 0
    aggregate_attached = 0
    peer_rows = 0
    aggregate_rows = 0
    ego_attached = 0
    matched_peer_keys = set()
    matched_ego_keys = set()
    with Path(ledger_path).open("r", encoding="utf-8") as source_handle, output_path.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line in source_handle:
            if not line.strip():
                continue
            original = json.loads(line)
            row = dict(original)
            sequence = str(row.get("sequence_id"))
            source = str(row.get("source"))
            frame = int(row.get("send_frame_idx", -1))
            source_track = str(row.get("source_track_id"))
            peer_records = [
                record
                for record in peer_index.get((sequence, source, frame, source_track), [])
                if str(record.get("role")) == "peer"
            ]
            if source != str(row.get("receiver")) and peer_records:
                peer_rows += 1
                if len(peer_records) != 1:
                    raise MTRContractError(
                        "ambiguous peer records for %s/%s/frame=%d/track=%s"
                        % (sequence, source, frame, source_track)
                    )
                row["forecast"] = _forecast_payload(peer_records[0], row)
                peer_attached += 1
                matched_peer_keys.add((sequence, source, frame, source_track))
            ego_records = peer_index.get(
                (sequence, str(row.get("receiver")), frame, str(row.get("receiver_local_track_id"))), []
            )
            ego_records = [record for record in ego_records if str(record.get("role")) == "ego"]
            if ego_records:
                if len(ego_records) != 1:
                    raise MTRContractError("ambiguous ego records for %s/frame=%d" % (sequence, frame))
                row["ego_forecast"] = _forecast_payload(ego_records[0], row)
                ego_attached += 1
                matched_ego_keys.add(
                    (sequence, str(row.get("receiver")), frame, str(row.get("receiver_local_track_id")))
                )
            aggregate_records = aggregate_index.get(
                (sequence, str(row.get("receiver")), frame, str(row.get("receiver_target_id"))), []
            )
            if aggregate_records:
                aggregate_rows += 1
                if len(aggregate_records) != 1:
                    raise MTRContractError("ambiguous aggregate records for %s/frame=%d" % (sequence, frame))
                row["cmp_aggregate_forecast"] = _forecast_payload(aggregate_records[0], row)
                row["aggregate_forecast"] = dict(row["cmp_aggregate_forecast"])
                aggregate_attached += 1
            output_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            ledger_rows += 1

    peer_key_count = sum(
        any(str(record.get("role")) == "peer" for record in values)
        for values in peer_index.values()
    )
    ego_key_count = sum(
        any(str(record.get("role")) == "ego" for record in values)
        for values in peer_index.values()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "ledger_rows": ledger_rows,
        "normalized_records": normalized_count,
        "peer_record_keys": len(peer_index),
        "aggregate_record_keys": len(aggregate_index),
        "peer_ledger_rows_with_match": peer_rows,
        "peer_attached_rows": peer_attached,
        "ego_attached_rows": ego_attached,
        "aggregate_ledger_rows_with_match": aggregate_rows,
        "aggregate_attached_rows": aggregate_attached,
        "unmatched_peer_keys": max(0, peer_key_count - len(matched_peer_keys)),
        "unmatched_ego_keys": max(0, ego_key_count - len(matched_ego_keys)),
        "merge_mode": "streaming_ledger_compact_prediction_index",
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
