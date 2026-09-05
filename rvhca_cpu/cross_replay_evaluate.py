"""Offline-only GT audit for the cross-replay pairing artifact.

This module is intentionally separate from the causal replay, pairing, and
paired-MTR-ledger modules.  Raw OPV2V YAML labels are used only to audit a
finished artifact; none of the returned values are written back to online
inputs.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from rvhca_cpu.evaluate import RawGTIndex, _build_track_matches


class CrossReplayAuditError(ValueError):
    """Raised when a supposedly GT-free pairing artifact is invalid for audit."""


def _require_gt_free(rows: Sequence[Mapping[str, Any]], label: str) -> None:
    for row in rows:
        if bool(row.get("uses_gt", False)):
            raise CrossReplayAuditError("%s declares uses_gt=true" % label)
        if str(row.get("tracking_frame", "world_fixed")) != "world_fixed":
            raise CrossReplayAuditError("%s is not in world_fixed tracking frame" % label)


def _local_track_index(
    rows: Sequence[Mapping[str, Any]], label: str
) -> Dict[Tuple[str, str, int, int], Mapping[str, Any]]:
    output: Dict[Tuple[str, str, int, int], Mapping[str, Any]] = {}
    for row in rows:
        try:
            key = (
                str(row["sequence_id"]),
                str(row["source"]),
                int(row["frame_idx"]),
                int(row["local_track_id"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CrossReplayAuditError("invalid %s local-track key" % label) from exc
        if key in output:
            raise CrossReplayAuditError("duplicate %s local-track key" % label)
        output[key] = row
    return output


def _group_tracks(
    rows: Sequence[Mapping[str, Any]]
) -> Dict[Tuple[str, str, int], List[Mapping[str, Any]]]:
    output: MutableMapping[Tuple[str, str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        output[(str(row["sequence_id"]), str(row["source"]), int(row["frame_idx"]))].append(row)
    return dict(output)


def _pair_offsets(
    pairing_rows: Sequence[Mapping[str, Any]]
) -> Dict[Tuple[str, str, str], int]:
    output: Dict[Tuple[str, str, str], int] = {}
    for row in pairing_rows:
        try:
            key = (str(row["sequence_id"]), str(row["receiver"]), str(row["source"]))
            offset = int(row["arrival_frame_idx"]) - int(row["send_frame_idx"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CrossReplayAuditError("invalid pairing frame key") from exc
        previous = output.get(key)
        if previous is not None and previous != offset:
            raise CrossReplayAuditError("pairing has inconsistent arrival offsets for one receiver/source")
        output[key] = offset
    return output


def _key_from_pairing(row: Mapping[str, Any], peer_gt_id: str) -> Tuple[str, str, str, int, str]:
    return (
        str(row["sequence_id"]),
        str(row["receiver"]),
        str(row["source"]),
        int(row["send_frame_idx"]),
        str(peer_gt_id),
    )


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


def evaluate_cross_replay_pairing(
    pairing_rows: Sequence[Mapping[str, Any]],
    canonical_tracks: Sequence[Mapping[str, Any]],
    peer_tracks: Sequence[Mapping[str, Any]],
    raw_root: Path,
    *,
    association_gate_m: float = 4.0,
) -> Dict[str, Any]:
    """Audit cross-replay association accuracy using raw GT only offline.

    ``conditional_pairing_recall`` is conditioned on independently GT-matched
    peer and canonical receiver tracks at arrival.  ``end_to_end_coverage``
    uses every raw-GT object visible to both source at send and receiver at
    arrival over the replayed frame range; its numerator additionally requires
    a correct *common* pairing, because only common rows are usable by the
    scientific ego-vs-peer paired ledger.
    """

    if association_gate_m <= 0.0:
        raise CrossReplayAuditError("association_gate_m must be positive")
    _require_gt_free(pairing_rows, "pairing row")
    _require_gt_free(canonical_tracks, "canonical track")
    _require_gt_free(peer_tracks, "peer track")
    canonical_index = _local_track_index(canonical_tracks, "canonical")
    peer_index = _local_track_index(peer_tracks, "peer")
    canonical_by_frame = _group_tracks(canonical_tracks)
    peer_by_frame = _group_tracks(peer_tracks)
    offsets = _pair_offsets(pairing_rows)
    gt_index = RawGTIndex(Path(raw_root))
    peer_gt_matches, _, _ = _build_track_matches(peer_tracks, gt_index, association_gate_m)
    canonical_gt_matches, _, _ = _build_track_matches(canonical_tracks, gt_index, association_gate_m)

    conditional_opportunities: Set[Tuple[str, str, str, int, str]] = set()
    max_peer_frame: MutableMapping[Tuple[str, str], int] = defaultdict(lambda: -1)
    max_canonical_frame: MutableMapping[Tuple[str, str], int] = defaultdict(lambda: -1)
    for scene, source, frame, _ in peer_index:
        max_peer_frame[(scene, source)] = max(max_peer_frame[(scene, source)], frame)
    for scene, receiver, frame, _ in canonical_index:
        max_canonical_frame[(scene, receiver)] = max(max_canonical_frame[(scene, receiver)], frame)

    # This denominator intentionally comes from independently matched local
    # tracks rather than accepted pairing rows.  Thus an association rejection
    # remains a false negative instead of silently disappearing from recall.
    for (scene, receiver, source), offset in offsets.items():
        for (peer_scene, peer_source, send_frame), rows in peer_by_frame.items():
            if peer_scene != scene or peer_source != source:
                continue
            arrival_rows = canonical_by_frame.get((scene, receiver, send_frame + offset), [])
            if not arrival_rows:
                continue
            receiver_gt_ids = {
                canonical_gt_matches.get((scene, receiver, send_frame + offset, int(item["local_track_id"])))
                for item in arrival_rows
            }
            receiver_gt_ids.discard(None)
            for peer_row in rows:
                peer_gt_id = peer_gt_matches.get(
                    (scene, source, send_frame, int(peer_row["local_track_id"]))
                )
                if peer_gt_id is not None and peer_gt_id in receiver_gt_ids:
                    conditional_opportunities.add((scene, receiver, source, send_frame, str(peer_gt_id)))

    accepted_evaluable_rows = 0
    correct_accepted_rows = 0
    correct_accepted_keys: Set[Tuple[str, str, str, int, str]] = set()
    correct_common_keys: Set[Tuple[str, str, str, int, str]] = set()
    common_rows = 0
    shared_only_rows = 0
    rejected_rows = 0
    for row in pairing_rows:
        target_scope = str(row.get("target_scope"))
        if target_scope == "common":
            common_rows += 1
        elif target_scope == "shared_only":
            shared_only_rows += 1
        else:
            rejected_rows += 1
        if target_scope not in {"common", "shared_only"}:
            continue
        if row.get("arrival_receiver_local_track_id") is None:
            continue
        try:
            peer_key = (
                str(row["sequence_id"]),
                str(row["source"]),
                int(row["send_frame_idx"]),
                int(row["peer_replay_source_track_id"]),
            )
            receiver_key = (
                str(row["sequence_id"]),
                str(row["receiver"]),
                int(row["arrival_frame_idx"]),
                int(row["arrival_receiver_local_track_id"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CrossReplayAuditError("invalid pairing local-track key") from exc
        # Ensure the audit never treats an arbitrary ID as present merely
        # because it appeared in a pairing JSONL row.
        if peer_key not in peer_index or receiver_key not in canonical_index:
            raise CrossReplayAuditError("pairing references a local track absent from its replay")
        peer_gt_id = peer_gt_matches.get(peer_key)
        receiver_gt_id = canonical_gt_matches.get(receiver_key)
        if peer_gt_id is None or receiver_gt_id is None:
            continue
        accepted_evaluable_rows += 1
        if peer_gt_id == receiver_gt_id:
            correct_accepted_rows += 1
            key = _key_from_pairing(row, peer_gt_id)
            correct_accepted_keys.add(key)
            if target_scope == "common":
                correct_common_keys.add(key)

    raw_receiver_visible_opportunities: Set[Tuple[str, str, str, int, str]] = set()
    for (scene, receiver, source), offset in offsets.items():
        maximum_send_frame = min(
            max_peer_frame.get((scene, source), -1),
            max_canonical_frame.get((scene, receiver), -1) - offset,
        )
        for send_frame in range(maximum_send_frame + 1):
            peer_gt = gt_index.by_frame(scene, source, send_frame)
            receiver_gt = gt_index.by_frame(scene, receiver, send_frame + offset)
            for gt_id in set(peer_gt).intersection(receiver_gt):
                raw_receiver_visible_opportunities.add((scene, receiver, source, send_frame, str(gt_id)))

    correct_conditional = len(correct_accepted_keys.intersection(conditional_opportunities))
    correct_end_to_end_common = len(correct_common_keys.intersection(raw_receiver_visible_opportunities))
    correct_end_to_end_arrival = len(correct_accepted_keys.intersection(raw_receiver_visible_opportunities))
    return {
        "schema_version": "rvhca.cross_replay_offline_audit.v1",
        "offline_only": True,
        "uses_gt": True,
        "pairing_rows": len(pairing_rows),
        "common_rows": common_rows,
        "shared_only_rows": shared_only_rows,
        "rejected_rows": rejected_rows,
        "accepted_evaluable_rows": accepted_evaluable_rows,
        "correct_accepted_rows": correct_accepted_rows,
        "conditional_pairing_precision": _rate(correct_accepted_rows, accepted_evaluable_rows),
        "conditional_pairing_recall": _rate(correct_conditional, len(conditional_opportunities)),
        "conditional_pairing_opportunities": len(conditional_opportunities),
        "correct_conditional_pairings": correct_conditional,
        "raw_receiver_visible_opportunities": len(raw_receiver_visible_opportunities),
        "correct_end_to_end_common_pairs": correct_end_to_end_common,
        "end_to_end_common_coverage": _rate(
            correct_end_to_end_common, len(raw_receiver_visible_opportunities)
        ),
        "end_to_end_arrival_pairing_coverage": _rate(
            correct_end_to_end_arrival, len(raw_receiver_visible_opportunities)
        ),
        "end_to_end_coverage": _rate(correct_end_to_end_common, len(raw_receiver_visible_opportunities)),
        "conditional_precision_definition": (
            "correct accepted arrival associations / accepted rows whose peer and receiver local tracks "
            "both have independent offline GT matches"
        ),
        "conditional_recall_definition": (
            "correct accepted arrival associations / independently GT-matched peer-to-receiver local-track "
            "opportunities at arrival"
        ),
        "end_to_end_coverage_definition": (
            "correct common pairings / raw-GT objects visible to peer at send and receiver at arrival over "
            "the replayed frame range; common is required because only common rows enter the paired MTR ledger"
        ),
    }
