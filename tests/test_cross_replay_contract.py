"""Regression tests for the GT-free cross-replay data contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from rvhca_cpu.online import TrackRow, build_replay, write_replay
from rvhca_cpu.cross_replay import build_cross_replay_pairings
from rvhca_cpu.cross_replay_evaluate import evaluate_cross_replay_pairing
from rvhca_cpu.mtr_io import MTRContractError, normalize_prediction_record
from rvhca_cpu.paired_mtr import build_paired_mtr_ledger


def _track(source: str, frame: int, local_id: int, center_x: float) -> TrackRow:
    return TrackRow(
        sequence_id="scene",
        source=source,
        frame_idx=frame,
        timestamp_key="%06d" % frame,
        time_s=frame * 0.1,
        local_track_id=local_id,
        center_local=[center_x, 0.0, 0.0],
        center_world=[center_x, 0.0, 0.0],
        dims_hwl=[1.5, 2.0, 4.0],
        yaw_local=0.0,
        yaw_world=0.0,
        score=0.9,
        pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        velocity_world=[1.0, 0.0, 0.0],
    )


def _detection_cache() -> dict:
    frames = {
        str(frame): {
            "timestamp_idx": frame,
            "timestamp_key": "%06d" % frame,
            "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "boxes3d": [],
            "boxes2d": [],
            "scores": [],
        }
        for frame in (0, 1)
    }
    return {"scenes": {"scene": {"ego": dict(frames), "peer": dict(frames)}}}


def _target_state(frame: int, local_id: int, target_id: int, center_x: float) -> dict:
    return {
        "schema_version": "rvhca.receiver_target_state.v1",
        "sequence_id": "scene",
        "receiver": "ego",
        "frame_idx": frame,
        "timestamp_key": "%06d" % frame,
        "receiver_local_track_id": local_id,
        "receiver_target_id": target_id,
        "center_world": [center_x, 0.0, 0.0],
        "velocity_world": [1.0, 0.0, 0.0],
        "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "track_age": frame + 1,
        "miss_count": 0,
        "tracking_frame": "world_fixed",
        "uses_gt": False,
    }


def _peer_track(local_id: int = 17) -> dict:
    return {
        "sequence_id": "scene",
        "source": "peer",
        "frame_idx": 0,
        "timestamp_key": "000000",
        "local_track_id": local_id,
        "center_world": [0.0, 0.0, 0.0],
        "velocity_world": [1.0, 0.0, 0.0],
        "tracking_frame": "world_fixed",
        "uses_gt": False,
    }


def _prediction_record(
    *,
    source: str = "peer",
    source_track_id: str = "17",
    role: str = "peer",
    status: str = "PASS",
    history_family: str = "corpbevtlidar_delay_1_frame_aug_c256",
) -> dict:
    return {
        "schema_version": "rvhca.mtr_prediction.v1",
        "sequence_id": "scene",
        "source": source,
        "source_track_id": source_track_id,
        "send_frame_idx": 0,
        "send_time": 0.0,
        "role": role,
        "forecast_frame": "world",
        "pred_trajs": [[[0.1, 0.0, 0.0]]],
        "pred_scores": [1.0],
        "time_offsets_s": [0.1],
        "model": "cmp_mtr_no_agg" if role == "peer" else "cmp_mtr_no_coop",
        "track_history_family": history_family,
        "checkpoint_history_family": history_family,
        "input_distribution_status": status,
        "history_reference_frame": "world",
        "uses_gt": False,
    }


def _common_pairing(peer_track_id: str = "17") -> dict:
    return {
        "schema_version": "rvhca.cross_replay_pairing.v1",
        "sequence_id": "scene",
        "receiver": "ego",
        "source": "peer",
        "send_frame_idx": 0,
        "arrival_frame_idx": 1,
        "peer_replay_source_track_id": peer_track_id,
        "canonical_receiver_target_id": 9,
        "arrival_receiver_local_track_id": 4,
        "send_receiver_local_track_id": 3,
        "association_cost_m": 0.0,
        "association_gate_m": 0.5,
        "target_scope": "common",
        "pairing_status": "common",
        "censor_reason": None,
        "uses_gt": False,
    }


def _paired_states() -> list:
    return [
        _target_state(0, 3, 9, 0.0),
        _target_state(1, 4, 9, 0.1),
        _target_state(2, 5, 9, 0.2),
    ]


def _two_step_record(**kwargs) -> dict:
    record = _prediction_record(**kwargs)
    record["pred_trajs"] = [[[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]]
    record["time_offsets_s"] = [0.1, 0.2]
    return record


class CrossReplayContractTest(unittest.TestCase):
    def test_build_replay_persists_one_target_state_per_receiver_track(self) -> None:
        def fake_track_source(sequence_id, source, frame_map, **_kwargs):
            offset = 0.0 if source == "ego" else 10.0
            return [
                _track(source, 0, 3, offset),
                _track(source, 1, 4, offset + 0.1),
            ]

        with patch("rvhca_cpu.online.track_source", side_effect=fake_track_source):
            replay = build_replay(
                _detection_cache(),
                aggregate_mode="none",
                horizons_s=(0.1,),
            )

        states = replay["receiver_target_states"]
        self.assertEqual(len(states), len(replay["tracks"]))
        state_keys = {
            (
                item["sequence_id"],
                item["receiver"],
                item["frame_idx"],
                item["receiver_local_track_id"],
            )
            for item in states
        }
        track_keys = {
            (item["sequence_id"], item["source"], item["frame_idx"], item["local_track_id"])
            for item in replay["tracks"]
        }
        self.assertEqual(state_keys, track_keys)
        self.assertTrue(all(item["uses_gt"] is False for item in states))
        with tempfile.TemporaryDirectory() as temp_dir:
            write_replay(Path(temp_dir), replay)
            self.assertTrue((Path(temp_dir) / "receiver_target_states.jsonl").is_file())

    def test_pairing_uses_arrival_geometry_then_same_target_id_at_send(self) -> None:
        rows, summary = build_cross_replay_pairings(
            [_target_state(0, 3, 9, 0.0), _target_state(1, 4, 9, 0.1)],
            [_peer_track()],
            frame_period_s=0.1,
            peer_delay_s=0.1,
            association_gate_m=0.5,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["canonical_receiver_target_id"], 9)
        self.assertEqual(rows[0]["arrival_receiver_local_track_id"], 4)
        self.assertEqual(rows[0]["send_receiver_local_track_id"], 3)
        self.assertEqual(rows[0]["target_scope"], "common")
        self.assertEqual(summary["receiver_expanded_candidates"], 1)
        self.assertEqual(summary["common_pairing_coverage"], 1.0)
        self.assertEqual(summary["arrival_pairing_coverage"], 1.0)

    def test_pairing_never_reassigns_at_send_when_target_id_is_absent(self) -> None:
        rows, _ = build_cross_replay_pairings(
            [_target_state(0, 3, 8, 0.0), _target_state(1, 4, 9, 0.1)],
            [_peer_track()],
            frame_period_s=0.1,
            peer_delay_s=0.1,
            association_gate_m=0.5,
        )
        self.assertEqual(rows[0]["canonical_receiver_target_id"], 9)
        self.assertEqual(rows[0]["target_scope"], "shared_only")
        self.assertIsNone(rows[0]["send_receiver_local_track_id"])
        self.assertEqual(rows[0]["censor_reason"], "receiver_target_not_observed_at_send")

    def test_pairing_geometry_is_invariant_to_consistent_peer_id_renumbering(self) -> None:
        states = [_target_state(0, 3, 9, 0.0), _target_state(1, 4, 9, 0.1)]
        original, original_summary = build_cross_replay_pairings(
            states, [_peer_track(17)], frame_period_s=0.1, peer_delay_s=0.1, association_gate_m=0.5
        )
        renumbered, renumbered_summary = build_cross_replay_pairings(
            states, [_peer_track(101)], frame_period_s=0.1, peer_delay_s=0.1, association_gate_m=0.5
        )
        projection = lambda rows: [
            (
                row["sequence_id"], row["receiver"], row["source"], row["send_frame_idx"],
                row["canonical_receiver_target_id"], row["arrival_receiver_local_track_id"],
                row["send_receiver_local_track_id"], row["association_cost_m"], row["target_scope"],
            )
            for row in rows
        ]
        self.assertEqual(projection(original), projection(renumbered))
        self.assertEqual(original_summary["common_pairs"], renumbered_summary["common_pairs"])

    def test_normalize_prediction_record_retains_provenance(self) -> None:
        normalized = normalize_prediction_record(_prediction_record())
        self.assertEqual(normalized["track_history_family"], "corpbevtlidar_delay_1_frame_aug_c256")
        self.assertEqual(normalized["checkpoint_history_family"], "corpbevtlidar_delay_1_frame_aug_c256")
        self.assertEqual(normalized["input_distribution_status"], "PASS")
        self.assertEqual(normalized["history_reference_frame"], "world")

    def test_paired_ledger_uses_same_target_id_for_send_and_future_state(self) -> None:
        rows, summary = build_paired_mtr_ledger(
            [_common_pairing()],
            _paired_states(),
            [_two_step_record()],
            [
                _two_step_record(
                    source="ego",
                    source_track_id="3",
                    role="ego",
                    history_family="point_pillar_sinbevt",
                )
            ],
            horizons_s=(0.2,),
            frame_period_s=0.1,
            peer_delay_s=0.1,
        )
        self.assertEqual(summary["paired_rows"], 1)
        self.assertEqual(rows[0]["receiver_target_id"], 9)
        self.assertEqual(rows[0]["receiver_local_track_id"], 3)
        self.assertEqual(rows[0]["matched_receiver_track_id"], 5)
        self.assertTrue(rows[0]["matured"])
        self.assertFalse(rows[0]["uses_gt"])
        self.assertEqual(rows[0]["realized_state"], [0.2, 0.0, 0.0])

    def test_unrenumbered_peer_prediction_is_unmatched_not_cross_attached(self) -> None:
        rows, summary = build_paired_mtr_ledger(
            [_common_pairing(peer_track_id="101")],
            _paired_states(),
            [_two_step_record(source_track_id="17")],
            [
                _two_step_record(
                    source="ego",
                    source_track_id="3",
                    role="ego",
                    history_family="point_pillar_sinbevt",
                )
            ],
            horizons_s=(0.2,),
            frame_period_s=0.1,
            peer_delay_s=0.1,
        )
        self.assertEqual(rows, [])
        self.assertEqual(summary["unmatched_peer_prediction_keys"], 1)

    def test_consistently_renumbered_peer_track_and_prediction_still_attach(self) -> None:
        pairing, _ = build_cross_replay_pairings(
            _paired_states(),
            [_peer_track(101)],
            frame_period_s=0.1,
            peer_delay_s=0.1,
            association_gate_m=0.5,
        )
        rows, summary = build_paired_mtr_ledger(
            pairing,
            _paired_states(),
            [_two_step_record(source_track_id="101")],
            [
                _two_step_record(
                    source="ego",
                    source_track_id="3",
                    role="ego",
                    history_family="point_pillar_sinbevt",
                )
            ],
            horizons_s=(0.2,),
            frame_period_s=0.1,
            peer_delay_s=0.1,
        )
        self.assertEqual(summary["unmatched_peer_prediction_keys"], 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["peer_replay_source_track_id"], "101")
        self.assertEqual(rows[0]["receiver_target_id"], 9)

    def test_paired_ledger_rejects_arrival_id_that_disagrees_with_canonical_state(self) -> None:
        pairing = _common_pairing()
        pairing["arrival_receiver_local_track_id"] = 99
        with self.assertRaises(MTRContractError):
            build_paired_mtr_ledger(
                [pairing],
                _paired_states(),
                [_two_step_record()],
                [
                    _two_step_record(
                        source="ego",
                        source_track_id="3",
                        role="ego",
                        history_family="point_pillar_sinbevt",
                    )
                ],
                horizons_s=(0.2,),
                frame_period_s=0.1,
                peer_delay_s=0.1,
            )

    def test_scientific_paired_ledger_rejects_nonpass_provenance(self) -> None:
        with self.assertRaises(MTRContractError):
            build_paired_mtr_ledger(
                [_common_pairing()],
                _paired_states(),
                [_two_step_record(status="MISMATCH")],
                [
                    _two_step_record(
                        source="ego",
                        source_track_id="3",
                        role="ego",
                        history_family="point_pillar_sinbevt",
                    )
                ],
                horizons_s=(0.2,),
                frame_period_s=0.1,
                peer_delay_s=0.1,
            )

    def test_offline_cross_replay_audit_counts_same_gt_pair_as_correct(self) -> None:
        peer = _track("peer", 0, 17, 0.0).as_dict()
        canonical = [_track("ego", 0, 3, 0.0).as_dict(), _track("ego", 1, 4, 0.1).as_dict()]
        pairing = _common_pairing()
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_root = Path(temp_dir)
            for source, frame, x in (("peer", 0, 0.0), ("ego", 0, 0.0), ("ego", 1, 0.1)):
                directory = raw_root / "scene" / source
                directory.mkdir(parents=True, exist_ok=True)
                (directory / ("%06d.yaml" % frame)).write_text(
                    yaml.safe_dump(
                        {"vehicles": {"vehicle": {"location": [x, 0.0, 0.0], "center": [0.0, 0.0, 0.0]}}}
                    ),
                    encoding="utf-8",
                )
            summary = evaluate_cross_replay_pairing(
                [pairing], canonical, [peer], raw_root, association_gate_m=1.0
            )
        self.assertEqual(summary["conditional_pairing_precision"], 1.0)
        self.assertEqual(summary["conditional_pairing_recall"], 1.0)
        self.assertEqual(summary["end_to_end_common_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
