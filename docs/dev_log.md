# Development log

## 2026-09-05 — cross-replay GT-free data contract

- Persisted one `rvhca.receiver_target_state.v1` row for every canonical
  PointPillar local-track row at the causal `ReceiverTargetManager.update()`
  point.
- Added arrival-only CoBEVT-to-PointPillar pairing.  It obtains one canonical
  `receiver_target_id` at arrival, then performs same-ID lookup at send and
  future; it does not reuse numerical AB3DMOT IDs across replays.
- Added a scientific paired MTR ledger requiring `PASS` provenance.  It emits
  realized/censored future state through the same canonical ID and leaves all
  errors and Harm Rate unset for later offline analysis.
- Added an offline-only raw-GT audit for conditional pairing precision/recall
  and end-to-end common coverage.
- No reliability controller, A/B/C evaluation, or aggregation training was
  added in this change.
- This environment exposes no GPU (`nvidia-smi` reports no devices), so no
  standalone MTR smoke or real paired-MTR export was run here.
