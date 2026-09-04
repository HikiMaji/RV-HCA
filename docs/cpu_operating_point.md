# CPU asynchronous operating-point scan

Date: 2026-09-04

This is a 40-frame asynchronous probe on scene `2021_08_22_09_08_29`, using
the sanitized detector cache and the same GT-free source-local tracking,
receiver association, and ledger code as the formal replay.  Raw labels are
read only by the independent evaluator.  `mean_probe_not_cmp` is a constant-
velocity smoke aggregate and must not be interpreted as CMP output.

| detector threshold | track precision | track recall | association precision / conditional recall | end-to-end association recall | end-to-end opportunities | ledger coverage | EWMA gain vs ego (m) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.843 | 0.740 | 0.896 / 0.896 | 0.426 | 484 | 0.630 | +0.0002 |
| 0.25 | 0.927 | 0.725 | 0.848 / 0.848 | 0.368 | 484 | 0.634 | -0.0251 |
| 0.30 | 0.959 | 0.722 | 0.841 / 0.841 | 0.349 | 484 | 0.633 | -0.0386 |
| 0.35 | 0.957 | 0.681 | 0.704 / 0.704 | 0.231 | 484 | 0.597 | -0.0294 |

## Detector-versus-tracker decomposition

An offline audit of the same 3-scene sanitized detector cache gives the
following candidate-level upper bound (3 m center matching; 4 m sensitivity
changes recall by less than 0.001):

| detector threshold | candidate precision | candidate recall | local-track recall |
|---:|---:|---:|---:|
| 0.20 | 0.941 | 0.765 | 0.740 |
| 0.25 | 0.971 | 0.744 | 0.722 |
| 0.30 | 0.988 | 0.703 | 0.722 |
| 0.35 | 0.993 | 0.640 | 0.681 |

At the lowest available threshold, candidate recall is already below 0.80 and
local tracking retains most of that candidate recall.  This makes a
tracker-only fix unlikely to satisfy the current support gate with the same
detector cache.

## Decision

The previous table used an event-conditioned association denominator. With the
independent raw-GT opportunity denominator, threshold 0.20 has end-to-end recall
0.426 (conditional recall 0.896), while local track recall remains 0.740 and
track precision falls to 0.843. No tested operating point meets both tracking
precision and recall at 0.80. The formal 0.25 replay is therefore not failing
because of an obviously bad threshold alone; the current recall bottleneck is
detector/track coverage and receiver-side visibility.

The EWMA baseline is effectively neutral at threshold 0.20 (+0.0002 m) and
negative at the other points.  Metadata-only is strongly harmful, and recent
error is negative.  The mean-probe aggregate has non-zero harm and is not a
CMP comparison.

## Go/no-go

Do **not** implement the learned receiver-side reliability controller yet.
Keep the GT-free contract and ledger as the reusable data layer.  A next
method phase requires either a better detector/tracker operating point or a
clearly scoped robustness/association benchmark.  GPU is not needed for this
decision; it becomes relevant only for raw CMP/MTR inference or training after
the data-support gate is resolved.

Artifacts:

- `data/intermediate/gtfree_cpu/opv2v_3scene/failure_analysis.md`
- `data/intermediate/gtfree_cpu/probe40_async_score020/offline_evaluation.json`
- `data/intermediate/gtfree_cpu/probe40_async_score025/offline_evaluation.json`
- `data/intermediate/gtfree_cpu/probe40_async_score030/offline_evaluation.json`
- `data/intermediate/gtfree_cpu/probe40_async_score035/offline_evaluation.json`
- `data/intermediate/gtfree_cpu/opv2v_3scene/detector_candidate_audit.json`
- `data/intermediate/gtfree_cpu/opv2v_3scene/detector_candidate_audit_gate4.json`
