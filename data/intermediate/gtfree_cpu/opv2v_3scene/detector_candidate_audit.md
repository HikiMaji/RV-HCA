# Detector-candidate coverage audit

This is an offline raw-label audit of the sanitized detector cache. It estimates how much recall is available before AB3DMOT and receiver-side association. Labels are not used by online replay.

## Overall detector operating points

| threshold | candidates | candidate precision | candidate recall |
|---:|---:|---:|---:|
| 0.20 | 13613 | 0.941 | 0.765 |
| 0.25 | 12842 | 0.971 | 0.744 |
| 0.30 | 11924 | 0.988 | 0.703 |
| 0.35 | 10803 | 0.993 | 0.640 |

## Interpretation

Compare candidate recall with the formal local-track recall (0.722). If candidate recall is already low, a tracker-only change cannot reach the support gate; if candidate recall is high but track recall is low, AB3DMOT/score filtering is the immediate CPU-side target. Candidate precision is not trajectory-prediction quality.
