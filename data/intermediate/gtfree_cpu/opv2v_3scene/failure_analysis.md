# CPU GT-free failure decomposition

This report scores the GT-free artifacts offline with raw OPV2V labels. Labels are not consumed by replay, tracking IDs, cross-source association, or ledger construction.

## Main finding

The dominant current blocker is recall, not false-positive precision. The replay keeps highly precise local tracks, but detector/track misses and receiver visibility gaps remove many candidate targets before a common event can be verified.

| scope | track rows | track precision | track recall | ID switches / 100 matched | common events | event-conditioned precision | event-conditioned candidate recall | shared-only receiver-track hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 12260 | 0.987 | 0.722 | 1.620 | 7104 | 0.990 | 0.638 | 0.337 |

The table above is an event-conditioned failure decomposition. The independent raw-GT opportunity metrics used for the main contract are:

| metric | value | denominator |
|---|---:|---:|
| end-to-end association recall | 0.336 | 20768 raw-GT opportunities |
| receiver raw-GT visibility rate | 0.825 | 20768 / 25170 peer opportunities |
| recall over all peer opportunities | 0.277 | 25170 raw peer opportunities |
| conditional association recall | 0.990 | 7039 both-side track opportunities |
| association precision (same evaluated events) | 0.990 | 7039 evaluated events |

End-to-end opportunities include peer/receiver detector or track misses; conditional recall intentionally conditions those failures out.

## Tracking by scene/source

| scene/source | rows | precision | recall | ID switches | persistence median | missed GT |
|---|---:|---:|---:|---:|---:|---:|
| 2021_08_18_19_48_05/1045 | 1867 | 0.993 | 0.591 | 82 | 1.000 | 1284 |
| 2021_08_18_19_48_05/1054 | 2186 | 0.995 | 0.728 | 12 | 1.000 | 812 |
| 2021_08_20_21_10_24/1996 | 1782 | 0.992 | 0.682 | 51 | 1.000 | 823 |
| 2021_08_20_21_10_24/2005 | 2510 | 0.998 | 0.813 | 12 | 1.000 | 577 |
| 2021_08_20_21_10_24/2014 | 2195 | 0.992 | 0.796 | 29 | 1.000 | 559 |
| 2021_08_22_09_08_29/5933 | 953 | 0.925 | 0.778 | 4 | 1.000 | 252 |
| 2021_08_22_09_08_29/5942 | 767 | 0.961 | 0.676 | 6 | 1.000 | 354 |

## Association failure categories

| scene/receiver->source | common | shared-only | peer unmatched | receiver label absent | receiver track miss | cross-source gate miss | wrong receiver target | correct common |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2021_08_18_19_48_05/1045->1054 | 482 | 1704 | 10 | 524 | 965 | 210 | 4 | 473 |
| 2021_08_18_19_48_05/1054->1045 | 489 | 1378 | 14 | 585 | 581 | 203 | 4 | 480 |
| 2021_08_20_21_10_24/1996->2005 | 1374 | 1136 | 4 | 400 | 500 | 242 | 0 | 1364 |
| 2021_08_20_21_10_24/1996->2014 | 851 | 1344 | 17 | 466 | 427 | 446 | 2 | 837 |
| 2021_08_20_21_10_24/2005->1996 | 1428 | 354 | 15 | 39 | 121 | 186 | 1 | 1420 |
| 2021_08_20_21_10_24/2005->2014 | 613 | 1582 | 17 | 150 | 415 | 1003 | 0 | 610 |
| 2021_08_20_21_10_24/2014->1996 | 878 | 904 | 15 | 199 | 283 | 420 | 1 | 864 |
| 2021_08_20_21_10_24/2014->2005 | 623 | 1887 | 4 | 433 | 461 | 994 | 0 | 618 |
| 2021_08_22_09_08_29/5933->5942 | 178 | 589 | 30 | 283 | 186 | 111 | 8 | 149 |
| 2021_08_22_09_08_29/5942->5933 | 188 | 765 | 71 | 333 | 277 | 107 | 8 | 157 |

## Ledger hindsight coverage

| horizon (s) | rows | valid | coverage | censor: not observed/track dead |
|---:|---:|---:|---:|---:|
| 0.100 | 7104 | 7104 | 1.000 | 0 |
| 0.300 | 7104 | 6688 | 0.941 | 416 |
| 0.500 | 7104 | 6339 | 0.892 | 765 |
| 1.000 | 7104 | 5582 | 0.786 | 1522 |
| 2.000 | 7104 | 4460 | 0.628 | 2644 |
| 3.000 | 7104 | 3556 | 0.501 | 3548 |
| 5.000 | 7104 | 2146 | 0.302 | 4958 |

## Track survival

| horizon | matched start rows | same local ID | same-GT survival |
|---:|---:|---:|---:|
| 1.0s | 11240 | 0.850 | 0.847 |
| 2.0s | 10276 | 0.760 | 0.757 |
| 3.0s | 9329 | 0.685 | 0.683 |
| 5.0s | 7412 | 0.585 | 0.582 |

## Interpretation and next gate

The 0.1 s arrival delay is fully exposed in the ledger; no row is silently evaluated before arrival. Coverage falls with horizon because the receiver target track is not observed or has died, reaching the lowest level at 5 s. Recent-error and EWMA selectors therefore have usable feedback but currently produce negative mean gain versus ego on the formal artifact.

Under the current contract gate (track and end-to-end association recall at least 0.80, precision at least 0.80, and common hindsight coverage at least 0.50), precision and persistence pass but track recall and end-to-end association recall do not. Do not start a learned reliability controller yet; first test whether a lower detector threshold or a better receiver-side gate can recover recall without collapsing precision. The present trajectory aggregate is a mean probe, not CMP, so no CMP-vs-GT-free ADE/FDE claim is made.
