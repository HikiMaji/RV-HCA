# Detector cache comparison

All values are candidate-level offline matches on the 3-scene formal replay allowlist. Raw labels are not used online.

| cache | threshold | candidates | candidate precision | candidate recall |
|---|---:|---:|---:|---:|
| opv2v_corpbevtlidar_delay_1_frame_aug_c256_test_gtfree | 0.20 | 17629 | 0.876 | 0.922 |
| opv2v_corpbevtlidar_delay_1_frame_aug_c256_test_gtfree | 0.25 | 17276 | 0.887 | 0.915 |
| opv2v_corpbevtlidar_delay_1_frame_aug_c256_test_gtfree | 0.30 | 16934 | 0.891 | 0.900 |
| opv2v_corpbevtlidar_delay_1_frame_aug_c256_test_gtfree | 0.35 | 16397 | 0.891 | 0.872 |
| opv2v_corpbevtlidar_delay_1_frame_aug_test_gtfree | 0.20 | 17836 | 0.874 | 0.930 |
| opv2v_corpbevtlidar_delay_1_frame_aug_test_gtfree | 0.25 | 17550 | 0.884 | 0.926 |
| opv2v_corpbevtlidar_delay_1_frame_aug_test_gtfree | 0.30 | 17224 | 0.892 | 0.916 |
| opv2v_corpbevtlidar_delay_1_frame_aug_test_gtfree | 0.35 | 16801 | 0.898 | 0.900 |
| opv2v_point_pillar_sinbevt_test_gtfree | 0.20 | 13613 | 0.941 | 0.765 |
| opv2v_point_pillar_sinbevt_test_gtfree | 0.25 | 12842 | 0.971 | 0.744 |
| opv2v_point_pillar_sinbevt_test_gtfree | 0.30 | 11924 | 0.988 | 0.703 |
| opv2v_point_pillar_sinbevt_test_gtfree | 0.35 | 10803 | 0.993 | 0.640 |
| opv2v_point_pillar_v2vnet_multiego_test_gtfree | 0.20 | 16799 | 0.919 | 0.921 |
| opv2v_point_pillar_v2vnet_multiego_test_gtfree | 0.25 | 15945 | 0.953 | 0.907 |
| opv2v_point_pillar_v2vnet_multiego_test_gtfree | 0.30 | 15250 | 0.973 | 0.885 |
| opv2v_point_pillar_v2vnet_multiego_test_gtfree | 0.35 | 14354 | 0.984 | 0.842 |

The cache with the highest candidate recall at the lowest threshold is the only sensible source for the next GT-free replay. Candidate metrics are an offline upper-bound diagnostic, not trajectory metrics. The CoBEVT and V2VNet rows are cooperative detector outputs (not independent single-source sensors), so their higher recall must not be presented as a source-independent detector comparison.
