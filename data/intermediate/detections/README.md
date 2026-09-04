# GT-free detector artifact

`opv2v_point_pillar_sinbevt_test_gtfree.pkl` is a sanitized copy of the official CMP `point_pillar_sinbevt` no-cooperation test detection cache.

The source cache also stored `matched_car_id` and a fused-feature path. Those fields were removed. The artifact keeps only detector boxes/scores, box-derived velocity, recorded lidar pose, and timestamp metadata. A strict recursive key check found no GT/object-ID/matching fields in the serialized object.

This is a detector-output input for the GT-free replay, not a GT-free tracking result. The replay must still assign fresh source-local AB3DMOT IDs and receiver-local target IDs. The recorded pose is treated as a pose measurement and must pass through the configured pose-noise/alignment step; it must not be replaced by GT pose or future state.

The cache covers the official OPV2V test scenes, while raw PCD/YAML files currently exist locally for only three scenes. Do not use CMP's precomputed tracking/prediction caches as online inputs: they contain GT-assisted filtering or GT-key conversion.
