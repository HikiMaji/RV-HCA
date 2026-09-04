# CMP OPV2V source audit for GT-free export

审计时间：2026-09-02  
源码来源：CMP `main`，commit `de0f83900abaff47d87bb7b8329a5ff4a369a49d`（按 `git ls-remote` 记录）。当前 `vendor/CMP/` 是为本审计按 raw 文件取得的最小源码子集，不宣称完整仓库可直接运行。

## 已确认的 GT 注入点

### 1. Detection → tracking 前

`opencood/tools/multi_ego_inference_opv2v.py` 调用 `inference_*` 后，把 `gt_object_id_tensor` 传入 `eval_utils.caluclate_tp_fp`，并保存 `result_stat[0.7]['matched_indices']` 为 `matched_car_id`。随后对 `matched_car_id == -1` 的检测做 mask，只把 GT 匹配的检测送入 AB3DMOT；`dets_frame['cav_id']` 也直接使用该数组。

对应代码：

- [`eval_utils.py`](../vendor/CMP/opencood/utils/eval_utils.py#L43-L108)
- [`multi_ego_inference_opv2v.py`](../vendor/CMP/opencood/tools/multi_ego_inference_opv2v.py#L193-L266)
- [`multi_ego_inference_opv2v.py`](../vendor/CMP/opencood/tools/multi_ego_inference_opv2v.py#L370-L405)

### 2. Tracking 后

脚本读取 AB3DMOT 的 `obj_id`，将 detection item 的最后一列当作 `cav_id`，再按每个轨迹最常见的 `cav_id` 把 local tracker ID 改成 OPV2V 标注 ID。这个转换使最终 tracking pickle 的 key 不是 receiver-local identity。

对应代码：

- [`multi_ego_inference_opv2v.py`](../vendor/CMP/opencood/tools/multi_ego_inference_opv2v.py#L447-L530)

### 3. Prediction dataset 前

`MTR/mtr/datasets/opv2v_multiego_dataset.py` 在读入 GT/pred pickle 后做 key intersection，并用 GT 的长度、有效帧和 future validity 过滤 pred track；`common_timestamps_keys` 也由 GT/pred 的 timestamp intersection 产生。即使 tracking pickle 已改成 local ID，这段代码仍会把 GT actor key 作为 prediction input 过滤器。

对应代码：

- [`opv2v_multiego_dataset.py`](../vendor/CMP/MTR/mtr/datasets/opv2v_multiego_dataset.py#L608-L643)
- [`opv2v_multiego_dataset.py`](../vendor/CMP/MTR/mtr/datasets/opv2v_multiego_dataset.py#L774-L817)

## GT-free 重构边界

不能只把 `matched_car_id` 重命名。第一版需要：

1. 评测统计与 tracking 输入分离：GT box/ID 只送独立 evaluator；在线 detection cache 保存所有通过 detector threshold/NMS 的候选，不做 GT mask。
2. `dets_frame['cav_id']` 改成 source-local detection metadata；AB3DMOT 的 `obj_id` 原样保留为 `(sequence, source)` local track ID。
3. 删除 obj→OPV2V-GT-ID 的 most-common 转换；保存每帧 local track state、pose、timestamp、lifecycle 和 source ID。
4. MTR input loader 只按 prediction/tracking artifact 的 local keys 和 timestamp 构造输入；GT future/GT key 只由离线 evaluator 加载。
5. 在 receiver 侧将对齐后的 peer track/forecast 与 receiver local tracks 做一对一关联，创建 `receiver_target_id`；这个 ID 才能写入 ledger。

## 复核约束

- `--no_coop` 也必须走同一 GT-free tracking exporter；不能把现有 no-coop cache 当作 ego-only 反事实。
- pose/delay/丢包等感知层扰动需要重新生成 detection/tracking artifact；不能仅在旧 prediction cache 上改标签。
- 预测 aggregation 模型当前通过 `center_objects_id` 字典聚合各 CAV 输出；使用 receiver-local IDs 后必须另做 receiver association/remapping，不能让 source-local ID 直接跨车碰撞。
- 本阶段不修改 aggregation 网络，只先让输入和 ledger 的 provenance 合法可审计。
