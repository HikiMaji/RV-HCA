# CMP/MTR prediction integration status

更新时间：2026-09-04。本文记录官方 MTR 的 GT-free 接入与代码审计；不包含 reliability controller。

> **结果冻结提示**：本地已修复 MTR world-history、fixed-world tracking、XY 主指标、EWMA 和 A/B/C target-level ego 条件。本文下方旧的三场景 MTR 数字来自修复前输入，全部标为 historical/invalid，不能作为当前科学结论；必须通过 `run_gtfree_mtr.py` 的 input-distribution audit 后重跑。

修复核验表见 [`docs/re_9-4_verification.md`](re_9-4_verification.md)。

## 当前事实

1. 4090 模式下已验证 CUDA 11.8、PyTorch 2.0 与 `mtr.ops.knn`/`mtr.ops.attention`（按 Python 3.8/`sm_89` 重编译，CUDA KNN kernel smoke 通过）。三场景本次运行实际使用 RTX 4090，882 组 forward、9583 个中心目标，MTR peer+ego 推理耗时 107.72 s。
2. 三套 MTR 预测 checkpoint 已完整下载：no-coop 1,863,607,503 bytes、no-aggregation 1,863,607,503 bytes、cooperative aggregate 4,253,524,373 bytes；ZIP CRC 检查均通过，三者均可在 4090 上 `torch.load(map_location='cuda:0')`。
3. 官方 `OPV2VMultiEgoDataset.generate_prediction_dicts` 保存 `object_id`、`gt_trajs`、`track_index_to_predict` 等 native evaluator 字段，但没有可靠的 source-local tracker ID；`ego_cav_id` 是场景 receiver 字段，不能自动当作 prediction source。官方 tracking/prediction trajectory cache 也不能直接作为 GT-free 输入。
4. 三个现有 scene 的 additional lane 已从官方 `additional-001.zip` 按 HTTP Range 选择性提取，共 952 张（18/20/22 三场景分别 356/294/302）；没有用零图或跨场景图补齐。可复现脚本为 [`scripts/extract_box_zip_ranges.py`](../scripts/extract_box_zip_ranges.py)。

## 合法接入路径

真正可用于本项目的 prediction record 必须由 GT-free MTR input wrapper 在模型 forward 时同时写出。当前已经从 formal local-track replay 构造出完整 10 帧历史 manifest（166,149 条记录）；它只用于外部可兼容 GPU 的 wrapper，不是模型预测结果：

```text
sequence_id, source, source_track_id, send_frame_idx, send_time,
forecast_frame, pred_trajs[M,T,2/3], pred_scores[M], time_offsets_s[T], model
```

输入 wrapper 可以使用 GT 作为独立 trajectory metric 的 future label，但不能用 GT 给 track ID、跨车 target matching 或 ledger key。模型输出通过 [`rvhca_cpu/mtr_io.py`](../rvhca_cpu/mtr_io.py) 以 derived ledger 方式附加；`scripts/ingest_mtr_predictions.py` 会在以下任一情况直接失败：

- record 含 `object_id`、`gt_trajs`、`matched_car_id`、`track_index_to_predict` 等 GT-derived key；
- peer record 没有 `source_track_id`；
- aggregate record 没有 receiver-side `receiver_target_id`；
- 只凭官方输出数组序号猜 local ID。

多模态 forecast 保留完整 `trajectories_xyz[M,T,3]` 与 `scores[M]`，同时为每个 ledger horizon 写 top-score mode 的 `position_xyz`，因此现有 evaluator 可以先做 committed/top-score error，后续再扩展 native minADE6/minFDE6。

为适配三场景文件体量，`ingest_mtr_predictions.py`、
`recompute_mtr_ledger_errors.py` 与 `evaluate_gtfree.py` 均采用 ledger 流式回放，
不在内存中保留全部 multimodal 数组。`analyze_mtr_predictiveness.py` 也只读取
每行的模型标签、因果时间字段和标量误差；输出中的
`heldout_scene_evaluation` 是显式 leave-one-scene-out 诊断，阈值只从其他 scene
的历史误差中取中位数，不训练 controller。

## 修复前的 4090 GT-free MTR 运行（historical，仅供审计）

运行器为 [`scripts/run_gtfree_mtr.py`](../scripts/run_gtfree_mtr.py)，先用
`cmp_mtr_no_agg` 生成 peer 预测，再换入 `cmp_mtr_no_coop` 生成 ego 预测；
每条记录均为 GT-free track、`uses_gt=false`，不含 native GT 字段。
三场景 send frame 从 10 开始，共 882 个 source/frame group、9,583 个中心目标，
导出 19,166 条真实多模态记录（peer/ego 各 9,583）。单场景 scene-22 的 2,082
条记录仍保留作为 smoke 基线；接入器曾发现并已修正 peer/ego 同键的角色过滤问题。

这些数字来自修复前的 historical run。修复后 runner 会先审计 `run_summary.json`
中的 detector provenance 与 YAML 的 `preprocessed_pred_traj_dir`；PointPillar
history 不再允许直接送入 CoBEVT-c256 peer checkpoint。`--role peer` 和
`--role ego` 可分别在各自 native history 上做 smoke/export；`--role both`
只有在一个明确兼容的 paired history 配置下才可用于科学比较，不能用
`--allow-input-mismatch` 的结果写论文。

产物：

- [`mtr_predictions_gtfree_scene22.jsonl`](../data/intermediate/gtfree_cpu/opv2v_3scene/mtr_predictions_gtfree_scene22.jsonl)
- [`mtr_attachment_summary.json`](../data/intermediate/gtfree_cpu/opv2v_3scene_mtr_scene22/mtr_attachment_summary.json)
- [`mtr_error_summary.json`](../data/intermediate/gtfree_cpu/opv2v_3scene_mtr_scene22_eval/mtr_error_summary.json)
- [`mtr_predictiveness.json`](../data/intermediate/gtfree_cpu/opv2v_3scene_mtr_scene22_eval/mtr_predictiveness.json)
- [`offline_evaluation.json`](../data/intermediate/gtfree_cpu/opv2v_3scene_mtr_scene22_eval/offline_evaluation.json)

三场景产物：

- [`mtr_predictions_gtfree_3scene.jsonl`](../data/intermediate/gtfree_cpu/opv2v_3scene/mtr_predictions_gtfree_3scene.jsonl)
- [`mtr_attachment_summary.json`](../data/intermediate/gtfree_cpu/opv2v_3scene_mtr_3scene/mtr_attachment_summary.json)
- [`mtr_error_summary.json`](../data/intermediate/gtfree_cpu/opv2v_3scene_mtr_3scene_eval/mtr_error_summary.json)
- [`mtr_predictiveness.json`](../data/intermediate/gtfree_cpu/opv2v_3scene_mtr_3scene_eval/mtr_predictiveness.json)
- [`offline_evaluation.json`](../data/intermediate/gtfree_cpu/opv2v_3scene_mtr_3scene_eval/offline_evaluation.json)
- [`contract_audit.json`](../data/intermediate/gtfree_cpu/opv2v_3scene_mtr_3scene_eval/contract_audit.json)
- [`mtr_coordinate_path_audit.json`](../data/intermediate/gtfree_cpu/opv2v_3scene_mtr_3scene_eval/mtr_coordinate_path_audit.json)

三场景 receiver-observation（GT-free）误差：

| scene | valid common MTR rows | peer mean (m) | ego mean (m) | peer−ego (m) | Harm Rate (+0.1 m) |
|---|---:|---:|---:|---:|---:|
| 2021-08-18-19-48-05 | 3,058 | 13.964 | 13.696 | +0.269 | 46.83% |
| 2021-08-20-21-10-24 | 19,559 | 10.961 | 10.631 | +0.330 | 46.61% |
| 2021-08-22-09-08-29 | 1,001 | 9.968 | 11.286 | −1.318 | 47.95% |
| **合计** | **23,618** | **11.308** | **11.055** | **+0.252** | **46.70%** |

合计按 horizon 的 peer−ego 差值为 0.1/0.3/0.5/1/2/3/5 s：
`+0.050/+0.049/+0.075/+0.191/+0.363/+0.799/+1.199 m`；对应 Harm Rate
为 `40.88/44.78/46.20/49.08/50.05/52.02/51.56%`。scene-22 的长 horizon
方向与另外两场景相反，说明协作误差具有明显 scene/source 条件性。

三场景共有 49,728 条 common ledger 行；peer/ego MTR 分别附着到 38,101/38,248
行，最终同时具备 receiver observation 且可重算两侧误差的行数为 23,618
（47.49% of common rows）。`shared_only` 仍只做覆盖统计，未进入上述 Harm Rate。

receiver-observation（不读 GT）的 1,001 条双侧有效 common 行上，3D
selected-mode 误差为 peer 9.968 m、ego 11.286 m，peer−ego 平均 −1.318 m；
按 horizon，0.1/0.3 s 分别为 +0.419/+0.193 m，1/2/3/5 s 分别为
−0.533/−1.563/−3.739/−8.424 m。`peer_error > ego_error + 0.1 m` 的行比例
为 47.95%。这只是一个 scene 的观测误差，不是 controller 效果或跨场景结论。

此前的 recent/EWMA diff 描述性分析仍保留在 JSON 中，但不再作为主结论。最终
relative-harm A/B/C LOSO、native minADE6/minFDE6、ego-vs-peer oracle、naive
mean fusion 与短→长 horizon 检验见单独报告
[`docs/rv_hca_core_scientific_check.md`](rv_hca_core_scientific_check.md)。其中
peer history 的独立增量主要出现在短 horizon 的 target-level；长 horizon 和
source-level 的符号跨 scene 不稳定。这些是离线可预测性证据，不是 controller
效果。

## GPU 运行命令

当前环境中的可复现实验命令如下（必须显式覆盖继承来的 `OMP_NUM_THREADS=0`
和 `MKL_NUM_THREADS=0`）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=$PWD/vendor/CMP-upstream/MTR:$PWD/vendor/CMP-upstream \
$PWD/.venv/bin/python scripts/run_gtfree_mtr.py \
  --scene 2021_08_22_09_08_29 --min-frame 10 --max-frame 107 \
  --output data/intermediate/gtfree_cpu/opv2v_3scene/mtr_predictions_gtfree_scene22.jsonl

# 三场景扩展（已执行完成；可复跑）
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=$PWD/vendor/CMP-upstream/MTR:$PWD/vendor/CMP-upstream \
$PWD/.venv/bin/python scripts/run_gtfree_mtr.py \
  --scene 2021_08_18_19_48_05 \
  --scene 2021_08_20_21_10_24 \
  --scene 2021_08_22_09_08_29 \
  --min-frame 10 \
  --output data/intermediate/gtfree_cpu/opv2v_3scene/mtr_predictions_gtfree_3scene.jsonl

# 后续三场景输出沿用同一套接入、误差重算、past→future 与离线评测命令，
# 只需把上面的 *_scene22.jsonl/目录替换为 *_3scene.jsonl/目录。

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD .venv/bin/python scripts/ingest_mtr_predictions.py \
  data/intermediate/gtfree_cpu/opv2v_3scene \
  data/intermediate/gtfree_cpu/opv2v_3scene/mtr_predictions_gtfree_scene22.jsonl \
  --output-dir data/intermediate/gtfree_cpu/opv2v_3scene_mtr_scene22

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD .venv/bin/python scripts/recompute_mtr_ledger_errors.py \
  data/intermediate/gtfree_cpu/opv2v_3scene_mtr_scene22/prediction_ledger.jsonl \
  --output-dir data/intermediate/gtfree_cpu/opv2v_3scene_mtr_scene22_eval

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD .venv/bin/python scripts/analyze_mtr_predictiveness.py \
  data/intermediate/gtfree_cpu/opv2v_3scene_mtr_scene22_eval/prediction_ledger.jsonl

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD .venv/bin/python scripts/evaluate_gtfree.py \
  data/intermediate/gtfree_cpu/opv2v_3scene_mtr_scene22_eval
```

`run_gtfree_mtr.py` 从 source-local `source_track_id` 生成记录；不能从官方
evaluator 的数组下标或 `object_id` 反推。每个 record 还必须通过
`rvhca_cpu.mtr_io.normalize_prediction_record` 的禁止字段检查。

## 当前结论

真实 CMP/MTR per-source multimodal prediction 已在 4090 上完成三场景 GT-free
export、ledger 接入、receiver-observation error、relative-harm A/B/C LOSO 与
短→长检验。GT-free export/ledger 路径的 contract audit 为 PASS 且未读取 raw
labels；raw-label 读取仅发生在独立的最终 offline evaluator。总体 receiver-
observation peer 比 ego 高 0.252 m，relative Harm Rate 为 46.70%；offline
top-score overall 为 peer/ego 13.533/13.473 m，native minADE6 为 11.682/11.003 m。
没有 cooperative aggregate 的 fused-BEV 输入，因此不报告 CMP aggregate gain，
也不进入 reliability controller 训练；本阶段结果与定义集中在
[`docs/rv_hca_core_scientific_check.md`](rv_hca_core_scientific_check.md)。

同一份报告还包含 global-world 与 receiver-observation 坐标路径审计：23,618
rows 的 3D 误差逐行差异小于 `2.8e-13 m`；XY 的少量差异来自 full 6-DoF
roll/pitch 投影，而非 source/receiver 或 future-observation 时间错位。
