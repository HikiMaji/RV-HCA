# CPU 阶段回放记录

更新时间：2026-09-04。这里实现的是 GT-free 数据契约，不是 RV-HCA reliability controller。

## 可运行入口

```bash
cd /root/autodl-tmp/RV-HCA
PYTHONPATH=$PWD /root/autodl-tmp/RV-HCA/.venv/bin/python \
  scripts/run_gtfree_cpu.py \
  --scene 2021_08_22_09_08_29 \
  --scene 2021_08_18_19_48_05 \
  --scene 2021_08_20_21_10_24 \
  --output-dir data/intermediate/gtfree_cpu/opv2v_3scene \
  --aggregate-mode mean_probe
```

`mean_probe` 只是把 ego 和 peer 的 constant-velocity probe 取均值，用来打通 aggregate/error/Harm Rate 字段；它不是 CMP aggregate，不能作为论文中的 CMP 结果。

真实 CMP/MTR 接入入口是 [`scripts/ingest_mtr_predictions.py`](../scripts/ingest_mtr_predictions.py)。它对 base ledger 写 derived ledger，不回写原始行；输入必须是由 GT-free MTR wrapper 输出的 normalized multimodal records，不能直接把官方带 `object_id`/`gt_trajs` 的 native evaluator pickle 当成预测输入。

独立 GT 评测必须单独调用：

```bash
PYTHONPATH=$PWD /root/autodl-tmp/RV-HCA/.venv/bin/python \
  scripts/evaluate_gtfree.py data/intermediate/gtfree_cpu/opv2v_3scene
```

## 输出

- `local_tracks.jsonl`：source-local AB3DMOT ID、测量 pose、world/ local center、速度；不含数据集 actor/GT ID。
- `association_events.jsonl`：packet 到达时的 receiver-side Hungarian 几何关联、gate、confidence、`common`/`shared_only`，并保存 send/arrival frame。
- `prediction_ledger.jsonl`：核心键 `(receiver, source, receiver_target_id, send_time, horizon, forecast)`，并附带 arrival、receiver-only observation、realized error 和删失原因。
- `baseline_summary.json`：recent-error、EWMA、metadata-only 的因果 prefix 选择汇总。
- `offline_evaluation.json`：唯一读取 raw YAML `vehicles` 的独立 evaluator 输出。
- `contract_audit.json`：不读取 raw labels 的字段、时间顺序和 ledger 唯一性审计。

## 已下载 3 场景结果

配置：score threshold 0.25、association/target gate 4 m、frame period 0.1 s、peer delay 0.1 s。

| 指标 | 数值 |
|---|---:|
| track rows | 12,260 |
| common association events | 7,104 |
| shared-only events | 11,643 |
| ledger rows | 49,728 |
| hindsight coverage | 0.721 |
| 0.1 s / 1 s / 5 s coverage | 1.000 / 0.786 / 0.302 |
| offline track precision / recall | 0.987 / 0.722 |
| conditional association precision / recall | 0.990 / 0.990 |
| end-to-end association recall | 0.336 (20,768 opportunities) |
| ID-switches / 100 matched rows | 1.62 |

association 的分母必须分开报告：end-to-end recall = 正确 common association / 到达时 raw receiver GT 中确实存在对应 peer object 的全部机会；conditional recall = 正确 / peer 与 receiver 两侧 local track 都有独立 GT match 的机会。PointPillar formal replay 的正确数为 `6,972`，两种分母分别为 `20,768` 和 `7,039`，所以两种 recall 为 `0.336 / 0.990`；旧的 event-conditioned `0.638` 不再作为主 recall。另有 `25,170` 个 raw peer opportunities，其中 receiver 可见 `20,768`（visibility rate `0.825`），若把 receiver 不可见目标也计入分母，补充 recall 为 `0.277`。track survival（起始 track 有 GT match、未来 exact frame 保留同一 local ID 且仍匹配同一 GT）为 `@1/2/3/5s = 0.847/0.757/0.683/0.582`；同一 local ID 的 presence 为 `0.850/0.760/0.685/0.585`。

官方 CoBEVT-c256 detector 的 40-frame CPU probe（scene `2021_08_22_09_08_29`、score 0.25）显示 candidate recall 约 `0.915` 并没有自动转化为 tracker recall：track precision/recall `0.783/0.732`，end-to-end/conditional association recall `0.450/1.000`。这是高召回 cache 的方向性审计，不替代正式 3-scene replay；同时提示新增 false positives 会先伤 precision，AB3DMOT 不是唯一瓶颈。一个完整 c256 scene 的方向性 replay 为 track P/R `0.946/0.866`、end-to-end/conditional association `0.498/0.999`，其 receiver visibility rate 为 `0.777`、全部 peer-opportunity recall 为 `0.387`，不能外推为 3-scene 结论。

同一 probe 的 c256 threshold scan：score `0.20` 的 track P/R=`0.777/0.735`、end-to-end/conditional association=`0.459/1.000`；score `0.30` 为 `0.782/0.731`、`0.450/1.000`。降低阈值主要增加 false positives，未把 tracker recall 推到 0.80；旧的 `0.63` 左右数字是事件条件化口径。

在 `mean_probe` 下 receiver-observation 误差为 ego 1.99 m、peer 2.42 m；GT future evaluator 的 probe aggregate harm rate 为 0.282（`eps=0.1 m`）。这些数值只用于暴露数据契约和负协作现象，不能替代 CMP/MTR 的原生预测结果。

正式 artifact 的 `cmp_aggregate_forecast`/`cmp_aggregate_error` 保持为空；当前没有把 mean probe 冒充 CMP aggregate。等 GPU/MTR 资产可用后，CMP output 只能作为一个新的输入模型写入，不改写已有 ledger 行。当前 `analyze_mtr_predictiveness.py` 对正式 ledger 的状态仍是 `BLOCKED_NO_REAL_MTR`，因此不能据此宣称历史误差无预测性或通过 NO-GO。

三个因果 prefix baseline 相对 ego 的平均 gain 均为负：recent-error -0.048 m、EWMA -0.012 m、metadata-only -0.227 m。当前结果不能支持“简单 hindsight 已经能降低 harm”的结论。

## 当前阻塞

track recall 0.722、end-to-end association recall 0.336（conditional recall 0.990），尚未达到契约中进入 reliability aggregation 的支持性门槛。40 帧 threshold probe 的新分母下 end-to-end recall 仅为 0.231--0.426，不能把条件化的 0.70--0.90 数字当成端到端覆盖；这应作为预注册 operating-point 扫描，而不是用测试集反调结果。

因此下一阶段先分析检测漏检、shared-only 覆盖和跨源匹配失败的来源；当前不启动 attention、concordance、censoring 或 learned regret controller。
