# RV-HCA 当前结果：外部审阅摘要

更新时间：2026-09-04

## 研究范围

当前只完成 RV-HCA 的 **GT-free data contract**，没有实现完整 reliability/regret controller，也没有完成 CMP 的真实 aggregate inference。

核心问题是：

> Can a receiver use realized errors of past peer forecasts to identify unreliable collaboration and reduce harmful cooperative prediction without ground-truth information at inference time?

## GT-free pipeline

在线 replay 的数据流为：

```text
sanitized detector outputs
    -> source-local AB3DMOT IDs
    -> world/receiver coordinate alignment
    -> arrival-time receiver-side cross-source association
    -> prediction ledger
    -> later receiver observation fills realized error
```

GT 不参与 tracking ID、receiver target ID、cross-source matching 或 ledger 构造。GT 只在离线 evaluator 中用于评分 tracking、association 和 trajectory metrics。

主任务只保留 common targets；shared-only targets 只统计覆盖率，没有设计完整处理方法。

## Formal CPU artifact

数据为 OPV2V test 的 3 个 scene（2、2、3 个 CAV），使用 sanitized detector cache 和 0.25 score threshold。正式 artifact 包含：

- 12,260 local track rows
- 18,747 arrival-time association events
- 49,728 common-target ledger rows
- 35,875 valid matured hindsight rows
- 总体 ledger hindsight coverage：0.721
- 不读取 raw labels 的 contract audit：PASS（tracks 12,260、association 18,747、ledger 49,728、full-history MTR manifest 166,149；无禁止字段、无重复 ledger key、时间顺序通过）

### Tracking

- precision：0.987
- recall：0.722
- ID switches：196
- ID switches / 100 matched rows：1.620
- persistence median：1.000

### Association

- conditional association precision / recall：0.990 / 0.990
- end-to-end association recall：0.336（6,972 correct / 20,768 raw-GT opportunities）
- conditional association opportunities：7,039
- raw peer opportunities：25,170；receiver visibility rate：0.825（20,768 / 25,170）；若把 receiver 不可见目标也计入分母，补充 recall 为 0.277
- correct common events：6,972
- shared-only receiver-GT availability：0.696（8,101 / 11,643）；其中 receiver local-track hit rate 为 0.337

离线失败分解：

- peer track unmatched：197
- receiver label absent / not visible：3,412
- receiver track missed：4,216
- cross-source gate miss：3,922
- wrong receiver target：28
- correct common：6,972

因此当前主要问题是 recall/visibility，而不是 false-positive precision。

### Ledger coverage by horizon

| horizon | coverage |
|---:|---:|
| 0.1 s | 1.000 |
| 0.3 s | 0.941 |
| 0.5 s | 0.892 |
| 1.0 s | 0.786 |
| 2.0 s | 0.628 |
| 3.0 s | 0.501 |
| 5.0 s | 0.302 |

到达延迟为 0.1 s，ledger 没有在 packet arrival 之前提前使用结果。

## 简单 hindsight baselines

在 formal artifact 上，mean selected error 相对 ego 的 gain 为：

- recent-error：-0.048 m
- EWMA：-0.012 m
- metadata-only：-0.227 m

负值表示比 ego-only 更差；因此目前没有简单 baseline 能稳定降低误差。

## Operating-point scan

在同一 scene 的 40-frame asynchronous probe 上扫描 detector threshold：

| threshold | track P/R | association precision / conditional recall | end-to-end association recall | ledger coverage |
|---:|---:|---:|---:|---:|
| 0.20 | 0.843 / 0.740 | 0.896 / 0.896 | 0.426 | 0.630 |
| 0.25 | 0.927 / 0.725 | 0.848 / 0.848 | 0.368 | 0.634 |
| 0.30 | 0.959 / 0.722 | 0.841 / 0.841 | 0.349 | 0.633 |
| 0.35 | 0.957 / 0.681 | 0.704 / 0.704 | 0.231 | 0.597 |

0.20 的 conditional recall 为 0.896，但独立 raw-GT opportunity denominator 下的
end-to-end recall 只有 0.426，tracking recall 仍为 0.740。没有 tested threshold
同时满足 tracking precision 和 recall ≥ 0.80。

## Detector-versus-tracker decomposition

对同一 3-scene sanitized detector cache 做独立 candidate-level 离线匹配
（3 m center gate；4 m sensitivity 的 recall 变化小于 0.001）：

| threshold | detector candidate P/R | local-track recall |
|---:|---:|---:|
| 0.20 | 0.941 / 0.765 | 0.740 |
| 0.25 | 0.971 / 0.744 | 0.722 |
| 0.30 | 0.988 / 0.703 | 0.722 |
| 0.35 | 0.993 / 0.640 | 0.681 |

因此当前 detector cache 在最低可用阈值下的 candidate recall 已低于 0.80，
而 AB3DMOT 只进一步损失约 2--3 个百分点。仅修改 tracker 很难使项目
通过当前 support gate；若坚持 RV-HCA，应先获得更高 recall 的 detector 或
明确把论文问题改成 association/robustness benchmark。

### 高召回官方 detector 的方向性复核

官方 CoBEVT-c256 raw detector cache 可以在不读取 GT 的 sanitizer 后进入同一
pipeline。在正式 3-scene allowlist 的 candidate-level offline audit 中，score
0.25 的 candidate precision/recall 为 `0.887/0.915`，明显高于 PointPillar
`0.971/0.744`。这确认 sanitized cache 的 recall 上限主要来自 detector 输出
选择，而不是 tracker；不过 40-frame CPU probe 上 c256 的 track
precision/recall 只有 `0.783/0.732`，说明高召回同时带来 false positives，不能
把 detector candidate recall 直接当作 end-to-end recall。

同一 40-frame probe 的 c256 threshold scan 也不改变判断：score 0.20 的
track P/R=`0.777/0.735`、end-to-end/conditional association=`0.459/1.000`；
score 0.30 为 `0.782/0.731`、`0.450/1.000`。主要代价是 false positives，
而不是 AB3DMOT 单独造成的 recall 损失。

完整 c256 scene 的 4,759 个 receiver-visible opportunities 对应
end-to-end/conditional=`0.498/0.999`；全部 6,125 个 raw peer opportunities
的 receiver visibility rate 为 `0.777`，补充 all-peer recall 为 `0.387`。

## 轨迹数值的解释边界

当前 ledger 中的 forecast 是 constant-velocity probe，aggregate 是 `mean_probe_not_cmp`，不是 CMP 输出。正式 artifact 的离线 trajectory metrics 为：

- ego-only ADE/FDE：2.803
- peer ADE/FDE：2.974
- mean-probe aggregate ADE/FDE：2.633
- mean-probe GT harm rate（aggregate error > ego error + 0.1 m）：0.282

这些数值只能说明数据契约和评测链路可运行，不能作为 CMP 或 RV-HCA 方法增益。

新版 evaluator 将 association 分成两种口径：end-to-end recall 是正确
common association 除以到达时 receiver raw GT 中确实存在对应 peer object 的
机会；conditional recall 是在 peer 与 receiver 两侧 local track 都有离线 GT
match 的机会中计算。PointPillar formal replay 的两者分别为 `0.336`（20,768
机会）和 `0.990`（7,039 机会）。同一 replay 的 exact-frame track survival
（same local ID 且仍匹配同一 GT）@1/2/3/5s 为 `0.847/0.757/0.683/0.582`，
故旧的单一 association recall `0.638` 不能再作为唯一数字。

## 当前结论

1. GT-free tracking、receiver association、prediction ledger 和离线 realized-error 回填已经实现并通过 schema/禁止字段检查。
2. ledger 的短 horizon coverage 足以做 hindsight 分析，但长 horizon coverage 快速下降。
3. 正式 0.25 replay 的 track recall 0.722；association end-to-end/conditional recall 为 0.336/0.990，未达到进入 learned reliability aggregation 的支持门槛。
4. 简单 threshold 调节不能解决 tracking recall；当前 bottleneck 是 detector/track coverage、receiver visibility 和 cross-source gate。
5. 目前不应实现 attention/concordance/censoring/regret controller，也不应声称 do-no-harm 或 CMP 改进。

真实 MTR 接入尚未发生：本机没有 GPU，MTR checkpoint 仅有未完成 `.part` 文件，
官方 attention 扩展与当前 Python/glibc 不兼容；native MTR output 还含
`object_id`/`gt_trajs` 且没有 source-local ID。已加入严格的
`rvhca_cpu/mtr_io.py` + `scripts/ingest_mtr_predictions.py` derived-ledger
入口，缺少 GT-free wrapper 写出的 `source_track_id` 或检测到 GT-derived key
时直接 fail closed。当前 predictiveness script 对现有 CV ledger 的状态是
`BLOCKED_NO_REAL_MTR`，不是用 probe 伪造 NO-GO。

## 工程与资源边界

当前环境为 1 CPU、2 GiB memory、无 GPU。CPU 工作已足够完成上述 contract 和 evaluator。CMP 原始 detector/MTR inference、MTR training 和 reliability model training 才需要 GPU；但目前 MTR 权重不完整，CUDA attention extension 也未构建，因此开 GPU 不能直接解决全部问题。

## 请外部 AI 重点审阅

请判断：

1. 以上 GT-free association recall 的定义和 denominator 是否科学、是否混合了 visibility 与 detector miss；
2. persistence median=1 是否可能因为当前 track 输出/统计定义而过于乐观；
3. 40-frame threshold scan 是否足以支持“简单 threshold 不能解决 recall”的暂定结论；
4. 在当前 recall gate 未通过且 simple baselines 无增益时，继续做 reliability controller 是否值得，还是应 pivot 为 association/robustness benchmark；
5. 若继续 RV-HCA，最小且不引入 GT 的下一步应是改 detector/track、改 receiver gate，还是增加跨 scene/source 的 held-out 验证。

## 复核文件

- formal failure decomposition：`data/intermediate/gtfree_cpu/opv2v_3scene/failure_analysis.md`
- operating-point report：`docs/cpu_operating_point.md`
- formal offline metrics：`data/intermediate/gtfree_cpu/opv2v_3scene/offline_evaluation.json`
- detector candidate audit：`data/intermediate/gtfree_cpu/opv2v_3scene/detector_candidate_audit.md`
