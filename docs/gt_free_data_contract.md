# RV-HCA GT-free data contract

版本：2026-09-02  
状态：**数据契约与 CPU-only GT-free 回放已实现；尚未进入 reliability/regret controller 实现**

本项目先验证一个窄问题：接收方能否在推理时只依赖自己已经收到的预测、自己的无协作观测和运行元数据，建立可回放的 prediction ledger，并在 common targets 上量化 peer forecast 与 CMP aggregation 的实际后果。GT 只存在于离线评测表，不进入 tracking、association、ledger 或任何在线输入。

## 1. 信任边界与流水线

```text
raw detections
    -> source-local AB3DMOT tracks (local_track_id)
    -> pose/time alignment into receiver frame
    -> receiver-side association (Hungarian + motion/geometry gate)
    -> receiver_target_id
    -> prediction ledger
    -> later receiver observation matures a row and appends realized error
```

| 阶段 | 在线允许的输入 | 产物 | GT 状态 |
|---|---|---|---|
| 检测 | raw LiDAR、检测器输出、测量/扰动 pose | detection boxes/scores | 禁止 |
| 跟踪 | 当前检测、历史 tracker state、时间戳 | source-local `local_track_id`、state、covariance、track lifecycle | 禁止 |
| 对齐 | sender/receiver 测量 pose、send/arrival time、延迟模型 | receiver-frame state/forecast | 禁止使用 GT pose |
| 跨源关联 | receiver local tracks、对齐后的 peer state、速度/航向/类别、gate cost | `receiver_target_id`、association confidence、common/shared-only 标记 | 禁止 |
| 预测账本 | 已关联的 peer forecast 和 receiver target ID | immutable forecast row | 禁止 |
| hindsight 回填 | 后续 receiver tracker observation；严格事件顺序 | realized state/error、valid/mask | 禁止；没有观测就保持删失 |
| 离线评测 | 独立保存的 GT，仅用于评测脚本 | association P/R、IDSW、persistence、ADE/FDE、Harm Rate | 允许，仅此处 |

原 CMP 导出的 tracking cache 不能直接复用为 GT-free 输入：当前脚本在 AB3DMOT 前用 GT IoU/`gt_object_id_tensor` 筛选并赋 ID，在 tracker 后又将 AB3DMOT ID 转成 OPV2V 的 GT `cav_id`。此外，MTR 数据集加载阶段还按 GT/pred key intersection 过滤目标。因此第一项实现必须是独立的 GT-free export/replay，而不是给旧 cache 改名。

## 2. 标识符与坐标

### 2.1 标识符

- `source_track_id`：发送方 AB3DMOT 的本地 ID；只在 `(sequence, source)` 命名空间内有效，不能跨车比较。
- `receiver_target_id`：接收方 tracker/association manager 创建的本地 ID；它是 ledger 的唯一目标键，不能由 GT ID、数据集 actor key 或发送方 ID 直接赋值。
- `gt_object_id`：只允许出现在离线评测表中。在线 artifact、预测输入和 ledger schema 不得包含该字段。
- `association_id`：一次 source-to-receiver 匹配的审计记录，可包含 source track ID、receiver track ID、cost、gate、时间戳；不得包含 GT 标签。

### 2.2 对齐和关联

1. 发送方先在自己的坐标系产生 local track state/forecast，并带 `source_track_id`、`send_time`、`arrival_time` 和所用 pose 来源。
2. 接收方用测量/扰动 pose 将 state 与 forecast 变换到自己的坐标系，再按接收时间轴做异步对齐；不得用 GT pose 或 GT future 修正。
3. 对 receiver local tracks 与 peer candidates 计算带门限的几何/运动代价（位置、速度、航向、类别、track age、协方差或固定 gate），用一对一 Hungarian assignment。门限、代价和 tie-breaker 必须写入配置并固定。
4. 匹配到 receiver track 才能进入 common-target 主任务；没有 receiver track 的 peer candidate 记录为 `shared_only`，仅统计覆盖率，不计算本阶段 Harm Rate。
5. association 在每个 sequence 按时间顺序回放，保留 births、misses、re-activation 和 ID-switch 事件。不能随机打乱帧来构造 ledger。

## 3. Ledger schema

每行的核心键保持用户指定的六元组：

```text
(receiver, source, receiver_target_id, send_time, horizon, forecast)
```

落盘时使用 JSONL/Parquet 均可，但必须有下列字段：

| 字段 | 类型/约束 | 含义 |
|---|---|---|
| `sequence_id` | string | 场景/序列；不能只用全局帧号 |
| `receiver` | string | 当前接收 CAV |
| `source` | string | 预测来源 CAV；ego 行使用固定 `ego` |
| `source_track_id` | int/string | 来源本地 ID，审计用 |
| `receiver_target_id` | int/string | 接收方目标 ID，ledger 主关联键 |
| `send_time` | float/int | 发送预测时刻 |
| `arrival_time` | float/int | 接收方收到该预测的时刻 |
| `horizon` | float | 预测偏移，OPV2V 先覆盖 0.1--5.0 s 的实际采样点 |
| `forecast` | object | 不可变的轨迹/GMM payload；必须带 `frame=receiver@send_time`、`model`、`time_offsets_s`，多模态时保存 `trajectories_xyz[M,T,3]`、`scores[M]`，并把 top-score 模式在本行 horizon 的点写入 `position_xyz` |
| `association_confidence` | float | receiver-side gate/assignment 置信度，不是 GT 置信度 |
| `track_age`, `miss_count` | int | 接收方 lifecycle 元数据 |
| `target_scope` | enum | `common` 或 `shared_only` |
| `matured` | bool | 后续观测是否已到期并成功回填 |
| `observation_time` | nullable | 实际用于验证的 receiver observation 时间 |
| `matched_receiver_track_id` | nullable | 回填时的 receiver track ID；用于审计 ID-switch |
| `realized_state` | nullable array | receiver-only 观测状态，不是 GT |
| `error_type` | enum | `displacement`, `nll` 或 `none` |
| `realized_error` | nullable float | peer/ego forecast 相对 receiver observation 的误差 |
| `valid_mask` | bool | 观测可见、时间对齐且 ID 未失效才为 true |
| `censor_reason` | nullable enum | `not_observed`, `occluded`, `track_dead`, `id_switch`, `timeout` 等 |

`forecast` 在写入后不可修改；真实 CMP/MTR 预测从 GT-free MTR wrapper 输出的 normalized record 以 derived-ledger 方式附加，要求 `source` + `source_track_id`（aggregate 还要求 `receiver_target_id`），不接受官方方便评测用的 `object_id`、`gt_trajs` 或按数组序号猜 ID。hindsight 只能追加 `matured`、观测和 error 字段。ledger 行的事件约束为

```text
send_time <= arrival_time <= observation_time
observation_time ~= send_time + horizon
```

允许的时间容差和异步重采样规则必须固定在 config 中。一个 key 只能有一行；重传、丢包和重复包通过 `packet_id`/`arrival_time` 作为额外审计字段区分，而不能覆盖旧 forecast。

## 4. Common/shared-only 定义

- **Common target**：在 receiver 侧同一时间窗口内既有 ego local track，又有至少一个经过 GT-free association 的 peer forecast，并且在所评 horizon 能得到 receiver observation。common 的判定只用 receiver artifact；GT 仅用于评测这次判定是否正确。
- **Shared-only target**：peer 有 forecast，但 receiver 当前没有 local track。它们计入 source/sequence/horizon 的覆盖率和删失原因分布，不进入本阶段 peer-vs-ego error、CMP aggregate error 或 Harm Rate。
- 评测脚本另外输出 common-target 的 GT precision/recall 和 shared-only 的离线命中率，但不得把离线匹配结果写回在线 ledger。

## 5. 离线评测与主指标

所有 association 指标在独立 evaluator 中用 GT 做一次 Hungarian matching；预测输入和 ledger 不读该表。

1. **Association**：按 IoU/中心距离门限报告 detection/track precision、recall；跨源关联同时报告两个主分母：`end_to_end_recall = correct common / raw receiver-GT 中存在对应 peer object 的全部到达机会`，`conditional_recall = correct common / peer 与 receiver local track 都有独立 GT match 的机会`。end-to-end 的 raw peer opportunity 总数、receiver 可见率和“除以全部 peer opportunities”的补充 recall 也一并输出，避免把 receiver visibility 与 detector/track miss 混成一个数字。后者不能替代前者，因为 receiver track miss 会被条件化掉。按 sequence 报 ID-switch 次数（以及每 100 tracked frames 的归一化值）；track persistence 定义为一个 receiver track 在其有效生命周期中连续有效帧比例，并报告均值/中位数/分位数。
2. **Track survival**：对离线起始 GT-match 的 local track，报告 exact frame `@1/2/3/5s` 的同一 local ID presence，以及仍匹配同一 GT 的 survival；未来没有记录帧的起点不进入该 horizon denominator，离开场景/遮挡在有未来帧时按失败保留。
3. **Ledger hindsight coverage**：
   - row coverage = `matured && valid` 行数 / 所有 common peer forecast 行数；
   - horizon coverage 按 `horizon`、延迟、source、sequence 分层；
   - target coverage = 至少一个 matured row 的 common `(sequence, receiver_target_id, source)` 对 / common 对总数；
   - shared-only 只报告候选行数和后续 receiver 重新出现的比例。
4. **Trajectory error**：在相同 `(sequence, receiver, receiver_target_id, send_time, horizon)` 的 paired rows 上计算 ego-only、peer realized、CMP aggregate 的 ADE/FDE。主比较使用 top-scored/committed trajectory，避免 minADE 的 oracle mode 选择；为兼容 CMP，再单独报告 native minADE6/minFDE6，不能把两者混用。
5. **Harm Rate**：在 valid common paired rows 上

   ```text
   HarmRate(eps) = mean[ E_aggregate > E_ego + eps ]
   ```

   主值预注册 `eps=0.10 m`，同时给出 `eps=0` 敏感性；分 horizon、source、延迟和 association-confidence 报告 bootstrap 置信区间。它是实证指标，不是保证。
6. **CMP gap**：同一检测权重、同一 held-out sequence、同一目标键，配对报告 GT-assisted CMP 原路径与 GT-free 重建路径的 ADE/FDE 差值和相对百分比；同时给 coverage-aware 表，防止通过丢弃难匹配目标制造虚假增益。

## 6. 只做简单 hindsight baseline

本阶段不实现 attention、concordance、censoring model 或 regret network。先固定以下可解释 baseline：

- `recent-error`：使用该 source-target 最近一个 matured 的 peer-vs-ego error difference；未知时 fallback ego，非正差值才启用 peer。
- `EWMA`：对 source-target 的差值做固定 `alpha`（默认 0.3）的指数平均；冷启动只报告，不用 GT 填充。若 target 历史不足，可单独报告 source-level metadata prior，但不能把未来目标标签迁移进来。
- `metadata-only`：只使用 delay、距离、track age、miss count、association cost/confidence 和 detection score。第一版采用固定阈值/验证集拟合的线性规则，推理时不读取 realized error；不得扩展成复杂 learned controller。

每个 baseline 都必须和 `ego-only`、原 CMP aggregate、以及始终启用 peer 的简单策略比较。baseline 的启用/回退决策只改变评测输出，不改写原始 ledger。

## 7. GT-free 验收与止损

### Gate -1：数据契约

- 静态检查：在线导出、association、ledger、prediction 输入路径不能 import/read GT、`gt_object_id_tensor`、dataset actor key 或 GT pose。
- 动态检查：用移除 GT 文件/替换为 poison object 的输入回放，在线产物仍可生成；GT 只能由独立 evaluator 读取。
- 账本审计：随机抽取行检查 key 唯一性、事件顺序、forecast 不变和 no-future leakage。

### Gate 0：是否足以支撑 reliability aggregation

在 held-out sequence 上先报告完整分布，不按单个最好场景判定。暂定继续条件是：common track precision/recall 均不低于 0.80，track persistence 中位数不低于 0.70，ID-switch 不高于每 100 个有效 track frames 10 次，common valid hindsight rows 至少 1,000 行且 row coverage 不低于 0.50；同时 GT-free CMP 的 coverage-aware ADE/FDE 相对 GT-assisted CMP 不劣化超过 10%。任一条件失败，先记录实际阻塞（检测、跟踪、关联、时间对齐或 ledger 删失），不进入 reliability aggregation，转为 GT-free association/robustness benchmark。

这些是进入下一阶段的支持性门槛，不是统计保证；第一次跑完后只允许在预注册的验证集上调整实现参数，不能用测试集反调 gate。

## 8. 数据切分与可复现性

- 按 scene/sequence held-out，不能按相邻帧随机切分；所有 baseline 只能在 train/validation 上拟合阈值。
- OPV2V 先做主实验，V2V4Real 只在契约稳定后验证；感知层 pose/delay/丢包扰动需要重新导出 detection/tracking，不能只在 prediction cache 上伪造。
- 原始 detection、local tracks、association events、ledger、offline GT evaluator 输出分目录保存，并记录 config、代码版本、运行时间和数据 split。
