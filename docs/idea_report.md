# RV-HCA idea report（Part 1：related work 与问题边界）

版本：2026-09-04  
状态：**Part 1 与 GT-free Gate -1 CPU 数据契约已完成；真实 MTR 输入仍待兼容 GPU/完整 checkpoint**

## 1. 题目与问题收窄

拟题：**Receiver-Verifiable Harm-Controlled Aggregation for Asynchronous Cooperative Motion Prediction**。

核心问题保持为：

> Can a receiver use realized errors of past peer forecasts to identify unreliable collaboration and reduce harmful cooperative prediction without ground-truth information at inference time?

第一阶段不声称已经解决 reliability/regret controller，而是先验证数据契约：GT-free detector-to-tracker-to-association 是否能产生足够稳定的 receiver target ID 和 prediction ledger，使 common targets 上的 peer realized error、CMP aggregate error 与 Harm Rate 可被公平测量。

叙事采用“收益依赖于 target、source、horizon 和运行条件，现有聚合器在部分条件下出现负协作”，不采用“协作者越多越差”的全称断言。CooperScene 的 CMP 对比确实给出 V→V+V 在 1/3/5 s 的恶化，而 V+2V 又可优于单车；这支持条件性负协作动机，不支持单调规律。

## 2. Part 1：研究必要性

### 2.1 应用必要性

异步 V2X 预测同时受到传输延迟、定位/坐标偏差、检测漏检和 source-specific 系统误差影响。只报告平均协作增益会把“某个来源帮助了某个目标”与“另一个来源造成 harm”混在一起，无法回答部署端是否应使用一个具体 peer。接收方已经拥有本地检测、跟踪和到达时间，因此用这些已揭示信号构造审计账本是部署可行的最小切口。

### 2.2 学术必要性

CMP 已证明多车预测聚合可以工作，但其论文方法是上下文条件化的 attention；它没有给出一个独立、可校准的 receiver-observable source×target×horizon reliability 量。Online Aggregation 把 predictors 当黑盒专家并用下一时刻损失在线更新全局权重；Learning Through Retrospection 把自身历史预测误差回馈单车模型。两者都留下了跨车辆身份关联、接收方可见性删失和 source-specific pose bias 的问题。Late Fusion 已明确处理 receiver-side prediction map、时空对齐和 local/shared-only 分类，因此本项目不能把 association 或 shared-only 本身宣称为新颖贡献；本项目先验证其在 CMP OPV2V pipeline 中的 GT-free data contract，再决定是否有条件进入 reliability aggregation。

### 2.3 研究必要性与止损

GT-free 不是把 GT-assisted cache 换一个文件名。CMP 当前导出脚本在 AB3DMOT 前后都使用 GT object ID，MTR 数据加载也以 GT/pred key intersection 过滤目标。若不先重建 tracking/association，任何 hindsight harm 结论都可能是标签泄漏。因而本阶段的科学产物是：可审计的 GT-free pipeline、common/shared-only 划分、ledger coverage，以及与原 GT-assisted CMP 的 paired ADE/FDE 差距。

## 3. Related-work matrix

| 工作 | 输入/输出与核心机制 | 与 RV-HCA 的关系 | 当前处理 |
|---|---|---|---|
| **CMP**（Wang et al., arXiv:2403.17916；RA-L 版本） | OPV2V/V2V4Real 的 cooperative perception + AB3DMOT + MTR；预测通过上下文条件化 attention 聚合；固定 100 ms/一帧延迟设定 | 代码基线和要重建的 pipeline；已有 contextual weighting，不能写成等权聚合；GT-assisted 导出是本项目 Gate -1 | 复现原路径；另建 GT-free export/replay；保留 native CMP aggregate |
| **CooperScene**（arXiv:2606.31219） | 多车协作场景/检测基准；报告不同协作者数量、延迟和模型的条件性增益与退化 | 提供负协作动机；V→V+V 退化但 V+2V 可改善，故只作条件性证据 | 只作 related-work/motivation，不把其检测 harm 当 trajectory 结论 |
| **Collaborative Trajectory Prediction via Late Fusion**（arXiv:2604.22973） | receiver-side prediction map；temporal alignment + spatial association；Category L（local）/S（shared-only）；GP 轨迹融合并携带 source uncertainty | 最近的 association/数据契约先例；S/L、时空关联和 GP 不属于本项目新颖性；其 uncertainty/source calibration 是后续边界 | 借鉴字段和 common/shared-only 协议；不复现 GP fusion |
| **Online Aggregation of Trajectory Predictors**（arXiv:2502.07178） | 黑盒 GMM experts；用下一时刻揭示状态计算损失，SQUINT/OCO 更新全局 expert mixture，带非平稳 discount | 最接近“用 realized error 选协作者”的方法；但专家不是跨车 source，权重不是 target×horizon，且不处理跨源身份与删失 | Part 1 对照；本阶段不实现 SQUINT/regret controller |
| **Learning Through Retrospection**（arXiv:2504.13785） | 单车 rolling error buffer；掩码未来不可用项；Ret-S/Ret-C 用历史预测–观测差修正后续预测 | 时间 hindsight 的直接先例；本项目区别是验证他车预测，且 receiver observation 对目标可见性/ID-switch 有删失 | 只用其“recent error 有信息”的启发；不实现 learned retrospection |
| **V2INet: Conformal Trajectory Prediction**（arXiv:2408.00374） | 多视角图/跨图 attention；后验 conformal prediction 输出覆盖区间 | 说明事后不确定性校准是邻近方向；它校准区间而非按 source×target×horizon 改融合权重 | 作为 uncertainty boundary；不实现 conformal module |
| **V2VNet**（arXiv:2008.07519） | 图消息传递的 V2V 中间特征协作 | CMP perception/fusion 的历史基线；不能解释 receiver-side realized error | 只保留作 pipeline baseline |
| **OPV2V**（arXiv:2109.07644） | CARLA/OpenCDA 多车协作数据集，含同步/异步和 pose 设定 | 主实验数据契约和 scene-held-out split；GT 仅用于离线 evaluator | 先 OPV2V，稳定后再 V2V4Real |
| **V2V4Real**（arXiv:2303.07601） | 真实 V2V 感知/跟踪数据 | 用于跨域验证 tracking/association 与 ledger coverage，而非先行开发 | Gate 0 通过后验证 |
| **V2X-Graph**（arXiv:2311.00371） | 将多源轨迹作为图节点并做延迟对齐/交互建模 | 说明 source/target 图表示已有先例；本项目贡献边界必须放在 receiver-verifiable hindsight 与评测协议 | 作为后续 baseline/相关工作 |

### 3.1 新颖性边界

不能宣称的新颖点：attention 聚合、GP/方差融合、local/shared-only 分类、一般的时空 association、单车 error buffer、conformal interval、在线专家混合本身。

本项目只有在 Gate 0 通过、并在 common targets 上显示稳定的可预测性后，才考虑把贡献收窄为：

1. **跨车 receiver verification**：误差来自接收方自己的观测，而不是 source 自报方差；
2. **GT-free ledger contract**：local AB3DMOT ID → receiver target ID → matured peer forecast 的顺序回放和删失审计；
3. **按 target×horizon 的实证 harm protocol**：paired ego/peer/CMP errors、Harm Rate 和 risk–coverage（后者属于下一阶段，不在当前实现）；
4. **common/shared-only 的可复现边界**：shared-only 在本阶段只计覆盖率，避免用不可验证目标制造假性收益。

如果 GT-free association 的 precision/recall、ID-switch、track persistence 或 hindsight coverage 不足，则不进入 reliability controller；研究结论转为 GT-free cooperative prediction association/robustness benchmark，而不是强行包装成 RV-HCA。

## 4. 本阶段研究问题与可证伪预测

- **RQ1（契约可行性）**：不使用 GT object ID，receiver-side association 能否在 OPV2V common targets 上保持可接受的 precision/recall、ID-switch 和 track persistence？
- **RQ2（可观测 hindsight）**：有多少 peer forecast 行在后续 receiver observation 到达后能合法回填 realized error？coverage 是否随 horizon、delay、source 和 track lifecycle 系统变化？
- **RQ3（负协作可测量性）**：在同一 receiver-target-horizon 配对上，ego-only、peer、CMP aggregate 的误差和 Harm Rate 是否能在 GT-free ledger 上被稳定计算，并与原 GT-assisted CMP 形成可解释差距？

本阶段不把“recent/EWMA/metadata 选择能降低 harm”写成已知结论；它只是 Gate 0 通过后的最小可证伪实验。

## 5. 已读资料与本地副本

论文 PDF 和纯文本精读副本位于 [`docs/papers/`](./papers/)，包括 CMP、Late Fusion、Online Aggregation、Learning Through Retrospection、CooperScene、V2INet。CMP 官方源码审计副本位于 [`vendor/CMP/`](../vendor/CMP/)。在线引用使用论文/作者提供的 arXiv 与 GitHub 页面，不把下载归档的完整性误写成代码可运行性。

## 6. 状态与下一步

`Part 1`、Gate -1 的 GT-free export/replay、独立 association evaluator、track survival 和 prediction-ledger contract 已完成。PointPillar formal replay 的 track P/R=`0.987/0.722`，receiver-visible end-to-end/conditional association recall=`0.336/0.990`，raw receiver visibility=`0.825`，ledger hindsight coverage=`0.721`；因此当前不进入 reliability controller。CoBEVT-c256 已从 CMP 官方 raw detector cache 重新 sanitizer 并做了单场景 CPU 方向性复核，但 cooperative detector 语义和 false positives 仍需 held-out 验证。

下一步只有在兼容 GPU、完整 MTR checkpoint 和 GT-free source-local-ID wrapper 三者齐备后，才运行真实 CMP/MTR per-source multimodal predictions，写入 derived ledger，再按 source/target/horizon 和 held-out scene 检验历史 realized-error predictiveness。当前 formal ledger 的脚本状态为 `BLOCKED_NO_REAL_MTR`；不把 constant-velocity probe 当作 MTR，也不提前报告 NO-GO。

## References

1. [CMP paper](https://arxiv.org/abs/2403.17916) / [official CMP repository](https://github.com/tasl-lab/CMP)
2. [CooperScene](https://arxiv.org/abs/2606.31219)
3. [Collaborative Trajectory Prediction via Late Fusion](https://arxiv.org/abs/2604.22973)
4. [Online Aggregation of Trajectory Predictors](https://arxiv.org/abs/2502.07178)
5. [Learning Through Retrospection](https://arxiv.org/abs/2504.13785)
6. [V2INet: Conformal Trajectory Prediction](https://arxiv.org/abs/2408.00374)
7. [V2VNet](https://arxiv.org/abs/2008.07519)
8. [OPV2V](https://arxiv.org/abs/2109.07644)
9. [V2V4Real](https://arxiv.org/abs/2303.07601)
10. [V2X-Graph](https://arxiv.org/abs/2311.00371)
