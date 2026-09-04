# RV-HCA 最后核心科学检验（3-scene GT-free MTR ledger）

更新时间：2026-09-04。此检查只做指标核对和离线可预测性分析；没有训练
attention、MLP、reliability controller 或 aggregation controller。

## 数据和标签边界

- 输入是三场景 GT-free local-track → receiver-side association → MTR ledger。
- 真实 CMP/MTR 记录为 peer `cmp_mtr_no_agg` 和 ego `cmp_mtr_no_coop`，两侧均为
  `(6, 50, 3)` multimodal trajectories；导出阶段 `uses_gt=false`。
- 本检查的相对 harm 标签只在离线统计中使用：
  `harm = peer_realized_error > ego_only_error + 0.1m`。
  realized error 是 receiver 的 ego-only tracker 观测回填值；GT 只由独立
  offline evaluator 用于最终指标、association audit 和 oracle。
- A/B/C 是一个线性 logistic **诊断 probe**，只回答加入特征是否带来 held-out
  AUC 增量，不产生在线 gate、动作或 checkpoint。

## 1. 绝对误差和约 11 m 的来源

下面是独立 offline evaluator 的 selected-score MTR 点误差。每个 horizon 是
receiver-at-send 坐标中该时刻的 3D Euclidean GT 误差；这里的 `ADE/FDE` 只是
单点误差别名，不是整条轨迹的 ADE/FDE。

| horizon (s) | rows | ego top-score (m) | peer top-score (m) |
|---:|---:|---:|---:|
| 0.1 | 4,631 | 1.014 | 1.046 |
| 0.3 | 4,560 | 2.766 | 2.800 |
| 0.5 | 4,481 | 4.518 | 4.557 |
| 1.0 | 4,215 | 9.056 | 9.163 |
| 2.0 | 3,697 | 18.997 | 19.149 |
| 3.0 | 3,203 | 30.183 | 30.363 |
| 5.0 | 2,188 | 55.210 | 55.000 |

这组 GT top-score 统计跨 horizon 的总体均值为 peer **13.533 m**、ego
**13.473 m**（26,975 paired rows）。此前约 **11 m** 的数值来自另一项
GT-free receiver-observation audit：23,618 个同时具有 matured MTR 和 ego-only
观测的 common rows 上，peer **11.308 m**、ego **11.055 m**。两者的行筛选、未来
观测可用性和误差参照不同，不能混写成同一个 ADE。

完整六模态轨迹的 native-style 指标如下（每个 source/target/send-time 只计一次，
只对未来 0.1--5.0 s 全部可用的 rows 统计）：

| model/role | rows | minADE6 (m) | minFDE6 (m) | top-score full ADE (m) | top-score full FDE (m) |
|---|---:|---:|---:|---:|---:|
| peer `cmp_mtr_no_agg` | 2,238 | 11.682 | 23.139 | 26.774 | 54.710 |
| ego `cmp_mtr_no_coop` | 2,530 | 11.003 | 21.469 | 26.436 | 54.101 |

## 2. Relative harm（主标签）

GT-free receiver-observation ledger 的 23,618 paired rows：总体 harm rate 为
**46.70%**，peer−ego 平均误差为 **+0.252 m**。按 horizon 的 peer/ego/error
和 harm 如下：

| horizon (s) | rows | ego (m) | peer (m) | peer−ego (m) | harm |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 4,645 | 0.929 | 0.979 | +0.050 | 40.88% |
| 0.3 | 4,415 | 2.672 | 2.721 | +0.049 | 44.78% |
| 0.5 | 4,216 | 4.407 | 4.482 | +0.075 | 46.20% |
| 1.0 | 3,755 | 8.911 | 9.102 | +0.191 | 49.08% |
| 2.0 | 2,979 | 18.647 | 19.010 | +0.363 | 50.05% |
| 3.0 | 2,322 | 29.438 | 30.237 | +0.799 | 52.02% |
| 5.0 | 1,286 | 53.692 | 54.891 | +1.199 | 51.56% |

## 3. LOSO A/B/C relative-harm probe

A = 当前 receiver metadata（association confidence、track age、miss count、
arrival delay）；B = A + ego realized-error history；C = B + peer realized-error
history。每个 cell 为 `AUC(A) / AUC(B) / AUC(C) (C−B)`。模型只在另外两个
scene、同一 horizon 上拟合，测试 scene 完全留出；历史严格满足
`observation_time <= current send_time` 且 `history send_time < current send_time`。

### target-level history，latest

| horizon | scene 18 | scene 20 | scene 22 |
|---:|---:|---:|---:|
| 0.1 | .570/.625/.893 (+.269) | .607/.601/.873 (+.273) | .778/.785/.921 (+.136) |
| 0.3 | .552/.589/.807 (+.218) | .580/.623/.831 (+.208) | .754/.709/.758 (+.050) |
| 0.5 | .544/.558/.758 (+.201) | .581/.622/.777 (+.154) | .601/.598/.615 (+.017) |
| 1.0 | .530/.511/.575 (+.064) | .578/.615/.679 (+.063) | .575/.616/.564 (−.052) |
| 2.0 | .532/.556/.534 (−.022) | .594/.627/.644 (+.017) | .620/.614/.589 (−.025) |
| 3.0 | .498/.546/.505 (−.040) | .593/.558/.558 (−.000) | .672/.685/.645 (−.040) |
| 5.0 | .498/.455/.455 (+.000) | .454/.453/.453 (+.000) | .163/.163/.163 (+.000) |

### target-level history，EWMA

| horizon | scene 18 | scene 20 | scene 22 |
|---:|---:|---:|---:|
| 0.1 | .570/.625/.863 (+.238) | .607/.601/.832 (+.231) | .778/.785/.869 (+.084) |
| 0.3 | .552/.589/.775 (+.186) | .580/.623/.812 (+.189) | .754/.709/.716 (+.008) |
| 0.5 | .544/.558/.738 (+.180) | .581/.622/.766 (+.144) | .601/.598/.567 (−.031) |
| 1.0 | .530/.511/.596 (+.085) | .578/.615/.676 (+.061) | .575/.616/.527 (−.090) |
| 2.0 | .532/.556/.519 (−.038) | .594/.627/.631 (+.005) | .620/.614/.569 (−.046) |
| 3.0 | .498/.546/.500 (−.046) | .593/.558/.553 (−.005) | .672/.685/.645 (−.040) |
| 5.0 | .498/.455/.456 (+.000) | .454/.453/.454 (+.000) | .163/.163/.163 (+.000) |

### source-level history，latest

| horizon | scene 18 | scene 20 | scene 22 |
|---:|---:|---:|---:|
| 0.1 | .570/.630/.701 (+.071) | .607/.555/.562 (+.007) | .778/.834/.807 (−.027) |
| 0.3 | .552/.583/.646 (+.063) | .580/.576/.624 (+.048) | .754/.754/.717 (−.037) |
| 0.5 | .544/.566/.613 (+.047) | .581/.571/.593 (+.022) | .601/.636/.619 (−.018) |
| 1.0 | .530/.498/.508 (+.011) | .578/.592/.613 (+.021) | .575/.579/.563 (−.016) |
| 2.0 | .532/.519/.519 (+.000) | .594/.426/.435 (+.008) | .620/.632/.632 (+.001) |
| 3.0 | .498/.538/.584 (+.046) | .593/.471/.472 (+.001) | .672/.704/.625 (−.079) |
| 5.0 | .498/.445/.445 (+.000) | .454/.452/.452 (+.000) | .163/.163/.163 (+.000) |

### source-level history，EWMA

| horizon | scene 18 | scene 20 | scene 22 |
|---:|---:|---:|---:|
| 0.1 | .570/.630/.731 (+.100) | .607/.555/.582 (+.027) | .778/.834/.780 (−.054) |
| 0.3 | .552/.583/.696 (+.113) | .580/.576/.753 (+.177) | .754/.754/.729 (−.025) |
| 0.5 | .544/.566/.654 (+.088) | .581/.571/.714 (+.144) | .601/.636/.634 (−.002) |
| 1.0 | .530/.498/.525 (+.027) | .578/.592/.651 (+.059) | .575/.579/.489 (−.090) |
| 2.0 | .532/.519/.568 (+.049) | .594/.426/.494 (+.068) | .620/.632/.538 (−.093) |
| 3.0 | .498/.538/.498 (−.041) | .593/.471/.469 (−.002) | .672/.704/.528 (−.176) |
| 5.0 | .498/.445/.445 (+.001) | .454/.452/.452 (+.000) | .163/.163/.163 (+.000) |

总体判断：target-level peer history 在 0.1--0.5 s 的 C−B 明显为正（latest
约 +0.017 到 +0.273），到 2--3 s 变成接近零或负；source-level 增量整体小且
跨 scene 符号不稳定。21 个 scene×horizon cells 中，target latest 有 14 个
正增量（均值 +0.071、median +0.017），target EWMA 有 13 个（均值 +0.053、
median +0.005）；source latest 的均值仅 +0.008，source EWMA +0.018。因而
“控制 target difficulty 和 metadata 后 peer history 有独立信息”只在短 horizon、
尤其 target-level 上得到支持，不能外推为稳定的长 horizon/source-level 结论。

## 4. ego-vs-peer oracle 和 naive mean fusion

这是可实现动作空间的离线 GT 上界，不是可部署 selector：
`oracle_error = min(ego_error, peer_error)`；`naive_mean_fusion` 是两个 top-score
3D 位置的算术平均。

| horizon | ego (m) | peer (m) | oracle (m) | oracle headroom (m, %) | mean fusion (m) | fusion−ego (m) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 1.014 | 1.046 | 0.870 | 0.145 (14.3%) | 1.020 | +0.005 |
| 0.3 | 2.766 | 2.800 | 2.385 | 0.381 (13.8%) | 2.772 | +0.006 |
| 0.5 | 4.518 | 4.557 | 3.912 | 0.606 (13.4%) | 4.524 | +0.006 |
| 1.0 | 9.056 | 9.163 | 7.752 | 1.304 (14.4%) | 9.082 | +0.026 |
| 2.0 | 18.997 | 19.149 | 15.953 | 3.044 (16.0%) | 18.977 | −0.020 |
| 3.0 | 30.183 | 30.363 | 25.203 | 4.979 (16.5%) | 30.099 | −0.084 |
| 5.0 | 55.210 | 55.000 | 47.259 | 7.951 (14.4%) | 54.777 | −0.433 |

完整 paired rows 的 oracle headroom 为 **2.047 m / 15.2%**；naive mean fusion
为 **13.432 m**，相对 ego **−0.041 m**，但短 horizon 略差、长 horizon 略好，
不能解释成可靠协作收益。

## 5. 短 horizon → 长 horizon relative harm

对同一 scene/receiver/source/target，仅使用在当前 send time 前已兑现的 peer
误差；下面是 pooled AUC（latest/EWMA），标签是当前 2/3/5 s 的 relative harm。

| past peer → current harm | rows | AUC latest | AUC EWMA |
|---|---:|---:|---:|
| 0.3 → 2.0 s | 2,613 | .737 | .727 |
| 0.3 → 3.0 s | 2,009 | .760 | .737 |
| 0.3 → 5.0 s | 1,058 | .661 | .624 |
| 0.5 → 2.0 s | 2,417 | .712 | .713 |
| 0.5 → 3.0 s | 1,842 | .730 | .717 |
| 0.5 → 5.0 s | 937 | .625 | .597 |
| 1.0 → 2.0 s | 2,019 | .692 | .705 |
| 1.0 → 3.0 s | 1,500 | .699 | .712 |
| 1.0 → 5.0 s | 682 | .584 | .593 |

短到长 pooled 结果在 2/3 s 有中等信号，5 s 明显减弱；scene-level 方向并不
完全一致（完整逐 scene 数值见 JSON）。这支持“存在条件性预测信号”，不足以
支持当前就训练 controller。

## 产物和停止点

- 完整指标：[`offline_evaluation.json`](../data/intermediate/gtfree_cpu/opv2v_3scene_mtr_3scene_eval/offline_evaluation.json)
- 完整 A/B/C 与短到长结果：[`mtr_predictiveness.json`](../data/intermediate/gtfree_cpu/opv2v_3scene_mtr_3scene_eval/mtr_predictiveness.json)
- GT-free contract audit：[`contract_audit.json`](../data/intermediate/gtfree_cpu/opv2v_3scene_mtr_3scene_eval/contract_audit.json)
- 分析脚本：[`analyze_mtr_predictiveness.py`](../scripts/analyze_mtr_predictiveness.py)

本阶段到此停止：没有训练或实现 reliability controller；后续是否继续应以新的
数据/场景和预注册的 selective-action 评测为前提，而不是把本诊断 probe 当成方法。

## 6. global-world XY 与 receiver-observation 路径审计

对同一批 23,618 个 valid common MTR rows，审计了：

`source-local forecast → source_pose_at_send → global world`

以及

`attached receiver@send_time forecast → receiver-observation realized_state`。

结果：

- source prediction 经 `source_pose_at_send` 再经 receiver send pose 变换后，与
  ledger 中的 receiver prediction 残差为 **0 m**；raw source pose 与 local-track
  send pose 残差也为 **0 m / 0°**。
- future target world center 经 **receiver_pose_at_send 的逆变换**后，与
  `realized_state` 残差为 **0 m**。`observation_time − send_time − horizon` 最大
  约 `1.8e−15 s`。
- 3D Euclidean world error 与 receiver error 逐行一致：peer 最大差
  `2.63e−13 m`，ego 最大差 `2.76e−13 m`，即数值精度内一致。
- 取完整 6-DoF 变换后的 XY 投影时，peer/ego 最大差分别为 `0.565/0.360 m`；
  这不是时间错位，而是 receiver pose 含 roll/pitch、且预测与目标 z 不完全相同，
  所以 3D 刚体距离保持不变，但简单 XY 投影不保持不变。将 receiver pose 置为
  yaw-only 的 planar control 后，world XY 与 receiver XY 最大差为
  `2.04e−13/1.92e−13 m`。

因此当前语义已经定位清楚：`observation_frame_idx` 是 `send_frame_idx + horizon`
的 future target 观测时刻，但 `realized_state` 被有意转换回
**receiver@send_time**；它没有使用 receiver 在 observation time 的 pose。若误用
observation-time pose，坐标会出现明显 frame shift（该控制量的均值约 11.52 m）。
完整逐行结果见 [`mtr_coordinate_path_audit.json`](../data/intermediate/gtfree_cpu/opv2v_3scene_mtr_3scene_eval/mtr_coordinate_path_audit.json)，
审计脚本为 [`audit_mtr_coordinate_paths.py`](../scripts/audit_mtr_coordinate_paths.py)。
