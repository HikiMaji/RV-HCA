# `re-9-4.md` 核验与本地修复记录

更新时间：2026-09-04。本文只记录本地代码变更；本轮没有提交或推送 GitHub。

## 逐条判断

| 审查项 | 对当前代码的判断 | 本地处理 |
|---|---|---|
| MTR 多帧 history 混用每一帧的 source-local 坐标 | 属实。`center_local/yaw_local` 随 source pose 变化，却被当作同一轨迹坐标系 | `run_gtfree_mtr.py` 现在只接受 `center_world/yaw_world` 构造 history；manifest 同时保存 world 字段并标记 `history_reference_frame=world` |
| 直接打开 AB3DMOT `ego_com` 即可补偿 | 结论不完整。上游 OPV2V AB3DMOT 的 `ego_motion_compensation` 当前对坐标是 no-op，且其 OXTS 约定不是本项目的 OPV2V 6DoF pose | 不照搬 flag；`online.py` 先把检测中心/航向用测量 pose 变到固定 world，再在 world 中跟踪，输出时反变换回当前 local。该路径仍完全 GT-free |
| peer checkpoint 与 PointPillar history 分布不匹配 | 属实。peer YAML 声明 CoBEVT-c256 history，旧 replay 声明 PointPillar | MTR runner 增加 checkpoint/history provenance audit；默认拒绝未验证或不匹配输入，`--allow-input-mismatch` 仅用于明确标注的 smoke。增加 `--role peer/ego`，允许分别导出各自 native-distribution 的单角色结果；未把不兼容结果拼成科学 paired ledger |
| “EWMA” 实际是最近五项均值 | 属实 | predictiveness 分析改为显式 `ewma_alpha`（默认 0.3）的真正指数递推；short→long 分析也使用同一递推 |
| A/B/C 的 target-level ego history 被 source 条件化 | 属实 | target-level ego key 改为 `(scene, receiver, target, horizon)`，peer key 保留 source；同一 send time 的 ego history 跨 peer source 去重。source-level 描述仍保持 source 条件 |
| MTR XY 预测却把 3D 当主误差 | 属实 | primary top-score/native/oracle/Harm 误差改为 XY；3D 值写入 `*_error_3d` 并在 offline evaluator 的 supplemental 区域报告 |
| world/receiver 输出变换与时间审计 | 原有数值审计通过 | `audit_mtr_coordinate_paths.py` 兼容新的 `forecast_frame=world`；legacy `source@send_time` 仍可审计 |

## 已执行的本地验证

- `python -m compileall -q rvhca_cpu scripts`：通过。
- 随机 pose/detection round-trip：world-fixed detection 变换后反变换回 local，中心与航向恢复到数值误差范围内。
- fake tracker smoke：确认 AB3DMOT 接收的是 world 坐标，序列化的 `center_local` 仍是当前 pose 下的 local 视图。
- 固定 world tracker 的 12-frame scene-22 CPU replay：ledger、association、hindsight 字段正常生成；primary summary 明确标记 XY，3D 为 supplemental。
- manifest smoke：每个有效 history item 同时含 local/world 审计值，MTR reference frame 标记为 world。
- world-frame normalized MTR attachment smoke：`mtr_io.py` 将 world 轨迹正确变换到 receiver@send_time。

固定 world 的三场景 PointPillar replay（只用于验证跟踪/账本，不是 MTR
科学结果）重新通过独立 GT evaluator：track precision/recall 为
`0.9886/0.7429`，end-to-end/conditional association recall 为
`0.3396/0.9911`，ledger hindsight coverage 为 `0.7223`；track survival@1/2/3/5s
为 `0.8914/0.8229/0.7651/0.6782`。这些数值仅证明固定 world 路径没有破坏
契约，GT 只在 evaluator 中读取。

当前官方 peer/no-agg checkpoint 声明 `corpbevtlidar_delay_1_frame_aug_c256`
history，而 ego/no-coop 声明 `point_pillar_sinbevt` history；单个 PointPillar
replay 的 `--role both` 会被 strict audit 拒绝（peer family mismatch）。当前
容器也没有 CUDA device，因此没有把不兼容输入强行跑成新的 MTR paired 结果。

旧的三场景 MTR 数字（包括约 11m、Harm、A/B/C、oracle 和 short→long）不在本地修复后自动复用；它们必须在 fixed-world tracker、world history、XY metric 且通过输入分布审计后重新导出。当前尚未把不兼容的 peer/ego detector history 强行合成 paired scientific run。

## 当前边界

本轮仍不训练 reliability/controller，也没有修改 receiver association 算法。MTR runner 的 strict provenance gate 是防止再次把 PointPillar history 喂给 CoBEVT peer checkpoint；要得到可配对的最终实验，下一步必须先选定并生成与两个 checkpoint 数据分布一致、且共享可审查 receiver/target 键的 replay。未通过该 gate 的输出只能作为 smoke，不用于论文结论。
