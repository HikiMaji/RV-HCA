# RV-HCA 独立项目用户需求

版本：2026-09-02

## 范围

- 项目根目录为 `/root/autodl-tmp/RV-HCA`，不修改 `/root/autodl-tmp/C`。
- 当前只做 GT-free 数据契约、tracking/receiver-side association、prediction ledger 和离线指标。
- 暂不实现完整 reliability/regret controller、attention、concordance、censoring model 或复杂 learned selector。
- 主任务只做 common targets；shared-only 只统计覆盖率。

## GT 边界

- GT 不得参与 tracking ID、跨源 target matching、prediction input 或 ledger 构造。
- GT 只能在独立离线 evaluator 中用于 association precision/recall、ID-switch、track persistence、ADE/FDE 和 Harm Rate。
- 必须能在移除/poison GT 输入后回放在线 export/association/ledger。

## 必须产物

1. Part 1 related-work matrix。
2. raw detection → local AB3DMOT ID → coordinate/time alignment → receiver-side association 的可审计路径。
3. ledger 核心键 `(receiver, source, receiver_target_id, send_time, horizon, forecast)`，以及后续 receiver observation 的 realized-error 回填。
4. GT-free association 与 ledger coverage 报告，并和原 CMP GT-assisted path 做 paired ADE/FDE gap。
5. ego-only、peer realized、CMP aggregate、Harm Rate，以及 recent error/EWMA/metadata-only 三个最简单 hindsight baseline 的接口。

## 进入下一阶段的条件

只有当 common-target association 质量、track persistence、hindsight coverage 和样本量足以支持 held-out 实验时，才进入 receiver-side reliability aggregation；否则报告实际阻塞并转为 GT-free association/robustness benchmark。

## 当前状态

Part 1、数据契约和 CPU-only GT-free 回放已完成。当前实现覆盖 source-local AB3DMOT、测量 pose 对齐、packet-arrival receiver association、common/shared-only 标记、ledger hindsight 回填、独立 GT evaluator、end-to-end/conditional association recall、track survival 和三个简单 hindsight baseline。CoBEVT-c256 raw detector 也已 sanitizer 为高召回候选 cache。完整 CMP/MTR 预测和 reliability/regret controller 仍未实现；MTR 受完整 checkpoint、兼容 CUDA runtime 和 GT-free source-local-ID wrapper 共同阻塞。
