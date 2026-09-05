# RV-HCA

当前仓库先完成 GT-free 数据契约的 CPU 回放，不包含完整 reliability/regret controller。

项目根目录：`/root/autodl-tmp/RV-HCA`

## 可审查快照

仓库只提交 RV-HCA 自有代码、脚本、文档和小型结果摘要。OPV2V 原始数据、预测账本、checkpoint、`.venv`、论文文件和 CMP/OpenCOOD 第三方 checkout 均不提交；运行 MTR 需要按 [`docs/download_manifest.md`](docs/download_manifest.md) 准备本地依赖。CMP 上游代码见 <https://github.com/tasl-lab/CMP>。

## CPU 回放

环境与 GPU 边界见 [`docs/environment_status.md`](docs/environment_status.md)，数据下载状态见 [`docs/download_manifest.md`](docs/download_manifest.md)。

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

`mean_probe` 只用于验证 aggregate/error/Harm Rate 字段，不能当作 CMP 输出。独立 GT 评测：

```bash
PYTHONPATH=$PWD /root/autodl-tmp/RV-HCA/.venv/bin/python \
  scripts/evaluate_gtfree.py data/intermediate/gtfree_cpu/opv2v_3scene
```

不读取 raw labels 的契约审计（检查必需字段、因果时间顺序、唯一 ledger key 和禁止 GT 字段）：

```bash
PYTHONPATH=$PWD /root/autodl-tmp/RV-HCA/.venv/bin/python \
  scripts/audit_gtfree_contract.py \
  data/intermediate/gtfree_cpu/opv2v_3scene \
  --mtr-manifest data/intermediate/gtfree_cpu/opv2v_3scene/gtfree_mtr_input_manifest_fullhistory.jsonl
```

实现和结果说明见 [`docs/cpu_replay.md`](docs/cpu_replay.md)。GT 边界和 ledger 契约见 [`docs/gt_free_data_contract.md`](docs/gt_free_data_contract.md)。

## 高召回 detector 与 MTR 接口

官方 CMP CoBEVT-c256 raw detector 已 sanitizer 为
`data/intermediate/detections/opv2v_corpbevtlidar_delay_1_frame_aug_c256_test_gtfree.pkl`；其 candidate audit 与 PointPillar 对比见 [`docs/detector_cache_comparison.md`](docs/detector_cache_comparison.md)。

从 GT-free local tracks 构造 MTR 输入 manifest（完整 10 帧历史；当前 formal replay 产出 166,149 条）可运行：

```bash
PYTHONPATH=$PWD /root/autodl-tmp/RV-HCA/.venv/bin/python \
  scripts/build_gtfree_mtr_inputs.py \
  data/intermediate/gtfree_cpu/opv2v_3scene/local_tracks.jsonl \
  --output data/intermediate/gtfree_cpu/opv2v_3scene/gtfree_mtr_input_manifest_fullhistory.jsonl
```

真实 MTR normalized records 接入 derived ledger：

```bash
PYTHONPATH=$PWD /root/autodl-tmp/RV-HCA/.venv/bin/python \
  scripts/ingest_mtr_predictions.py \
  data/intermediate/gtfree_cpu/opv2v_3scene \
  /path/to/gtfree_mtr_predictions.jsonl
```

该入口会拒绝 `object_id`/`gt_trajs` 及缺少 `source_track_id` 的官方 native evaluator 输出；当前 checkpoint/runtime 状态见 [`docs/mtr_integration_status.md`](docs/mtr_integration_status.md)。历史误差 predictiveness 只对真实 MTR 行运行：

```bash
PYTHONPATH=$PWD /root/autodl-tmp/RV-HCA/.venv/bin/python \
  scripts/analyze_mtr_predictiveness.py \
  data/intermediate/gtfree_cpu/opv2v_3scene_mtr_3scene_eval/prediction_ledger.jsonl
```

修复后的 MTR wrapper 要求 history 与 checkpoint 的 detector family 一致；当前
官方 peer/no-agg 与 ego/no-coop checkpoint 使用不同 history family，因此应分别
使用 `run_gtfree_mtr.py --role peer` / `--role ego` 导出，只有共享且兼容的 paired
replay 才能使用 `--role both`。不兼容输入会被 strict audit 拒绝。

## Cross-replay canonical pairing（GT-free）

PointPillar fixed-world replay 是 canonical receiver state stream；CoBEVT 和
PointPillar 的 AB3DMOT ID 绝不直接 join。先在 packet arrival 做一次 CoBEVT
track 到 PointPillar canonical target 的几何匹配，随后 send 与 future 都只以同一
`receiver_target_id` 查询。旧的 `prediction_ledger.jsonl` 不作为 paired row 的
生成来源，因为它只保存既有 peer association 的子集，不能恢复完整 canonical
continuity。

先分别生成新的 fixed-world replay。PointPillar canonical replay 必须有
`receiver_target_states.jsonl`；两者都必须声明 `tracking_frame=world_fixed`（旧
`rvhca.cpu_replay.v0` 输出不满足此条件），然后：

```bash
PYTHONPATH=$PWD /root/autodl-tmp/RV-HCA/.venv/bin/python \
  scripts/build_cross_replay_pairing.py \
  /path/to/pointpillar_worldfixed_replay \
  /path/to/cobevt_worldfixed_replay

# 只在离线 audit 读取 raw OPV2V labels；不写回任何 online artifact。
PYTHONPATH=$PWD /root/autodl-tmp/RV-HCA/.venv/bin/python \
  scripts/evaluate_cross_replay_pairing.py \
  /path/to/pointpillar_worldfixed_replay \
  /path/to/cobevt_worldfixed_replay \
  /path/to/pointpillar_worldfixed_replay/cross_replay_pairing.jsonl \
  /path/to/OPV2V/test
```

当两个 standalone MTR export 都带完整且 `PASS` 的 provenance 后，构造 scientific
paired ledger：

```bash
PYTHONPATH=$PWD /root/autodl-tmp/RV-HCA/.venv/bin/python \
  scripts/build_paired_mtr_ledger.py \
  /path/to/pointpillar_worldfixed_replay \
  /path/to/cross_replay_pairing.jsonl \
  /path/to/cobevt_peer_mtr_predictions.jsonl \
  /path/to/pointpillar_ego_mtr_predictions.jsonl
```

该命令对 `input_distribution_status != PASS`、history/checkpoint family 不一致、
或 peer ID 未随其 replay 一致重编号的记录 fail-closed；不计算 Harm Rate，也不训练
reliability controller。

## 审查入口

建议先阅读 [`docs/rv_hca_core_scientific_check.md`](docs/rv_hca_core_scientific_check.md) 和 [`docs/mtr_integration_status.md`](docs/mtr_integration_status.md)，再查看 `data/intermediate/gtfree_cpu/opv2v_3scene_mtr_3scene_eval/` 下的 JSON 摘要。完整 ledger 和原始输入留在本地，不属于 GitHub 快照。

本地 review 核验记录见 [`docs/re_9_4_verification.md`](docs/re_9_4_verification.md)。
