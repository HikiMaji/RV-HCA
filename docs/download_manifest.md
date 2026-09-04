# RV-HCA 数据下载清单

更新时间：2026-09-04。当前下载策略是先准备能验证 GT-free 数据契约的最小可回放集；没有把 21 GB 的完整测试压缩包一次性落盘。

## 已完成

| 用途 | 本地路径 | 内容/状态 |
|---|---|---|
| OPV2V 标准测试 smoke scene | [`data/raw/OPV2V/test/2021_08_22_09_08_29`](../data/raw/OPV2V/test/2021_08_22_09_08_29) | 2 CAV、302 PCD、303 YAML、1,208 camera PNG；已按 ZIP 条目 CRC 校验并解压 |
| OPV2V 标准测试追加 scene | [`data/raw/OPV2V/test/2021_08_18_19_48_05`](../data/raw/OPV2V/test/2021_08_18_19_48_05) | 2 CAV、356 PCD、357 YAML、1,424 camera PNG；连续 ZIP 范围下载并逐条 CRC 校验 |
| OPV2V 标准测试追加 scene | [`data/raw/OPV2V/test/2021_08_20_21_10_24`](../data/raw/OPV2V/test/2021_08_20_21_10_24) | 3 CAV、295 PCD、296 YAML、1,180 camera PNG；连续 ZIP 范围下载并逐条 CRC 校验 |
| OPV2V additional smoke scene | [`data/raw/OPV2V/additional/test/2021_08_22_09_08_29`](../data/raw/OPV2V/additional/test/2021_08_22_09_08_29) | 302 additional YAML、1,812 张 BEV/visibility/lane PNG；additional YAML 含 Python/Numpy 专用 tag，不能直接用 SafeLoader |
| CMP 感知 checkpoint | [`data/assets/checkpoints/CMP/pretrained/opv2v/corpbevtlidar_delay_1_frame_aug_c256/net_epoch25.pth`](../data/assets/checkpoints/CMP/pretrained/opv2v/corpbevtlidar_delay_1_frame_aug_c256/net_epoch25.pth) | 739,545,807 bytes；官方 CMP Google Drive 资产 |
| Swin backbone | [`data/assets/checkpoints/CMP/pretrained/opv2v/swin-base-patch4-window7-224`](../data/assets/checkpoints/CMP/pretrained/opv2v/swin-base-patch4-window7-224) | config、safetensors、preprocessor config 已齐 |
| OPV2V 预处理聚类字典 | [`data/assets/preprocessed_data/opv2v/opv2v_cluster_64_center_dict.pkl`](../data/assets/preprocessed_data/opv2v/opv2v_cluster_64_center_dict.pkl) | 可读 pickle；用于 MTR 轨迹离散化 |
| MTR no-coop checkpoint | [`data/assets/checkpoints/CMP/MTR/output/opv2v_multiego_no_coop/ckpt/best_model.pth`](../data/assets/checkpoints/CMP/MTR/output/opv2v_multiego_no_coop/ckpt/best_model.pth) | 1,863,607,503 bytes；ZIP 4850 个成员、CRC 通过 |
| MTR no-agg checkpoint | [`data/assets/checkpoints/CMP/MTR/output/opv2v_multiego_cobevt_c256_no_agg/ckpt/best_model.pth`](../data/assets/checkpoints/CMP/MTR/output/opv2v_multiego_cobevt_c256_no_agg/ckpt/best_model.pth) | 1,863,607,503 bytes；ZIP 4850 个成员、CRC 通过 |
| MTR aggregate checkpoint | [`data/assets/checkpoints/CMP/MTR/output/opv2v_multiego_cobevt_c256/ckpt/best_model.pth`](../data/assets/checkpoints/CMP/MTR/output/opv2v_multiego_cobevt_c256/ckpt/best_model.pth) | 4,253,524,373 bytes；ZIP 5145 个成员、CRC 通过 |
| GT-free detector cache | [`data/intermediate/detections/opv2v_point_pillar_sinbevt_test_gtfree.pkl`](../data/intermediate/detections/opv2v_point_pillar_sinbevt_test_gtfree.pkl) | 从 CMP 官方 no-coop detection cache 移除 `matched_car_id`/fused-feature path；16 scenes、45 sources、5,976 frames、77,923 detector candidates；详见同目录 README |
| GT-free high-recall detector cache | [`data/intermediate/detections/opv2v_corpbevtlidar_delay_1_frame_aug_c256_test_gtfree.pkl`](../data/intermediate/detections/opv2v_corpbevtlidar_delay_1_frame_aug_c256_test_gtfree.pkl) | 从官方 CoBEVT-c256 raw output sanitizer 生成；candidate-level score 0.25 precision/recall=`0.887/0.915`（3-scene audit） |
| GT-free detector alternatives | [`data/intermediate/detections/opv2v_corpbevtlidar_delay_1_frame_aug_test_gtfree.pkl`](../data/intermediate/detections/opv2v_corpbevtlidar_delay_1_frame_aug_test_gtfree.pkl)、[`opv2v_point_pillar_v2vnet_multiego_test_gtfree.pkl`](../data/intermediate/detections/opv2v_point_pillar_v2vnet_multiego_test_gtfree.pkl) | 官方 CoBEVT delay-1 与 V2VNet multi-ego raw outputs 的同一严格 sanitizer；用于 detector ceiling 对照，不改变 GT-free replay contract |
| CMP 完整源码 | [`vendor/CMP-upstream`](../vendor/CMP-upstream) | 官方 `de0f83900abaff47d87bb7b8329a5ff4a369a49d`；与审计用最小源码分开保存 |

## MTR checkpoint 状态

- 三套官方 MTR 权重均已完整下载，正式文件中不再有 `.part` 断点；本地 `zipfile.testzip()` 全部通过。
- 当前容器内存上限约 2 GiB，`torch.load` 反序列化 no-coop 时被 OOM；因此 ZIP 完整性已确认，但可加载性需在外部 GPU/较大内存环境验证。
- 权重下载完成不等于已完成 GT-free MTR 推理；仍需兼容 CUDA runtime、重新编译 MTR attention 扩展，以及 source-local ID exporter。

## 数据来源与边界

- 标准 OPV2V 的官方数据说明在 [OpenCOOD data introduction](https://github.com/DerrickXuNu/OpenCOOD/blob/main/docs/md_files/data_intro.md#opv2v)，原始场景包含按 agent/timestamp 组织的点云、YAML 和相机文件。
- 标准测试压缩包来自 UCLA Mobility Lab 的公开 Box 分享；本地只抽取指定场景的连续 ZIP 字节范围，未把 GT 写入 tracking ID、跨源关联或 prediction ledger。
- `vehicles` 等 GT 标注仍保留在 raw YAML，后续只能由独立离线 evaluator 读取。在线回放必须使用检测器候选、source-local AB3DMOT ID、测量/扰动 pose 和 receiver-side association。

## 下一步下载顺序

1. CPU-only GT-free replay 已完成；正式 artifact 在 [`data/intermediate/gtfree_cpu/opv2v_3scene`](../data/intermediate/gtfree_cpu/opv2v_3scene)，包含 source-local tracks、arrival-time association、ledger、baseline 和独立 evaluator。
2. 在外部 GPU/较大内存环境对三套完整 MTR 权重做 `torch.load` 和单场景 smoke inference；当前只完成文件大小与 ZIP CRC 检查。
3. 若 Gate -1 需要更多协作者数量，再从标准 test ZIP 按场景抽取 1 个 5--6 CAV held-out scene；当前 raw scene 覆盖 2/3 CAV，但 detector cache 覆盖 16 scenes。
4. GT-free exporter 跑通后，再接入 ledger；不再新增 association 优化或 reliability controller。
