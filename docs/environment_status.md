# RV-HCA 环境状态

更新时间：2026-09-04。当前目标是先跑通 GT-free 数据契约的 CPU 回放，不启动完整 reliability/regret controller。

## 可复用环境

- Python 环境：`/root/autodl-tmp/RV-HCA/.venv`
- 创建方式：`python -m venv --system-site-packages /root/autodl-tmp/RV-HCA/.venv`
- Torch：`2.0.0+cu118`，编译 CUDA 版本为 11.8
- 已验证导入：`easydict 1.13`、`glob2 0.7`、`timm 0.6.13`、`transformers 4.35.2`、`huggingface_hub 0.16.4`、`safetensors 0.4.0`、`terminaltables 3.1.0`、`open3d 0.18.0`、`spconv 2.3.6`、`scipy 1.10.1`、`filterpy 1.4.5`、`torch_geometric 2.6.1`。
- CMP/OpenCOOD 和 MTR 通过源码路径使用：

  ```bash
  export PYTHONPATH=/root/autodl-tmp/RV-HCA/vendor/CMP-upstream:/root/autodl-tmp/RV-HCA/vendor/CMP-upstream/MTR:/root/autodl-tmp/RV-HCA/vendor/CMP-upstream/AB3Dmot:/root/autodl-tmp/RV-HCA/vendor/CMP-upstream/AB3Dmot/Xinshuo_PyToolbox
  ```

  不要把 `AB3DMOT_libs` 子目录本身单独放到 `PYTHONPATH`；其中的 `io.py` 会遮蔽 Python 标准库 `io`。

## CPU 冒烟结果

使用 [GT-free detector cache](../data/intermediate/detections/opv2v_point_pillar_sinbevt_test_gtfree.pkl)，移除 `matched_car_id` 后按 CMP 的 box 转换送入 AB3DMOT。一个 178 帧 source 序列的前 20 帧成功回放：输入 279 个有效检测，输出 266 个 track rows，未使用 GT、GPU 或预计算 tracking cache。

目前 3 个已下载 raw scene 也已完成完整 CPU 回放和独立 evaluator；正式产物见 `data/intermediate/gtfree_cpu/opv2v_3scene`。这证明调用链可在 CPU 上运行，但是否足以支撑 reliability aggregation 仍由 association recall/track recall 门槛决定。

## GPU 结论

当前容器没有可用 GPU：`torch.cuda.is_available() = False`、`torch.cuda.device_count() = 0`，`nvidia-smi -L` 返回 `No devices found`。因此：

| 阶段 | 当前是否需要 GPU | 说明 |
|---|---:|---|
| 清洗 detector cache、AB3DMOT、坐标/时间对齐、receiver association、ledger、离线 association/trajectory 指标 | 否 | 这是当前 Gate -1，CPU 可做 |
| 从 raw PCD 重新运行 CMP CoBEVT/PointPillar 感知 | 是（实际需要 CUDA） | 现有感知 checkpoint 已下载，但本机不能推理 |
| MTR 预测推理/训练，编译 `mtr.ops.attention.attention_cuda` | 是（实际需要 CUDA） | 当前只完成环境导入；MTR CUDA extension 尚未构建 |
| 未来 RV-HCA reliability aggregation 训练 | 建议 GPU | 不是当前 GT-free 数据契约的前置条件 |

所以现在不需要为了 Gate -1 申请或安装 GPU；可以先用已清洗 cache 完成 tracking/association/ledger。若要声称“从原始点云端到端重跑 CMP”或进入 MTR/RV-HCA 训练，则必须换到暴露 CUDA 设备的机器。

## 已知非阻塞项

- `pip check` 只报告系统范围 `argoverse` 的可选依赖/旧版 numpy 约束；不影响当前 CMP detector-cache、AB3DMOT 和 ledger CPU 路径。
- 三套 MTR 权重已完整下载并通过 ZIP CRC；当前容器 `torch.load` 因约 2 GiB 内存上限被 OOM，需在外部 GPU/较大内存环境验证加载。
- 根文件系统剩余空间约 3.5 GB，`/root/autodl-tmp` 约 47 GB；暂不下载完整 21 GB OPV2V 包或更多 checkpoint。
