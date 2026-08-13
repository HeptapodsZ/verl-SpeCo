# DFlashAttention Triton/TileLang Kernel 与 RTX 5080 Benchmark

## 1. 结论

本次实现和正式验收范围已按最终要求收敛为：`B=1`、`block_size=16`、
`context_len={512,2048,8192,16384,32768,65536}`、A=64、bf16、
`Hq/Hkv=32/8`、`D=128`。只在 RTX 5080（SM120、16 GB）上开发和测量；
H100/SM90 只保留配置注册入口，没有运行 smoke 或性能测试。

在上述六个 attention-core 形状上：

- Flex、SDPA、Triton、TileLang 共 24 个点全部通过 forward/backward 精度门槛；
- Triton 和 TileLang 共 12 个优化点全部通过性能验收；
- Triton 相对 SDPA 的 forward/backward 加速范围为 `5.08–10.37× / 4.37–5.97×`；
- TileLang 相对 SDPA 的 forward/backward 加速范围为 `4.62–7.43× / 3.75–7.04×`；
- 相对 Flex，Triton forward/backward latency 为 `0.74–0.86× / 0.61–0.72×`，
  TileLang 为 `0.89–1.03× / 0.52–0.78×`；所有 95% bootstrap CI 也满足
  `latency <= 1.10 × Flex`；
- 两个自定义 backend 的峰值 allocated 均不高于 SDPA。

`B=1, C=65536, block=16, hidden=4096` 的完整 `DFlashAttention` forward 和
backward 均可在 RTX 5080 上完成，四后端精度全部通过。Triton 完整模块
forward/backward 为 `24.80/57.09 ms`，TileLang 为 `25.70/60.69 ms`，
SDPA 为 `53.99/134.05 ms`，Flex 为 `26.20/68.16 ms`。

正式机器可读结果：

- [attention-core JSON](../benchmark_results/dflash_attention_rtx5080_b1_bs16_final.json)
- [attention-core CSV](../benchmark_results/dflash_attention_rtx5080_b1_bs16_final.csv)
- [完整模块 JSON](../benchmark_results/dflash_attention_module_b1_c65536_bs16_official.json)
- [完整模块 CSV](../benchmark_results/dflash_attention_module_b1_c65536_bs16_official.csv)
- [Nsight Compute 摘要 JSON](../benchmark_results/dflash_attention_nsight_b1_bs16_summary.json)
- [Nsight Compute 摘要 CSV](../benchmark_results/dflash_attention_nsight_b1_bs16_summary.csv)

## 2. 原始 DFlashAttention 分析

### What

`DFlashAttention` 的 Q 只来自 draft hidden states，K/V 来自
`context + draft`。Qwen3-8B 配置下 Q 有 32 个 head，K/V 有 8 个 head，
每个 head 维度为 128。每个 anchor 对应一个 16-token draft block：

- 有效 block 的每个 query 可见 `[0, anchor)` context 和本 block 全部 draft token；
- draft block 内是双向 attention，不是 causal attention；
- dummy block 的每个 query 只看自己的 draft key，保证 softmax finite；
- dropout 为 0，scale 为 `1/sqrt(D)`。

### How

原路径先分别投影 draft Q、context K/V、draft K/V，拼接 K/V，应用 RoPE 和
Q/K RMSNorm，再进入 FlexAttention 或 SDPA。Flex 使用稀疏 `BlockMask`；SDPA
使用 materialized dense boolean mask。新增路径把 `anchor_positions`、
`block_keep_mask` 和 `block_size` 作为结构化 metadata 直接传给 kernel，避免
构造 dense mask 或 Flex `BlockMask`。

执行流如下：

```text
draft_hidden -> q_proj -> Q --------------------------+
context_hidden -> k/v_proj -> K/V context             |
draft_hidden   -> k/v_proj -> K/V draft               |
                         concat + RoPE + Q/K norm      |
                                                        v
                   auto | flex | sdpa | triton | tilelang
                                                        |
                                                        v
                              transpose/reshape -> o_proj
```

### Why

SDPA 无法利用 DFlash mask 的规则性，dense mask 和通用 attention 会产生大量
无效 QK 工作及额外显存。Flex 能表达稀疏结构，但仍有 BlockMask 建立、通用
调度与 kernel 选择成本。DFlash 的可见集合可由 `(anchor, block_id, keep)` 唯一
确定，因此专用 kernel 能只遍历真正可见的 context prefix 和一个本地 block。

## 3. 集成接口

新增 `dflash_attention_backend = auto | flex | sdpa | triton | tilelang`：

- `auto` 保持原有语义：CUDA 且有 BlockMask 时走 Flex，否则走 SDPA；
- 显式 `triton/tilelang` 必须提供结构化 metadata，传入 dense/BlockMask 会报错；
- v1 自定义 kernel 只支持 CUDA、fp16/bf16、`D={64,128}`、
  `block_size=16`、`context_len<=65536`；不兼容输入 fail closed；
- SM120 有实测 tuning profile；SM90 注册表为空，作为 H100 后续扩展点。

主要实现文件：

- `verl_speco/models/dflash/modeling_dflash.py`
- `verl_speco/backends/dflash_trainer_backend.py`
- `verl_speco/models/dflash/kernels/dispatch.py`
- `verl_speco/models/dflash/kernels/tuning.py`
- `verl_speco/models/dflash/kernels/triton_attention.py`
- `verl_speco/models/dflash/kernels/tilelang_attention.py`

## 4. Kernel 设计

### 4.1 Forward

Triton 和 TileLang 都按 `(batch, anchor, KV head, Q-head group)` 分块。每个程序：

1. 从 anchor 0 流式遍历到 `anchor`，不读取 anchor 之后的 context；
2. 用 FP32 online softmax 更新 row max、row sum 和 accumulator；
3. 最后处理一个精确 16-token 的本地 draft tile；
4. 输出输入 dtype 的 O，同时保存 FP32 natural-log LSE；
5. 不保存 attention matrix。

GQA 通过一个程序处理多个 Q head 来复用 K/V。最终 RTX 5080 配置中 Triton
使用 2 heads/program；TileLang 在 `C<=512` 使用 2 heads/program，长 context
使用 4 heads/program。

dummy block 的 context online-softmax 状态保持空 identity，直到 self-only 本地
tile；这避免 `(-inf)-(-inf)` NaN。

### 4.2 Backward

Backward 不保存概率矩阵，而是根据 Q/K/V/LSE 重算：

1. `delta = sum(O * dO)`；
2. 重算 context prefix 和本地 block 概率，得到 dQ；
3. context dK/dV 按 key tile pull 所有能看到它的 query block，归约 GQA heads；
4. draft dK/dV 只处理自己的 16-token block；
5. dummy block 显式执行数学恒等式 `dQ=dK=0`、`dV=sum_GQA(dO_self)`。

Triton 始终使用无原子的 pull-based context dK/dV。TileLang 在
`C<=2048` 使用短窗 atomic schedule，在长窗使用无原子 pull schedule。所有
softmax、delta 和梯度归约均为 FP32，最终梯度转换回输入 dtype。

## 5. 精度验收

### 5.1 方法

精度分两层：

- 小形状使用独立 FP32 dense reference；它不调用 Flex 或 SDPA，并同时校验
  O、LSE、dQ、dK、dV；
- 65K 生产矩阵受 16 GB 显存限制，不 materialize 约 8.7 GB 的 FP32 score
  matrix；使用同一结构 mask 的 SDPA 结果作为可扩展 control，在 GPU 上计算
  error reductions，并保留小形状 FP32 reference 作为语义锚点。

bf16 门槛：forward `atol=rtol=2e-2`、relative-L2 `<=5e-3`、cosine
`>=0.9999`；backward `atol=rtol=3e-2`、relative-L2 `<=1e-2`、cosine
`>=0.999`。JSON 同时记录 `max_abs`、`mean_abs`、RMSE、relative-L2、cosine、
`allclose`、`mismatch_count` 和 NaN/Inf 数量。每个 core backend 重复三次；
正式矩阵的 repeat drift 为零。

完整模块从同一 state dict 克隆，并对齐 output、draft/context hidden gradient、
q/k/v/o projection weight gradient 和 q/k norm weight gradient。外部 `dO` 按
1024 个 draft token 做 mean-reduction，与训练 loss 的 token mean 语义一致。

测试还覆盖 fp16、bf16、D=64/128、GQA ratio=1/4、非整 context tile、重复
anchor、anchor=0/1/尾部、部分 dummy、全 dummy 和长窗 pull backward。最终
性能矩阵只使用 bf16、D=128、uniform anchor、seed=0。

### 5.2 Attention-core 汇总

下表是六个正式 context 形状上的最坏值；`mismatch` 为所有检查张量、所有形状
累计的 `isclose=False` 元素数。

| Backend | O max-abs | O max rel-L2 | O min cosine | grad max-abs | grad max rel-L2 | grad min cosine | mismatch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flex | 0.003906 | 0.001463 | 0.9999989 | 0.03125 | 0.004098 | 0.9999917 | 0 |
| SDPA | 0 | 0 | 0.9999999 | 0.000244 | 0.000011 | 0.9999999 | 0 |
| Triton | 0.003906 | 0.001020 | 0.9999995 | 0.015625 | 0.004103 | 0.9999916 | 0 |
| TileLang | 0.003906 | 0.001020 | 0.9999995 | 0.015625 | 0.004219 | 0.9999910 | 0 |

### 5.3 完整模块 65K 汇总

| Backend | O rel-L2 | O cosine | grad max rel-L2 | grad min cosine | grad max-abs | mismatch |
|---|---:|---:|---:|---:|---:|---:|
| Flex | 0.000778 | 0.9999997 | 0.005667 | 0.9999840 | 0.000122 | 0 |
| SDPA | 0 | 1.0000002 | 0.000163 | 0.9999999 | 0.000015 | 0 |
| Triton | 0.000587 | 0.9999999 | 0.005864 | 0.9999828 | 0.000122 | 0 |
| TileLang | 0.000587 | 0.9999999 | 0.005943 | 0.9999822 | 0.000122 | 0 |

## 6. Benchmark 方法与环境

环境：

- GPU：NVIDIA GeForce RTX 5080，SM120，约 15.92 GiB；
- Python 3.12.3；PyTorch 2.12.1+cu130；CUDA 13.0；
- Triton 3.7.1；TileLang 0.1.12；
- Nsight Systems 2025.3.2；Nsight Compute 2025.3.1；
- 基线 commit：`c9846b27604c15a2dc8dd7b4bfd083af76e7ef86`，结果包含未提交的本次改动。

Attention-core 每项稳定 warmup 后测 5 轮；每轮至少 100 次，并自动增加迭代数
使累计时间至少 2 秒。完整模块每轮至少 20 次或累计至少 2 秒。报告 p50/p90/p99；
首次 forward/backward 编译时间独立记录，不计入 steady-state latency。

首次编译时间范围：Flex forward/backward `120–1051 / 0.9–1941.7 ms`，SDPA
`0.9–38.1 / 2.8–100.8 ms`，Triton `8.5–371.4 / 8.0–1124.8 ms`，TileLang
`117.5–5218.6 / 147.8–8247.1 ms`。TileLang 冷编译明显较慢。

## 7. Attention-core 性能

`Q tok/s`、`QK pairs/s` 和 effective TFLOP/s 使用 forward+backward p50。
最后两列分别为 `SDPA speedup (F/B)` 和 `latency / Flex (F/B)`。完整 p90/p99、
reserved memory 和 95% CI 见正式 CSV/JSON。

| C | Backend | F p50 ms | B p50 ms | F+B p50 ms | Q tok/s | QK pairs/s | eff. TFLOP/s | peak alloc Δ MiB | vs SDPA F/B | /Flex F/B |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | Flex | 0.112 | 0.685 | 0.799 | 1281.3K | 11.15G | 17.13 | 22.2 | 7.63/3.63 | 1.00/1.00 |
| 512 | SDPA | 0.859 | 2.485 | 3.352 | 305.4K | 2.66G | 4.08 | 103.3 | 1.00/1.00 | 7.63/3.63 |
| 512 | Triton | 0.083 | 0.416 | 0.503 | 2037.4K | 17.73G | 27.24 | 23.3 | 10.37/5.97 | 0.74/0.61 |
| 512 | TileLang | 0.116 | 0.353 | 0.487 | 2101.5K | 18.29G | 28.10 | 51.3 | 7.43/7.04 | 1.03/0.52 |
| 2048 | Flex | 0.306 | 1.265 | 1.571 | 651.9K | 21.70G | 33.33 | 28.2 | 5.69/3.81 | 1.00/1.00 |
| 2048 | SDPA | 1.739 | 4.819 | 6.546 | 156.4K | 5.21G | 8.00 | 154.3 | 1.00/1.00 | 5.69/3.81 |
| 2048 | Triton | 0.246 | 0.910 | 1.157 | 885.2K | 29.46G | 45.25 | 28.3 | 7.07/5.30 | 0.80/0.72 |
| 2048 | TileLang | 0.271 | 0.856 | 1.127 | 908.9K | 30.25G | 46.46 | 68.3 | 6.41/5.63 | 0.89/0.68 |
| 8192 | Flex | 1.135 | 4.410 | 5.566 | 184.0K | 24.21G | 37.18 | 52.2 | 4.57/3.22 | 1.00/1.00 |
| 8192 | SDPA | 5.186 | 14.184 | 19.361 | 52.9K | 6.96G | 10.69 | 359.3 | 1.00/1.00 | 4.57/3.22 |
| 8192 | Triton | 0.897 | 2.930 | 3.841 | 266.6K | 35.08G | 53.89 | 52.3 | 5.78/4.84 | 0.79/0.66 |
| 8192 | TileLang | 1.057 | 3.449 | 4.540 | 225.6K | 29.68G | 45.59 | 140.3 | 4.91/4.11 | 0.93/0.78 |
| 16384 | Flex | 2.261 | 8.648 | 10.912 | 93.8K | 24.65G | 37.86 | 84.2 | 4.43/3.12 | 1.00/1.00 |
| 16384 | SDPA | 10.025 | 26.987 | 37.032 | 27.7K | 7.26G | 11.16 | 630.3 | 1.00/1.00 | 4.43/3.12 |
| 16384 | Triton | 1.766 | 5.611 | 7.392 | 138.5K | 36.38G | 55.89 | 84.3 | 5.68/4.81 | 0.78/0.65 |
| 16384 | TileLang | 2.088 | 6.593 | 8.694 | 117.8K | 30.94G | 47.52 | 236.3 | 4.80/4.09 | 0.92/0.76 |
| 32768 | Flex | 4.087 | 15.458 | 19.562 | 52.3K | 27.47G | 42.20 | 148.2 | 4.37/3.13 | 1.00/1.00 |
| 32768 | SDPA | 17.846 | 48.332 | 66.138 | 15.5K | 8.13G | 12.48 | 1174.3 | 1.00/1.00 | 4.37/3.13 |
| 32768 | Triton | 3.514 | 11.059 | 14.616 | 70.1K | 36.77G | 56.47 | 148.3 | 5.08/4.37 | 0.86/0.72 |
| 32768 | TileLang | 3.820 | 11.858 | 16.064 | 63.7K | 33.45G | 51.38 | 428.3 | 4.67/4.08 | 0.93/0.77 |
| 65536 | Flex | 8.172 | 33.432 | 42.494 | 24.1K | 25.28G | 38.83 | 276.2 | 4.66/2.92 | 1.00/1.00 |
| 65536 | SDPA | 38.094 | 97.568 | 139.414 | 7.3K | 7.71G | 11.84 | 2262.3 | 1.00/1.00 | 4.66/2.92 |
| 65536 | Triton | 7.014 | 21.988 | 29.035 | 35.3K | 37.00G | 56.83 | 276.3 | 5.43/4.44 | 0.86/0.66 |
| 65536 | TileLang | 8.246 | 26.002 | 34.234 | 29.9K | 31.38G | 48.20 | 812.3 | 4.62/3.75 | 1.01/0.78 |

## 8. 完整 DFlashAttention 65K

形状为 `B=1,C=65536,A=64,block=16,hidden=4096,Hq/Hkv=32/8,D=128`。
`F/B p50/p90/p99` 单位为 ms；throughput 是完整 forward+backward 的 draft
query tokens/s。

| Backend | F p50/p90/p99 | B p50/p90/p99 | F+B p50 | query tok/s | peak alloc MiB | peak Δ MiB | headroom | vs SDPA F/B | /Flex F/B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Flex | 26.20/26.28/26.29 | 68.16/68.17/68.17 | 94.54 | 10831 | 4684.7 | 1682.1 | 71.3% | 2.06/1.97 | 1.00/1.00 |
| SDPA | 53.99/54.05/54.06 | 134.05/134.18/134.19 | 187.93 | 5449 | 6389.0 | 2679.4 | 60.8% | 1.00/1.00 | 2.06/1.97 |
| Triton | 24.80/24.86/24.88 | 57.09/57.11/57.11 | 81.80 | 12518 | 4424.7 | 1682.1 | 72.9% | 2.18/2.35 | 0.95/0.84 |
| TileLang | 25.70/25.73/25.73 | 60.69/60.81/60.83 | 86.42 | 11849 | 4424.7 | 1682.1 | 72.9% | 2.10/2.21 | 0.98/0.89 |

独立 65K standalone smoke 也完成了完整模块 backward：Triton 和 TileLang
output shape 都是 `[1,1024,4096]`，输出 finite，单进程 peak allocated
`2467.15 MiB`、peak reserved `4740 MiB`。

## 9. Nsight Compute / Systems

### 9.1 Nsight Compute

NCU 使用 `--profile-from-start off`，由 `cudaProfilerStart/Stop` 只采正式迭代。
下表的 duration 是 NCU replay/profile 时间，不替代无 profiler 的正式 latency。

| Kernel | C | duration us | SM/tensor % | DRAM % | L2 throughput % | GB/s | L2 hit % | issue busy % | no eligible % | regs | dyn smem KiB | theor/ach occ % | branch eff % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Triton F | 512 | 97.92 | 27.38 | 16.13 | 39.01 | 152.53 | 85.50 | 17.83 | 79.47 | 236 | 45.06 | 16.67/16.01 | 100 |
| TileLang F | 512 | 125.09 | 20.74 | 20.24 | 34.44 | 191.44 | 89.21 | 8.07 | 89.48 | 242 | 40.96 | 8.33/7.99 | 100 |
| Triton F | 8192 | 1080 | 32.63 | 4.61 | 47.63 | 43.63 | 97.50 | 16.86 | 80.60 | 236 | 45.06 | 16.67/16.06 | 100 |
| TileLang F | 8192 | 1200 | 29.33 | 4.16 | 21.94 | 39.38 | 95.63 | 9.69 | 87.21 | 188 | 49.15 | 16.67/15.28 | 100 |
| Triton F | 65536 | 9760 | 33.22 | 5.41 | 48.91 | 44.07 | 97.89 | 16.85 | 80.65 | 236 | 45.06 | 16.67/16.06 | 100 |
| TileLang F | 65536 | 10010 | 29.87 | 5.11 | 22.30 | 45.14 | 94.23 | 9.78 | 87.25 | 188 | 49.15 | 16.67/15.26 | 100 |
| Triton context dKV | 65536 | 13630 | 45.54 | 5.83 | 19.33 | 49.56 | 91.74 | 9.62 | 90.24 | 255 | 10.24 | 16.67/16.65 | 100 |
| TileLang context dKV | 65536 | 15290 | 31.55 | 3.82 | 16.02 | 31.37 | 94.33 | 9.21 | 88.48 | 168 | 43.01 | 16.67/15.48 | 100 |

NCU roofline 规则显示 forward 的 Tensor pipeline 是最高利用 pipeline，但峰值仍低于
60%，主要限制是低 occupancy、寄存器/共享内存占用以及大量 no-eligible cycles。
长窗 Triton 相比 TileLang 有更高 issue-slot 利用率和更低 no-eligible 比例；这与其
长窗 forward/backward latency 优势一致。所有代表 kernel branch efficiency 为 100%。

### 9.2 Nsight Systems

`C=512` 的 Nsys 报告成功记录 CUDA API 和 NVTX，但当前 WSL2/Nsys 组合没有写入
CUDA kernel timeline；`cuda_gpu_kern_sum` 明确返回 “does not contain CUDA kernel
data”。因此不能从该报告可靠给出 kernel launch gap 或 GPU overlap，不能把 API
wall time冒充 GPU kernel time。

可用的 CUDA API 摘要：80 次 `cuLaunchKernelEx`，平均 `18.24 us`；20 次
`cudaLaunchKernel`，平均 `27.27 us`；3 次 `cudaDeviceSynchronize` 共
`2.43 ms`，占已记录 CUDA API 时间的 54.7%。底层 SM、memory、occupancy 和
warp stall 指标使用上面的 NCU 报告。若需要完整 Systems GPU timeline，应在原生
Linux 或支持 WSL CUDA trace 的更新驱动/Nsys 组合上重采。

## 10. 复现命令

所有 WSL 命令必须先激活指定环境：

```bash
source ~/.bashrc && source ~/venvs/cuda-triton/bin/activate
cd /mnt/d/Code_projects/VeRL/verl_speco/verl-SpeCo
```

运行单元与配置测试：

```bash
pytest -q tests/unit/test_dflash_attention_kernels.py -s
pytest -q tests/config/test_speco_config_overlay.py
```

运行六点正式 attention-core benchmark：

```bash
python benchmarks/dflash_attention/benchmark.py \
  --batch-sizes 1 \
  --context-lens 512,2048,8192,16384,32768,65536 \
  --block-sizes 16 \
  --backends flex,sdpa,triton,tilelang \
  --warmup 10 --rounds 5 --min-iterations 100 --min-seconds 2 \
  --output benchmark_results/dflash_attention_rtx5080_b1_bs16_final.json
```

只重跑精度门槛：

```bash
python benchmarks/dflash_attention/benchmark.py \
  --batch-sizes 1 \
  --context-lens 512,2048,8192,16384,32768,65536 \
  --block-sizes 16 --backends flex,sdpa,triton,tilelang \
  --accuracy-only \
  --output benchmark_results/dflash_attention_rtx5080_b1_bs16_accuracy.json
```

运行 65K 完整模块 benchmark 和 standalone smoke：

```bash
python benchmarks/dflash_attention/benchmark_module.py \
  --batch-sizes 1 --context-lens 65536 --block-sizes 16 \
  --backends flex,sdpa,triton,tilelang --hidden-size 4096 \
  --warmup 5 --rounds 5 --min-iterations 20 --min-seconds 2 \
  --output benchmark_results/dflash_attention_module_b1_c65536_bs16_official.json

python tests/special_standalone/dflash_attention_65k_gpu_smoke.py \
  --backend triton --context-len 65536 --block-size 16
python tests/special_standalone/dflash_attention_65k_gpu_smoke.py \
  --backend tilelang --context-len 65536 --block-size 16
```

运行 Nsight 和汇总 NCU：

```bash
BACKEND=triton bash benchmarks/dflash_attention/run_nsight.sh
BACKEND=tilelang bash benchmarks/dflash_attention/run_nsight.sh

python benchmarks/dflash_attention/summarize_ncu.py \
  benchmark_results/nsight/*.ncu-rep \
  --output benchmark_results/dflash_attention_nsight_b1_bs16_summary.json
```

## 11. 已知边界与下一步

- 结论只适用于本机 RTX 5080、B=1、block=16、bf16、D=128 和所列 context；
  不外推 H100、B>1、其他 block size 或完整 RLHF 训练吞吐。
- TileLang 的冷编译成本高于 Triton；长生命周期训练可摊销，但短进程任务应缓存。
- TileLang 长窗显存高于 Triton/Flex，但仍明显低于 SDPA；若继续优化，优先降低
  forward/backward workspace 和提高 eligible warps。
- Nsys 在当前 WSL 环境缺少 GPU kernel timeline，这是唯一未能在本机完整采集的
  Systems 底层项；NCU 原始报告保留在本地 `benchmark_results/nsight/`。
- 没有下载模型/数据集，没有启动完整 RLHF，也没有运行 H100 smoke。
