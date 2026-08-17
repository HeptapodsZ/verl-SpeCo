# `triton_one_fixed_grid` DFlash Attention

## 实现思路

`triton_one_fixed_grid` 是 SpeCo overlay 中显式 opt-in 的 Triton 后端。其
forward 主 kernel 的 `gridDim.x` 不再等于
`num_anchors * batch_kv_chunks`，而是由
`actor_rollout_ref.rollout.drafter.training.dflash_one_fixed_grid_size`
指定，默认值为 `40`。benchmark 可用 `--fixed-grid-size` 覆盖。

固定数量的 program 构成 worker 池：

```text
program_id = tl.program_id(0)
for work_id in range(program_id, total_work, fixed_grid_size):
    batch_kv_chunk = work_id % batch_kv_chunks
    anchor_id = work_id // batch_kv_chunks
    process(batch_kv_chunk, anchor_id)
```

以 batch/KV/head chunk 为逻辑最快维，使每个长期存活的 program 跨越完整
anchor 范围，缓解 anchor 越晚、可见 context 越长造成的尾部不均衡。每个
逻辑工作项仍独占对应的 Q/O/LSE 行，不需要原子操作、辅助队列或额外同步。
backward 复用 `triton_one_grid` 的一维 pull-reduction kernel，固定值只约束
任务所要求的 forward 主 launch。

非法的非正 grid size 会 fail closed。`auto`、无 drafter、SDPA、FlexAttention
及已有 Triton 后端的默认行为均未改变。

## RTX 5080 实验

形状为 `B=1, A=64, block_size=16, Lq=1024, Hq/Hkv=32/8, D=128`，
bf16，uniform anchors。每个 backend 在独立进程运行；latency 是 10 次
warmup 后 5 个 round 的 CUDA-event mean 的 p50，每个 phase 至少 20 次迭代
且累计目标至少 2 秒。Throughput 是有用 query token 吞吐 (`Kq/s`)；memory
是单个 phase 的 CUDA peak allocated (`MiB`)。

### Latency

| C | Phase | SDPA-40 ms | Grid 40 ms | Speedup | SDPA-80 ms | Grid 80 ms | Speedup |
|---:|---|---:|---:|---:|---:|---:|---:|
| 512 | Forward | 0.790 | 0.197 | 4.02x | 0.789 | 0.115 | 6.86x |
| 512 | F+B | 3.089 | 0.567 | 5.45x | 3.093 | 0.519 | 5.95x |
| 2,048 | Forward | 1.599 | 0.579 | 2.76x | 1.601 | 0.329 | 4.87x |
| 2,048 | F+B | 6.060 | 1.412 | 4.29x | 6.060 | 1.164 | 5.21x |
| 8,192 | Forward | 4.798 | 2.128 | 2.26x | 4.798 | 1.213 | 3.96x |
| 8,192 | F+B | 17.988 | 4.836 | 3.72x | 18.013 | 3.933 | 4.58x |
| 16,384 | Forward | 9.140 | 4.218 | 2.17x | 9.143 | 2.400 | 3.81x |
| 16,384 | F+B | 33.964 | 9.365 | 3.63x | 34.050 | 7.594 | 4.48x |
| 32,768 | Forward | 17.760 | 8.985 | 1.98x | 17.807 | 5.000 | 3.56x |
| 32,768 | F+B | 65.881 | 19.207 | 3.43x | 66.073 | 15.224 | 4.34x |
| 65,536 | Forward | 35.010 | 18.847 | 1.86x | 35.163 | 10.164 | 3.46x |
| 65,536 | F+B | 130.167 | 39.059 | 3.33x | 130.303 | 30.429 | 4.28x |

### Throughput

| C | Phase | SDPA-40 Kq/s | Grid 40 Kq/s | SDPA-80 Kq/s | Grid 80 Kq/s |
|---:|---|---:|---:|---:|---:|
| 512 | Forward | 1,296.2 | 5,211.0 | 1,297.3 | 8,897.9 |
| 512 | F+B | 331.4 | 1,806.1 | 331.1 | 1,971.7 |
| 2,048 | Forward | 640.5 | 1,768.3 | 639.5 | 3,115.7 |
| 2,048 | F+B | 169.0 | 725.0 | 169.0 | 879.7 |
| 8,192 | Forward | 213.4 | 481.3 | 213.4 | 844.5 |
| 8,192 | F+B | 56.9 | 211.7 | 56.8 | 260.4 |
| 16,384 | Forward | 112.0 | 242.8 | 112.0 | 426.6 |
| 16,384 | F+B | 30.1 | 109.3 | 30.1 | 134.8 |
| 32,768 | Forward | 57.7 | 114.0 | 57.5 | 204.8 |
| 32,768 | F+B | 15.5 | 53.3 | 15.5 | 67.3 |
| 65,536 | Forward | 29.2 | 54.3 | 29.1 | 100.7 |
| 65,536 | F+B | 7.9 | 26.2 | 7.9 | 33.7 |

### Peak allocated memory

Grid 40 与 80 的峰值一致，因为调度变化没有新增 buffer。

| C | Phase | SDPA MiB | Grid 40 MiB | Grid 80 MiB |
|---:|---|---:|---:|---:|
| 512 | Forward | 58.5 | 30.1 | 30.1 |
| 512 | F+B | 126.8 | 45.3 | 45.3 |
| 2,048 | Forward | 94.0 | 36.1 | 36.1 |
| 2,048 | F+B | 186.3 | 56.3 | 56.3 |
| 8,192 | Forward | 232.0 | 60.1 | 60.1 |
| 8,192 | F+B | 420.3 | 104.3 | 104.3 |
| 16,384 | Forward | 416.0 | 92.1 | 92.1 |
| 16,384 | F+B | 732.3 | 168.3 | 168.3 |
| 32,768 | Forward | 784.0 | 156.1 | 156.1 |
| 32,768 | F+B | 1,356.3 | 296.3 | 296.3 |
| 65,536 | Forward | 1,520.0 | 284.1 | 284.1 |
| 65,536 | F+B | 2,604.3 | 552.3 | 552.3 |

全部 24 个 latency 对照点均快于 SDPA。grid 40 的最低加速为 forward
`1.86x`，grid 80 为 `3.46x`。六个 context 的 accuracy-only 矩阵也全部
通过 SDPA gate；最坏 relative L2 为 output `0.001020`、dQ `0.003555`、
dK `0.004091`、dV `0.002725`。

原始结果位于 `benchmark_results/dflash_attention_fixed_grid_{40,80}_compare_rtx5080.{json,csv}`
和 `benchmark_results/dflash_attention_fixed_grid_{40,80}_accuracy_rtx5080.{json,csv}`。
这些是 attention-core 单卡结果，不代表完整 drafter training 或端到端 RLHF
吞吐；换 GPU、shape、dtype 或 anchor 分布后需要重新测量。
