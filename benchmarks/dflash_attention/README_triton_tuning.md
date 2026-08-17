# DFlash Triton kernel parameter tuning

## 1. Implementation summary

本实现为 5 个已有 DFlash Triton forward 变体增加了 correctness-gated、离线的
kernel parameter tuning。生产调用仍是：

```python
dflash_sparse_attention(..., backend="triton[_two_anchor|_persistent|_one_grid|_one_fixed_grid]")
```

未改变 attention 数值公式、Tensor 输入输出、autograd contract 或现有 backend
名称。具体改动如下：

- `verl_speco/models/dflash/kernels/tuning.py`
  - 定义有界、可复现且经过剪枝的 standard/full search space；
  - 校验非法 `block_m/block_n/num_warps/num_stages/backward_schedule`；
  - 用 `DFlashTuningKey` 表达精确 workload；
  - 提供 context-local benchmark override；
  - 可从 `VERL_SPECO_DFLASH_TUNING_PROFILE` 指向的 tuner JSON 加载精确 shape
    profile；未设置环境变量或 key 不匹配时保持原 profile/fallback 行为。
- `verl_speco/models/dflash/kernels/triton_attention.py`
  - launch wrapper 将 B、context、anchor 数、Hq/Hkv、head dim、dtype、variant 和
    fixed-grid size 传给 tuning selector；
  - backward 保留原 forward variant，以区分 2-D 与 one-grid launch topology；
  - kernel implementation 本身未修改。
- `benchmarks/dflash_attention/tune_triton.py`
  - 复用已有 `Case/make_inputs/SDPA reference/accuracy metrics/measure_cuda`；
  - 搜索 forward 与 backward config，隔离 compile/OOM/accuracy failure；
  - 搜索结束后独立复测 baseline、forward-only、backward-only、combined finalist；
  - 输出可直接加载的 JSON profile、完整 trials 和 CSV summary。
- `benchmarks/dflash_attention/run_triton_tuning.sh`
  - 提供 `smoke/standard/full` 三档 WSL 复现入口。
- `tests/unit/test_dflash_kernel_tuning.py`
  - 覆盖空间大小、剪枝、config validation、override 恢复、profile 精确匹配和
    invalid profile fail-closed。
- `tests/unit/test_dflash_attention_kernels.py`
  - 用非默认 tuning 对 5 个 forward 变体及共享 backward 做 GPU 数值测试。

机器可读实测结果：

- `benchmark_results/dflash_triton_tuning_rtx5080_b1_c512_d128.json`
- `benchmark_results/dflash_triton_tuning_rtx5080_b1_c512_d128.csv`
- `benchmark_results/dflash_triton_tuning_rtx5080_b1_c8192_d128.json`
- `benchmark_results/dflash_triton_tuning_rtx5080_b1_c8192_d128.csv`

## 2. Tuning design

### 2.1 Parameters and candidate configs

`DFlashKernelTuning` 的字段及处理如下：

| Parameter | Standard space | Full space | 设计理由 |
|---|---|---|---|
| `block_m` | `16` | `16` | 当前 dispatch 只支持 `block_size=16`；更大的值会被 mask，但会确定性增加 local/dQ/draft-dKV 工作，提前剪枝。 |
| `block_n` | `32,64,128` | `16,32,64,128` | 控制 context K/V streaming tile，覆盖更多循环次数与更大 tile 的带宽/occupancy 权衡。 |
| `num_warps` | `4,8` | `2,4,8` | 覆盖并行度与 register/CTA residency 权衡。 |
| `num_stages` | `2` 为主体，另测 default `block_n` 的 `1,3` | `1,2,3` | 控制 software pipeline 深度，避免默认空间做完整笛卡尔积。 |
| `backward_schedule` | `pull` | `pull` | 当前 Triton backward 只有无原子 pull 实现，其他值直接拒绝。 |

Standard space 每个 phase 最多 8 个 config；`two_anchor` forward 最多 6 个。
Full space 是显式 opt-in，仍有以下静态剪枝：

- `two_anchor` 同时扩大 query/local tiles，forward 不搜索 `block_n=128`；
- `block_n=16,num_warps=8` 属于明显过度并行，跳过；
- `head_dim=128,block_n=128,num_stages=3` 是明显的 resource cliff，跳过；
- `context_len=0` 时 context tile/pipeline 参数没有作用，只保留默认 config；
- 非 power-of-two tile、未知 warp/stage 数、非 pull schedule 在编译前失败。

### 2.2 Tuning key

生成 profile 使用 exact-match key：

```text
(device_profile, forward_variant, batch_size, context_len, block_size,
 num_anchors, query_heads, kv_heads, head_dim, dtype, fixed_grid_size)
```

这里的 `head_dim` 是 kernel 的 per-head dimension。模型 hidden size 可由
`query_heads * head_dim` 表示；同时保留 Hq/Hkv，避免把同一 hidden size、不同
GQA layout 的结果错误复用。架构使用 CUDA compute capability（如 `sm120`）；
因此必须在每类目标 GPU 上分别运行，不会把 RTX 5080 结果当成 H100 结果。

### 2.3 Search and selection

每个 candidate 都执行一次完整 O/dQ/dK/dV correctness gate，通过后才计时。
forward 和 backward 分开搜索以控制规模；之后对最多 4 个唯一组合进行独立、
完整的 correctness 与 forward/backward/forward+backward 复测。对每个非默认
finalist，再逐 round 交错执行 `baseline->candidate / candidate->baseline`，避免 GPU
时钟、温度和后台调度造成 baseline-first 顺序偏差。最终必须同时满足 paired
`forward_backward p50 >= 1.01x` 和 bootstrap speedup CI95 下界 `>1.0` 才替换
baseline，否则生成的 profile 仍写入原配置。

Tuner 是离线工具，没有使用 `@triton.autotune`。原因是这里需要同时优化多个
forward/backward kernel、以 SDPA 梯度做 correctness gate，并把编译/搜索成本与
训练首步隔离。生产运行只做一次 profile JSON 解析和 exact-key lookup。

## 3. Correctness validation

已执行：

```bash
pytest -q \
  tests/unit/test_dflash_kernel_tuning.py \
  tests/unit/test_dflash_attention_tuning_benchmark.py
# 44 passed

pytest -q tests/unit/test_dflash_attention_kernels.py
# 15 passed
```

GPU tuned-config case 为 BF16、`B=1,C=33,A=3,block=16,Hq/Hkv=4/2,D=64`，
包含 dummy block，并覆盖 5 个 Triton forward 变体及两种 backward grid topology。
非默认 config 是 `(block_m=16, block_n=32, warps=4, stages=1, pull)`。
另用生成 JSON 作为环境 profile 在独立进程运行 `triton_one_grid` accuracy-only
benchmark，exact-key dispatch 命中后 `accuracy_valid=True`。

RTX 5080 tuning benchmark 使用 BF16、`B=1,C=512,A=64,block=16,Hq/Hkv=32/8,
D=128`。5 个变体的全部 standard candidates 均通过已有阈值：

- forward: `atol=rtol=2e-2`, relative L2 `<=5e-3`, cosine `>=0.9999`；
- backward: `atol=rtol=3e-2`, relative L2 `<=1e-2`, cosine `>=0.999`；
- final baseline/tuned 组合还重复执行两次，检查 repeat drift；
- candidate compile failure、OOM、NaN/Inf 或 accuracy failure 均不会进入选择集。

这证明了测试 shape 上的 kernel 数值与集成正确性；不是长序列、多 GPU 或完整
RLHF correctness/performance 结论。

## 4. Performance results

环境：NVIDIA GeForce RTX 5080（SM120，16 GB），PyTorch `2.12.1+cu130`，
CUDA `13.0`，Triton `3.7.1`。固定 seed 0，5 warmups，10 rounds，每轮至少 20 次，
并自动提高 iterations 使每个 measurement 目标累计约 0.5 s。编译不计入 latency。
JSON 同时记录 command、git dirty state 和 4 个关键源码文件的 SHA-256。

代表性 `B=1,A=64,Hq/Hkv=32/8,D=128,bf16` paired F+B 结果：

| Context | Variant | Baseline F+B ms | Selected F+B ms | Speedup / CI95 | Selected F / B `(BN,W,S)` |
|---:|---|---:|---:|---|---|
| 512 | baseline | 0.5417 | 0.5417 | 1.000x / `[1,1]` | `(64,4,2)` / `(64,4,2)` |
| 512 | two_anchor | 0.5515 | 0.5515 | 1.000x / `[1,1]` | `(32,4,2)` / `(64,4,2)` |
| 512 | persistent | 0.5400 | 0.5400 | 1.000x / `[1,1]` | `(64,4,2)` / `(64,4,2)` |
| 512 | one_grid | 0.5290 | 0.5290 | 1.000x / `[1,1]` | `(64,4,2)` / `(64,4,2)` |
| 512 | one_fixed_grid | 0.6395 | 0.6395 | 1.000x / `[1,1]` | `(64,4,2)` / `(64,4,2)` |
| 8192 | baseline | 4.0232 | 4.0232 | 1.000x / `[1,1]` | `(64,4,2)` / `(64,4,2)` |
| 8192 | two_anchor | 4.0017 | 4.0017 | 1.000x / `[1,1]` | `(32,4,2)` / `(64,4,2)` |
| 8192 | persistent | 4.0662 | 4.0662 | 1.000x / `[1,1]` | `(64,4,2)` / `(64,4,2)` |
| 8192 | one_grid | 3.6535 | 3.6535 | 1.000x / `[1,1]` | `(64,4,2)` / `(64,4,2)` |
| 8192 | one_fixed_grid | 4.9707 | 4.9707 | 1.000x / `[1,1]` | `(64,4,2)` / `(64,4,2)` |

结论是：这两个 shape 上没有 candidate 同时通过 1.01 point threshold 和 CI95
门槛，因此 production profile 正确保留原配置。最接近的 rejected finalists 是
`C=512 one_grid` 的 1.011x、CI95 `[0.974,1.055]`，以及
`C=8192 one_fixed_grid` 的 1.009x、CI95 `[0.976,1.017]`。这不是负面的工具结果：
它避免把顺序测量中一度出现的假阳性写入训练 profile。headline 也不是完整
DFlash module 或 RL step speedup。

## 5. Reproduction

脚本会自动进入仓库、激活项目指定 WSL 环境，并清除已有 profile，避免污染
baseline：

```bash
# 快速检查：B=1,C=512,D=64，5 variants
bash benchmarks/dflash_attention/run_triton_tuning.sh

# 受控矩阵：B=1,C={512,2048,8192},D={64,128}
DFLASH_TUNING_MODE=standard \
  bash benchmarks/dflash_attention/run_triton_tuning.sh

# 显式、较昂贵的 bounded full space；16 GB preflight 会跳过风险 shape
DFLASH_TUNING_MODE=full \
  bash benchmarks/dflash_attention/run_triton_tuning.sh
```

复现上表的精确命令：

```bash
source ~/.bashrc
source ~/venvs/cuda-triton/bin/activate
cd /mnt/d/Code_projects/VeRL/verl_speco/verl-SpeCo
unset VERL_SPECO_DFLASH_TUNING_PROFILE
python benchmarks/dflash_attention/tune_triton.py \
  --batch-sizes 1 \
  --context-lens 512 \
  --head-dims 128 \
  --num-anchors 64 \
  --query-heads 32 \
  --kv-heads 8 \
  --variants baseline,two_anchor,persistent,one_grid,one_fixed_grid \
  --warmup 5 \
  --rounds 10 \
  --min-iterations 20 \
  --min-seconds 0.5 \
  --min-speedup 1.01 \
  --output benchmark_results/dflash_triton_tuning_rtx5080_b1_c512_d128.json
```

显式应用生成 profile：

```bash
export VERL_SPECO_DFLASH_TUNING_PROFILE="$PWD/benchmark_results/dflash_triton_tuning_rtx5080_b1_c512_d128.json"
```

只有 exact key 命中时才使用 tuning 结果；其他 architecture/shape 自动回退原静态
profile。部署到训练前，应使用目标 workload 重跑 benchmark，并至少增加长序列、
目标 batch、FP16/BF16、目标 GQA layout 和完整 DFlash module validation。
