# DFlash Draft Attention：GPU → Ascend NPU 迁移

本目录将 `triton_gpu.py`（DFlash 论文 arXiv:2602.06036 的前向 draft attention，
GPU/CUDA 版 Triton kernel）迁移到华为 Ascend NPU（triton-ascend）。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `triton_gpu.py` | 迁移基线（GPU 版，未改动） |
| `triton_npu.py` | **NPU 版 kernel + host launcher**（自带 `@triton.autotune`） |
| `autotune_npu.py` | autotune 扫描脚本：多组典型 shape 触发配置搜索并测稳态耗时 |
| `test_correctness_npu.py` | 正确性对拍：triton kernel vs torch 参考实现（含 dummy block 分支） |

## 快速开始（需 CANN + torch_npu + triton-ascend 环境）

```bash
# 1) 正确性对拍（固定 BLOCK_N=64，速度快）
python my_kernels/dflash_attention/test_correctness_npu.py

# 2) 走 autotune 路径的对拍（首次触发配置搜索，较慢）
python my_kernels/dflash_attention/test_correctness_npu.py --autotune

# 3) autotune 扫描 + 稳态 benchmark
TRITON_PRINT_AUTOTUNING=1 python my_kernels/dflash_attention/autotune_npu.py

# 推荐配合的环境变量
export TRITON_PRINT_AUTOTUNING=1   # 打印最优配置
export TRITON_BENCH_METHOD=npu     # on-chip 计时（短 kernel 更准）
export TRITON_ALWAYS_COMPILE=1     # 禁用编译缓存
```

业务侧调用示例：

```python
from my_kernels.dflash_attention.triton_npu import dflash_attention_forward

o, lse = dflash_attention_forward(
    q, k, v, anchors, keep,
    ctx_len=ctx_len,          # K/V 前 ctx_len 行为 target-context KV
    scale=head_dim ** -0.5,
    block_size=16,            # draft block 大小
    head_chunks=1,            # GQA query-head 拆分份数（需整除 GROUPS）
)                             # 默认走 autotune；传 block_n=64 则固定分块直跑
```

## 数据布局约定（与 GPU 版一致）

- `q`: `(B, HQ, Q_LEN, HEAD_DIM)`，`Q_LEN >= NUM_ANCHORS * block_size`；
  第 j 个 anchor block 的 query 位于 `q[:, :, j*block_size:(j+1)*block_size, :]`。
- `k`/`v`: `(B, HK, KV_LEN, HEAD_DIM)`，`KV_LEN >= ctx_len + NUM_ANCHORS*block_size`；
  前 `ctx_len` 行为 target-context KV，其后按 anchor 顺序拼接各 draft block 的 KV。
- `anchors`: `(B, NUM_ANCHORS)` int32，`keep=True` 时需满足 `0 <= anchor <= ctx_len`
  （该 block 可见 context 前缀 `[0, anchor)`）；`keep=False`（dummy）的取值不影响结果。
- `keep`: `(B, NUM_ANCHORS)` bool/int8，True=真实 block（全 context + block 内双向），
  False=dummy/padding block（无 context、local tile 仅对角线，输出退化为自身 V）。

## 迁移要点（NPU 适配清单）

对照仓库 `AGENTS.md` 的实操清单逐条落实：

1. **算子类型**：CV 融合（`tl.dot` + softmax/mask 向量后处理）。
2. **grid 对齐物理核数**：GPU 2D grid `(NUM_ANCHORS, B*HK*HEAD_CHUNKS)` →
   NPU 1D grid = `num_aicore`，核内 `range(pid, total_blocks, num_cores)` 循环；
   任务分解（anchor/batch/kv_head/head_chunk 位次）与 GPU 版一一对应。
3. **UB 控制**：autotune 只搜 `BLOCK_N`；`_early_config_prune` 按 192KB UB 粗估过滤，
   编译期 `ub overflow` 由 autotuner 剔除失败配置 + ubtuner 兜底。
4. **dtype**：NPU Vector CMP 不支持 int32/int64——所有索引比较转 fp32（
   `offs_m.to(tl.float32) < ACTIVE_ROWS` 等），索引算术保持 int32。
5. **PIPELINE_STAGES**：GPU `tl.range(..., num_stages=)` → NPU 由编译选项
   `multibuffer`/`num_stages` 控制流水（在 autotune configs 中搜索）。
6. **BLOCK_M 折叠**：GPU 版 local tile 宽度 BLOCK_M 恒等于 BLOCK_SIZE，
   NPU 版直接使用 BLOCK_SIZE（语义不变，减少一个 constexpr）。
7. **autotune**：`import triton.backends.ascend.runtime` + `@triton.autotune` 直接包
   `@triton.jit` + 手写 `triton.Config`（含 Ascend 编译选项）+ `prune_configs_by`；
   `key=["B","NUM_ANCHORS","HEAD_DIM","BLOCK_SIZE","GROUPS","HEAD_CHUNKS"]` 缓存复用。
8. **正确性优先**：`test_correctness_npu.py` 与 torch 参考实现对拍
   （fp16 atol/rtol=1e-2，bf16=2e-2），覆盖 keep=True/False、GQA、多 batch。

## 待办 / 后续优化方向

- 有 NPU 环境后用 `msprof` 实测（`scalar_ratio`/`mte2_ratio`/`cube_ratio`），
  若 Vector 等待 Cube 可继续调 `enable_hivm_auto_cv_balance`、`tile_mix_vector_loop`；
- 更长上下文可考虑对 context 循环做 `BLOCK_N_SUB` 双层分块；
- 若 Q tile 的无效行（ACTIVE_ROWS < QUERY_ROWS 场景）成为常态，
  可给 `tl.load` 加 `care_padding=False` 消除补零依赖。
