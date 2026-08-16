"""DFlash draft attention 前向 kernel：GPU(Triton/CUDA) -> Ascend NPU 迁移版。

迁移基线
--------
`my_kernels/dflash_attention/triton_gpu.py`（DFlash: Block Diffusion for Flash
Speculative Decoding, arXiv:2602.06036 的 draft 前向注意力）。

算法语义（与 GPU 版逐行等价）
----------------------------
每个 program 处理 "1 个 batch × 1 个 KV head × 1 个 query-head chunk × 1 个 anchor block"：

1. 查询 Q 块：``anchor_id * BLOCK_SIZE + [0, BLOCK_SIZE)`` 共 BLOCK_SIZE 个 draft token，
   乘上该 KV head 对应（GQA）的 HEADS_PER_PROGRAM 个 query head，拼成
   (QUERY_ROWS=HEADS_PER_PROGRAM*BLOCK_SIZE, HEAD_DIM) 的 Q tile。
2. Target-context KV 前缀循环：K/V 位置 [0, anchor)（anchor 为运行时标量，来自 ANCHORS），
   online-softmax 累积；keep=False 的 dummy block 看不到任何 context。
3. Local block tile：K/V 位置 [CTX_LEN + anchor_id*BLOCK_SIZE, +BLOCK_SIZE)，
   keep=True 时 block 内全双向；keep=False（dummy/padding block）时仅对角线
   （token_offset == offs_n_local，输出退化为自身的 V，保持 online-softmax 恒等状态）。
4. 写回 O 与 LSE = row_max + log(row_sum)。

迁移到 NPU 的改动
-----------------
- **grid**：GPU 用 2D grid (NUM_ANCHORS, B*HK*HEAD_CHUNKS)；NPU 改为 1D grid = 物理
  AI Core 数，核内 ``for block_idx in range(pid, total_blocks, num_cores)`` 循环，
  任务分解方式（batch/kv_head/head_chunk/anchor 的位次）与 GPU 版完全一致。
- **PIPELINE_STAGES**：GPU 的 ``tl.range(..., num_stages=PIPELINE_STAGES)`` 是 CUDA
  流水提示；NPU 改用 Ascend 编译选项 multibuffer/num_stages 控制流水（见 autotune
  configs）。上下文循环改为 ``range(0, anchor, BLOCK_N)``（运行时上界，不会被展开）。
- **BLOCK_M**：GPU 版 local tile 宽度单独用 BLOCK_M 表示，但实际调用中恒有
  BLOCK_M == BLOCK_SIZE（local tile 恰好覆盖一个 anchor block 的 BLOCK_SIZE 个 token，
  local_valid 掩码恒真），NPU 版直接合并为 BLOCK_SIZE，减少一个 constexpr。
- **dtype**：NPU Vector CMP 不支持 int32/int64，所有索引比较转 fp32 再做，避免
  退化为标量运算（索引算术仍保持 int32，int32 的 Vector Add/Mul 是支持的）。
- **exp2/log**：使用 ``tl.math.exp2/tl.math.log``（Ascend 后端已验证支持）。

UB 预算（A2 系列 UB = 192KB = 1,572,864 bits）由 ``_early_config_prune`` 粗过滤 +
autotuner 编译失败自动剔除 + ubtuner 兜底三层保证。

运行前提：已安装 CANN + torch_npu + triton-ascend（本机无 NPU 环境，请勿在无
NPU 环境下运行本文件）。
"""

import torch
import torch_npu  # noqa: F401  NPU 设备支持；缺失时本模块无法导入

from typing import Optional

import triton
import triton.language as tl

# 必须导入：进入 Triton-Ascend 的 autotune 扩展路径（configs 才会被 Ascend 侧处理）
import triton.backends.ascend.runtime  # noqa: F401
import triton.runtime.driver as driver

_LOG2E: float = 1.4426950408889634  # log2(e)，与 GPU 版一致使用 exp2 基数


def _early_config_prune(configs, named_args, **kwargs):
    """按 UB 容量粗估过滤明显会溢出的 BLOCK_N 候选。

    保守估计（fp16/bf16 输入，fp32 score/acc）：Q tile + acc(fp32) + scores/probs
    (fp32) + K/V（multibuffer 双缓冲计 ×2）。返回空时回退为原列表，最终由编译器
    ``ub overflow`` 报错 + autotuner 剔除失败配置 + ubtuner 兜底。
    """
    try:
        head_dim = int(kwargs.get("HEAD_DIM", named_args.get("HEAD_DIM", 128)))
        block_size = int(kwargs.get("BLOCK_SIZE", named_args.get("BLOCK_SIZE", 16)))
        groups = int(kwargs.get("GROUPS", named_args.get("GROUPS", 1)))
        head_chunks = int(kwargs.get("HEAD_CHUNKS", named_args.get("HEAD_CHUNKS", 1)))
        query_rows = (groups // max(head_chunks, 1)) * block_size
        dtype_bits = 16  # fp16 / bf16
        pruned = []
        for cfg in configs:
            block_n = int(cfg.kwargs.get("BLOCK_N", 64))
            bits_kv = 4 * block_n * head_dim * dtype_bits       # k + v，multibuffer ×2
            bits_q = query_rows * head_dim * dtype_bits
            bits_acc = query_rows * head_dim * 32               # fp32 accumulator
            bits_sp = 2 * query_rows * block_n * 32             # scores + probs (fp32)
            if bits_kv + bits_q + bits_acc + bits_sp <= 1_572_864:
                pruned.append(cfg)
        return pruned or configs
    except Exception:  # noqa: BLE001  形状信息缺失时不做裁剪，交给编译器/ubtuner
        return configs


@triton.autotune(
    configs=[
        # 基础候选：仅 BLOCK_N（上下文分块）参与 tiling 搜索，
        # 其余结构参数（B/HQ/HK/.../QUERY_ROWS）在 launch 时固定。
        triton.Config({"BLOCK_N": 32}),
        triton.Config({"BLOCK_N": 64}),
        triton.Config({"BLOCK_N": 128}),
        triton.Config({"BLOCK_N": 256}),
        # 短上下文场景：关 multibuffer（UB 更省、无流水收益时开销更小）
        triton.Config({"BLOCK_N": 64, "multibuffer": False}),
        triton.Config({"BLOCK_N": 128, "multibuffer": False}),
        # CV 融合调优变体：自动 CV balance + vector loop 切分（详见
        # docs/zh/migration_guide/architecture_difference.md 编译优化能力表）
        triton.Config({"BLOCK_N": 64, "enable_hivm_auto_cv_balance": True, "tile_mix_vector_loop": 2}),
        triton.Config({"BLOCK_N": 128, "enable_hivm_auto_cv_balance": True, "tile_mix_vector_loop": 4}),
    ],
    key=["B", "NUM_ANCHORS", "HEAD_DIM", "BLOCK_SIZE", "GROUPS", "HEAD_CHUNKS"],
    prune_configs_by={"early_config_prune": _early_config_prune},
)
@triton.jit
def _forward_kernel_npu(
    Q,
    K,
    V,
    ANCHORS,
    KEEP,
    O,
    LSE,
    B: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    CTX_LEN: tl.constexpr,
    NUM_ANCHORS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUPS: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    HEAD_CHUNKS: tl.constexpr,
    ACTIVE_ROWS: tl.constexpr,
    QUERY_ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(HEAD_DIM % 16 == 0)
    tl.static_assert(BLOCK_SIZE >= 16)

    # NPU 多核并行：grid = 物理 AI Core 数（含 tl.dot 的 CV 算子按 num_aicore 发射），
    # 核内按 stride=num_cores 循环处理所有逻辑任务；任务分解与 GPU 版 2D grid 一一对应：
    #   block_idx -> (anchor_id, batch_id, kv_head, head_chunk)
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)
    total_blocks = NUM_ANCHORS * (B * HK * HEAD_CHUNKS)
    for block_idx in range(pid, total_blocks, num_cores):
        anchor_id = block_idx // (B * HK * HEAD_CHUNKS)
        batch_kv_chunk = block_idx - anchor_id * (B * HK * HEAD_CHUNKS)
        batch_id = batch_kv_chunk // (HK * HEAD_CHUNKS)
        kv_chunk = batch_kv_chunk - batch_id * (HK * HEAD_CHUNKS)
        kv_head = kv_chunk // HEAD_CHUNKS
        head_chunk = kv_chunk - kv_head * HEAD_CHUNKS

        offs_m = tl.arange(0, QUERY_ROWS)
        offs_d = tl.arange(0, HEAD_DIM)
        head_lane = offs_m // BLOCK_SIZE
        token_offset = offs_m - head_lane * BLOCK_SIZE
        query_head = kv_head * GROUPS + head_chunk * HEADS_PER_PROGRAM + head_lane
        query_index = anchor_id * BLOCK_SIZE + token_offset

        # NPU Vector CMP 不支持 int32/int64：比较统一转 fp32，避免退化为标量运算。
        # （索引算术保持 int32：int32 的 Vector Add/Mul 是支持的，且 2^31 足够覆盖地址）
        offs_m_f = offs_m.to(tl.float32)
        query_head_f = query_head.to(tl.float32)
        kv_head_f = kv_head.to(tl.float32)
        query_valid = (offs_m_f < ACTIVE_ROWS) & (query_head_f < kv_head_f * GROUPS + GROUPS)

        q_ptrs = (
            Q
            + ((batch_id * HQ + query_head[:, None]) * Q_LEN + query_index[:, None])
            * HEAD_DIM
            + offs_d[None, :]
        )
        q = tl.load(q_ptrs, mask=query_valid[:, None], other=0.0)

        anchor = tl.load(ANCHORS + batch_id * NUM_ANCHORS + anchor_id)
        anchor_f = anchor.to(tl.float32)
        keep = tl.load(KEEP + batch_id * NUM_ANCHORS + anchor_id) != 0

        row_max = tl.where(query_valid, -float("inf"), 0.0).to(tl.float32)
        row_sum = tl.zeros((QUERY_ROWS,), tl.float32)
        acc = tl.zeros((QUERY_ROWS, HEAD_DIM), tl.float32)

        # Target-context KV 前缀循环：[0, anchor)（anchor 为运行时标量，循环不会展开）。
        # 流水由 Ascend 编译选项 multibuffer/num_stages 控制（见 autotune configs）。
        for start_n in range(0, anchor, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            key_valid = offs_n.to(tl.float32) < anchor_f
            k_ptrs = (
                K
                + ((batch_id * HK + kv_head) * KV_LEN + offs_n[:, None]) * HEAD_DIM
                + offs_d[None, :]
            )
            v_ptrs = (
                V
                + ((batch_id * HK + kv_head) * KV_LEN + offs_n[:, None]) * HEAD_DIM
                + offs_d[None, :]
            )
            k = tl.load(k_ptrs, mask=key_valid[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=key_valid[:, None], other=0.0)
            scores = tl.dot(q, tl.trans(k)) * SCALE
            # 阶梯状 target-prefix mask：整块 context 前缀可见，dummy block 不可见
            allowed = query_valid[:, None] & key_valid[None, :] & keep
            scores = tl.where(allowed, scores, -float("inf"))
            tile_max = tl.max(scores, axis=1)
            # dummy block 保持 online-softmax 恒等状态，直到本地的 self-only tile；
            # 否则 (-inf) - (-inf) 产生的 NaN 会穿透 tl.where（与 GPU 版一致）
            has_context = query_valid & keep
            new_max = tl.where(has_context, tl.maximum(row_max, tile_max), row_max)
            alpha = tl.where(
                has_context,
                tl.math.exp2((row_max - new_max) * _LOG2E),
                1.0,
            )
            probs = tl.where(
                allowed,
                tl.math.exp2((scores - new_max[:, None]) * _LOG2E),
                0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(probs.to(q.dtype), v)
            row_sum = row_sum * alpha + tl.sum(probs, axis=1)
            row_max = new_max

        # Local block tile（本 anchor 的 draft block：K/V 位置
        # [CTX_LEN + anchor_id*BLOCK_SIZE, +BLOCK_SIZE)），宽度恒为 BLOCK_SIZE
        #（GPU 版 BLOCK_M 恒等于 BLOCK_SIZE，故无 tail 掩码）。
        offs_n_local = tl.arange(0, BLOCK_SIZE)
        local_index = CTX_LEN + anchor_id * BLOCK_SIZE + offs_n_local
        k_ptrs = (
            K
            + ((batch_id * HK + kv_head) * KV_LEN + local_index[:, None]) * HEAD_DIM
            + offs_d[None, :]
        )
        v_ptrs = (
            V
            + ((batch_id * HK + kv_head) * KV_LEN + local_index[:, None]) * HEAD_DIM
            + offs_d[None, :]
        )
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        # block-diagonal bidirectional mask：keep=True 全双向；dummy block 仅对角线
        valid_rows_cols = query_valid[:, None]
        local_allowed = tl.where(
            keep,
            valid_rows_cols,
            valid_rows_cols
            & (token_offset.to(tl.float32)[:, None] == offs_n_local.to(tl.float32)[None, :]),
        )
        scores = tl.where(local_allowed, scores, -float("inf"))
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, tile_max)
        alpha = tl.math.exp2((row_max - new_max) * _LOG2E)
        probs = tl.math.exp2((scores - new_max[:, None]) * _LOG2E)
        acc = acc * alpha[:, None] + tl.dot(probs.to(q.dtype), v)
        row_sum = row_sum * alpha + tl.sum(probs, axis=1)
        row_max = new_max

        output = acc / row_sum[:, None]
        o_ptrs = (
            O
            + ((batch_id * HQ + query_head[:, None]) * Q_LEN + query_index[:, None])
            * HEAD_DIM
            + offs_d[None, :]
        )
        lse_ptrs = LSE + (batch_id * HQ + query_head) * Q_LEN + query_index
        tl.store(o_ptrs, output.to(q.dtype), mask=query_valid[:, None])
        tl.store(lse_ptrs, row_max + tl.math.log(row_sum), mask=query_valid)


_NUM_AICORE_CACHE = {}


def _num_aicore():
    """含 tl.dot 的 CV 融合算子按物理 AI Core 数发射 grid。"""
    dev = torch.npu.current_device()
    if dev not in _NUM_AICORE_CACHE:
        _NUM_AICORE_CACHE[dev] = driver.active.utils.get_device_properties(dev)["num_aicore"]
    return _NUM_AICORE_CACHE[dev]


def dflash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    anchors: torch.Tensor,
    keep: torch.Tensor,
    ctx_len: int,
    scale: float,
    block_size: int,
    head_chunks: int = 1,
    block_n: Optional[int] = None,
    multibuffer: Optional[bool] = None,
):
    """DFlash draft 前向注意力（NPU 版），与 GPU 版 ``_forward_kernel`` 语义一致。

    Args:
        q:       Query，(B, HQ, Q_LEN, HEAD_DIM)，fp16/bf16，Q_LEN >= NUM_ANCHORS*block_size
        k:       Key，(B, HK, KV_LEN, HEAD_DIM)，KV_LEN >= ctx_len + NUM_ANCHORS*block_size
                 布局：前 ctx_len 行为 target-context KV，其后按 anchor 顺序存放
                 NUM_ANCHORS 个 draft block 的 KV
        v:       Value，形状/dtype 同 k
        anchors: (B, NUM_ANCHORS) int32，第 j 个 anchor 在 target 序列中的位置，
                 该 block 可见 context 前缀 [0, anchors[b, j])。约定：keep=True 的
                 anchor 必须满足 0 <= anchor <= ctx_len（保证前缀落在 context KV 区
                 且不越界）；keep=False（dummy）的 anchor 取值不影响结果（被 mask 掉）。
                 launcher 不做 device->host 同步检查，以保证异步下发不被阻塞
        keep:    (B, NUM_ANCHORS) bool/int8，True=真实 anchor block（全 context +
                 block 内双向）；False=dummy/padding block（无 context、仅自身对角线）
        ctx_len: target-context KV 行数（K/V 前缀长度）
        scale:   QK 缩放系数（通常 1/sqrt(HEAD_DIM)）
        block_size: draft block 大小 K（论文中的 block size，典型 16）
        head_chunks: 将每个 KV head 对应的 GROUPS 个 query head 拆成几份并行
                     （需整除 GROUPS）
        block_n: 显式指定上下文分块宽度（绕过 autotune，直接编译执行）
        multibuffer: 显式指定多缓冲开关（默认 None=由 autotune 决定/默认开启）

    Returns:
        (o, lse)：o 与 q 同 dtype；lse 为 fp32 log-sum-exp，(B, HQ, Q_LEN)
    """
    if q.device.type != "npu":
        raise RuntimeError("dflash_attention_forward 仅支持 NPU 设备，请先安装 torch_npu 并在 NPU 上运行")

    b, hq, q_len, head_dim = q.shape
    _, hk, kv_len, _ = k.shape

    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
            and anchors.is_contiguous() and keep.is_contiguous()):
        raise ValueError("输入张量必须连续（contiguous）")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"仅支持 fp16/bf16 输入，当前 q.dtype={q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q/k/v 的 dtype 必须一致（tl.dot 要求同 dtype 输入）")
    if anchors.dtype != torch.int32:
        raise ValueError(f"anchors 必须为 int32，当前 {anchors.dtype}")
    if keep.dtype not in (torch.bool, torch.int8, torch.int32):
        raise ValueError(f"keep 需为 bool/int8/int32，当前 {keep.dtype}")
    if anchors.dim() != 2 or anchors.shape[0] != b:
        raise ValueError(f"anchors 形状需为 (B, NUM_ANCHORS)，当前 {anchors.shape}")
    if keep.shape != anchors.shape:
        raise ValueError(f"keep 与 anchors 形状需一致，当前 {keep.shape}")
    if hq % hk != 0:
        raise ValueError(f"HQ 必须能被 HK 整除（GQA），当前 HQ={hq}, HK={hk}")
    groups = hq // hk
    if groups % head_chunks != 0:
        raise ValueError(f"GROUPS={groups} 必须能被 head_chunks={head_chunks} 整除")
    heads_per_program = groups // head_chunks
    query_rows = heads_per_program * block_size

    for val, name in [(block_size, "block_size"), (head_dim, "HEAD_DIM"),
                      (query_rows, "QUERY_ROWS"), (head_chunks, "head_chunks")]:
        if val <= 0 or (val & (val - 1)) != 0:
            raise ValueError(f"{name}={val} 必须为 2 的幂（tl.arange 要求）")
    if head_dim % 16 != 0:
        raise ValueError(f"HEAD_DIM={head_dim} 必须为 16 的倍数（Cube K 维对齐）")

    num_anchors = anchors.shape[1]
    if q_len < num_anchors * block_size:
        raise ValueError(f"Q_LEN={q_len} 不足：需要 >= NUM_ANCHORS*block_size={num_anchors * block_size}")
    if kv_len < ctx_len + num_anchors * block_size:
        raise ValueError(f"KV_LEN={kv_len} 不足：需要 >= ctx_len + NUM_ANCHORS*block_size="
                         f"{ctx_len + num_anchors * block_size}")

    o = torch.empty_like(q)
    lse = torch.empty((b, hq, q_len), dtype=torch.float32, device=q.device)

    total_blocks = num_anchors * b * hk * head_chunks
    # grid 对齐物理 AI Core 数；任务数不足核数时按任务数发射，避免空跑
    grid = (min(_num_aicore(), total_blocks),)

    fixed = dict(
        B=b,
        HQ=hq,
        HK=hk,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        CTX_LEN=ctx_len,
        NUM_ANCHORS=num_anchors,
        SCALE=float(scale),
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        GROUPS=groups,
        HEADS_PER_PROGRAM=heads_per_program,
        HEAD_CHUNKS=head_chunks,
        ACTIVE_ROWS=query_rows,
        QUERY_ROWS=query_rows,
    )

    if block_n is not None:
        # 显式给定 BLOCK_N：绕过 autotune 直接编译执行（kernel.fn 为 @triton.jit 本体）
        launch_kwargs = dict(fixed, BLOCK_N=block_n)
        if multibuffer is not None:
            launch_kwargs["multibuffer"] = multibuffer
        _forward_kernel_npu.fn[grid](q, k, v, anchors, keep, o, lse, **launch_kwargs)
    else:
        # autotune 路径：BLOCK_N 由 configs 搜索，结果按 key 缓存复用
        launch_kwargs = dict(fixed)
        if multibuffer is not None:
            launch_kwargs["multibuffer"] = multibuffer
        _forward_kernel_npu[grid](q, k, v, anchors, keep, o, lse, **launch_kwargs)
    return o, lse


def best_config():
    """返回最近一次 autotune 选出的最优配置（未触发 autotune 时为 None）。"""
    return getattr(_forward_kernel_npu, "best_config", None)
