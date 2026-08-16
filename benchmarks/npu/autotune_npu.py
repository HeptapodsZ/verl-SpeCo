"""DFlash draft attention（NPU 版）autotune 脚本。

在已安装 CANN + torch_npu + triton-ascend 的 NPU 环境上运行，对一组典型
shape 逐一遍历，触发 ``@triton.autotune`` 的候选配置 benchmark 并打印最优配置，
随后测量稳态耗时。

用法
----
.. code-block:: bash

    # 基本用法（打印每次 autotune 的最优配置 + 稳态耗时）
    python my_kernels/dflash_attention/autotune_npu.py

    # 推荐配合的环境变量：
    #   TRITON_PRINT_AUTOTUNING=1    打印 autotune 选中的最优配置
    #   TRITON_BENCH_METHOD="npu"    使用 on-chip 计时（短 kernel 更准、更慢）
    #   TRITON_ALWAYS_COMPILE=1      禁用编译缓存、强制重编译
    export TRITON_PRINT_AUTOTUNING=1 TRITON_BENCH_METHOD=npu

    # 只跑单组 shape（方便快速复现）
    python my_kernels/dflash_attention/autotune_npu.py --case 1

注意事项
--------
- 首次调用每个 shape 都会触发一次完整的 autotune（并行编译 + benchmark），耗时
  较长属正常现象；结果按 ``key=["B","NUM_ANCHORS","HEAD_DIM","BLOCK_SIZE",
  "GROUPS","HEAD_CHUNKS"]`` 缓存，同 shape 后续调用直接复用。
- autotune 的候选配置定义在 ``triton_npu.py`` 的 ``_forward_kernel_npu`` 装饰器上，
  如需扩大搜索空间（如更多 BLOCK_N / CV balance 组合），直接修改该 configs 列表。
"""

import argparse
import time

import torch
import torch_npu  # noqa: F401

try:  # 兼容作为脚本直接运行与作为包运行
    from .triton_npu import best_config, dflash_attention_forward
except ImportError:
    from triton_npu import best_config, dflash_attention_forward

DEVICE = "npu"


def make_inputs(b, hq, hk, num_anchors, block_size, ctx_len, head_dim, dtype, seed=0):
    """构造一组 DFlash 训练态输入（含少量 dummy anchor 覆盖 keep=False 分支）。"""
    torch.manual_seed(seed)
    q_len = num_anchors * block_size
    kv_len = ctx_len + num_anchors * block_size
    q = torch.randn(b, hq, q_len, head_dim, dtype=dtype, device=DEVICE)
    k = torch.randn(b, hk, kv_len, head_dim, dtype=dtype, device=DEVICE)
    v = torch.randn(b, hk, kv_len, head_dim, dtype=dtype, device=DEVICE)

    # 随机 anchor 位置：分布在 [block_size, ctx_len] 内，保证至少能读到一段 context
    anchors = torch.randint(block_size, ctx_len + 1, (b, num_anchors), dtype=torch.int32, device=DEVICE)

    # 每 batch 末尾放 1~2 个 dummy block（模拟序列尾部 padding），其余 keep=True
    keep = torch.ones(b, num_anchors, dtype=torch.int8, device=DEVICE)
    if num_anchors >= 2:
        keep[:, -1] = 0
    if num_anchors >= 4:
        keep[:, -2] = 0

    scale = head_dim ** -0.5
    return q, k, v, anchors, keep, ctx_len, scale, block_size


def bench_ms(fn, warmup=10, rep=100):
    """稳态计时（毫秒）。autotune 已在首次调用中完成，此处只测执行耗时。"""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) / rep * 1e3


def run_case(idx, b, hq, hk, num_anchors, block_size, ctx_len, head_dim, dtype,
             head_chunks=1, rep=100, fixed_bn=64):
    print(f"\n===== case {idx}: B={b} HQ={hq} HK={hk} (GQA={hq // hk}) "
          f"NUM_ANCHORS={num_anchors} BLOCK_SIZE={block_size} CTX={ctx_len} "
          f"HEAD_DIM={head_dim} dtype={dtype} HEAD_CHUNKS={head_chunks} =====")

    q, k, v, anchors, keep, ctx_len, scale, block_size = make_inputs(
        b, hq, hk, num_anchors, block_size, ctx_len, head_dim, dtype
    )

    def call():
        return dflash_attention_forward(
            q, k, v, anchors, keep, ctx_len=ctx_len, scale=scale,
            block_size=block_size, head_chunks=head_chunks,
        )

    t0 = time.perf_counter()
    o, lse = call()  # 首次调用触发 autotune（编译 + benchmark）
    tune_s = time.perf_counter() - t0

    cfg = best_config()
    print(f"[tune] 首次调用(含 autotune) 耗时 {tune_s:.2f}s, best_config = {cfg}")

    ms = bench_ms(call, rep=rep)
    print(f"[bench] 稳态耗时 {ms:.4f} ms | O shape={tuple(o.shape)} LSE shape={tuple(lse.shape)}")

    # 与固定 BLOCK_N 的非 autotune 路径做个直观对比（可选参考）
    def call_fixed():
        return dflash_attention_forward(
            q, k, v, anchors, keep, ctx_len=ctx_len, scale=scale,
            block_size=block_size, head_chunks=head_chunks, block_n=fixed_bn,
        )

    call_fixed()  # 触发编译
    ms_fixed = bench_ms(call_fixed, rep=rep)
    print(f"[bench] 固定 BLOCK_N={fixed_bn} 稳态耗时 {ms_fixed:.4f} ms")
    return ms, ms_fixed


# 覆盖典型 DFlash drafter 配置（最后一项为固定路径的 BLOCK_N，按 UB 保守估算选取）：
#   (B, HQ, HK, NUM_ANCHORS, BLOCK_SIZE, CTX_LEN, HEAD_DIM, dtype, fixed_BLOCK_N)
SWEEP = [
    (1, 8, 8, 16, 16, 4096, 128, torch.float16, 64),    # MHA，短 block，中长上下文
    (1, 32, 8, 16, 16, 8192, 128, torch.bfloat16, 64),  # GQA=4，长上下文
    (2, 32, 8, 32, 16, 2048, 64, torch.float16, 64),    # batch=2，小 head_dim
    (1, 16, 4, 64, 32, 16384, 128, torch.float16, 32),  # 大 block_size、超长上下文
]


def main():
    parser = argparse.ArgumentParser(description="DFlash draft attention NPU autotune")
    parser.add_argument("--case", type=int, default=None, help="只跑指定 case（1 起）")
    parser.add_argument("--rep", type=int, default=100, help="稳态计时重复次数")
    parser.add_argument("--head-chunks", type=int, default=1, help="query head 拆分份数（需整除 GROUPS）")
    args = parser.parse_args()

    results = []
    for i, (b, hq, hk, na, bs, ctx, hd, dt, bn) in enumerate(SWEEP, start=1):
        if args.case is not None and i != args.case:
            continue
        results.append(
            (i, run_case(i, b, hq, hk, na, bs, ctx, hd, dt,
                         head_chunks=args.head_chunks, rep=args.rep, fixed_bn=bn))
        )

    print("\n===== 汇总 =====")
    for i, (ms, ms_fixed) in results:
        print(f"case {i}: autotune {ms:.4f} ms | 固定 BLOCK_N=64 {ms_fixed:.4f} ms")


if __name__ == "__main__":
    main()
