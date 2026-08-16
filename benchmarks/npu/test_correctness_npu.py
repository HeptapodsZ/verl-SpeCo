"""DFlash draft attention（NPU 版）正确性对拍：triton kernel vs torch 参考实现。

在已安装 CANN + torch_npu + triton-ascend 的 NPU 环境上运行。参考实现按
DFlash 稀疏 mask 定义直接构造：

    A_ctx  [(j,u), c] = 1[c < anchor_j] * keep_j        （阶梯状 target-context 前缀）
    A_block[(j,u),(k,v)] = 1[j==k] * (keep_j ? 1 : 1[u==v])  （block-diagonal 双向；
                                                            dummy block 仅对角线）

用法
----
.. code-block:: bash

    # 固定 BLOCK_N=64 快速对拍（默认，跳过 autotune 的 benchmark 开销）
    python my_kernels/dflash_attention/test_correctness_npu.py

    # 走 autotune 路径（首次会触发配置搜索，较慢）
    python my_kernels/dflash_attention/test_correctness_npu.py --autotune

    # 只跑指定 case
    python my_kernels/dflash_attention/test_correctness_npu.py --case 1
"""

import argparse

import torch
import torch_npu  # noqa: F401

try:  # 兼容作为脚本直接运行与作为包运行
    from .triton_npu import dflash_attention_forward
except ImportError:
    from triton_npu import dflash_attention_forward

DEVICE = "npu"
_LOG2E = 1.4426950408889634

# (B, HQ, HK, NUM_ANCHORS, BLOCK_SIZE, CTX_LEN, HEAD_DIM, dtype, 固定路径 BLOCK_N)
# BLOCK_N 按 UB 保守估算选取：q(QUERY_ROWS*HD) + acc(fp32) + k/v(×2) + scores/probs
CASES = [
    (1, 8, 8, 4, 16, 256, 64, torch.float16, 64),
    (1, 8, 8, 4, 16, 256, 64, torch.bfloat16, 64),
    (1, 32, 8, 8, 16, 512, 128, torch.float16, 64),    # GQA=4
    (2, 32, 8, 16, 16, 1024, 128, torch.float16, 64),  # batch=2 + dummy blocks
    (1, 16, 4, 16, 32, 1024, 128, torch.bfloat16, 32),  # block_size=32、QUERY_ROWS=128
]


def ref_forward(q, k, v, anchors, keep, ctx_len, scale, block_size):
    """torch 参考实现：与 kernel 相同的 exp2-based online-softmax 逐 block 计算。"""
    b, hq, q_len, head_dim = q.shape
    _, hk, kv_len, _ = k.shape
    groups = hq // hk
    num_anchors = anchors.shape[1]

    qf, kf, vf = q.float(), k.float(), v.float()
    o = torch.zeros(b, hq, q_len, head_dim, dtype=torch.float32, device=q.device)
    lse = torch.zeros(b, hq, q_len, dtype=torch.float32, device=q.device)

    for bb in range(b):
        for h in range(hq):
            kh = h // groups
            for j in range(num_anchors):
                anchor = int(anchors[bb, j])
                kept = bool(keep[bb, j])
                assert 0 <= anchor <= ctx_len
                qs = qf[bb, h, j * block_size:(j + 1) * block_size]

                m = torch.full((block_size,), float("-inf"), dtype=torch.float32, device=q.device)
                l = torch.zeros(block_size, dtype=torch.float32, device=q.device)
                acc = torch.zeros(block_size, head_dim, dtype=torch.float32, device=q.device)

                # ---- target-context 前缀 [0, anchor)（仅 keep 的 block 可见）----
                if kept and anchor > 0:
                    sc = qs @ kf[bb, kh, :anchor].T * scale          # (BS, anchor)
                    tile_max = sc.max(dim=1).values
                    new_m = torch.maximum(m, tile_max)
                    alpha = torch.exp2((m - new_m) * _LOG2E)         # m=-inf -> 0
                    p = torch.exp2((sc - new_m[:, None]) * _LOG2E)
                    acc = acc * alpha[:, None] + p @ vf[bb, kh, :anchor]
                    l = l * alpha + p.sum(dim=1)
                    m = new_m

                # ---- local block tile（block-diagonal 双向；dummy 仅对角线）----
                lo = ctx_len + j * block_size
                sl = qs @ kf[bb, kh, lo:lo + block_size].T * scale   # (BS, BS)
                mask = torch.ones(block_size, block_size, dtype=torch.bool, device=q.device) if kept \
                    else torch.eye(block_size, dtype=torch.bool, device=q.device)
                sl = sl.masked_fill(~mask, float("-inf"))
                tile_max = sl.max(dim=1).values
                new_m = torch.maximum(m, tile_max)
                alpha = torch.exp2((m - new_m) * _LOG2E)
                p = torch.exp2((sl - new_m[:, None]) * _LOG2E) * mask
                acc = acc * alpha[:, None] + p @ vf[bb, kh, lo:lo + block_size]
                l = l * alpha + p.sum(dim=1)
                m = new_m

                o[bb, h, j * block_size:(j + 1) * block_size] = acc / l[:, None]
                lse[bb, h, j * block_size:(j + 1) * block_size] = m + torch.log(l)
    return o, lse


def make_inputs(b, hq, hk, num_anchors, block_size, ctx_len, head_dim, dtype, with_dummy, seed=0):
    torch.manual_seed(seed)
    q_len = num_anchors * block_size
    kv_len = ctx_len + num_anchors * block_size
    q = torch.randn(b, hq, q_len, head_dim, dtype=dtype, device=DEVICE) * 0.8
    k = torch.randn(b, hk, kv_len, head_dim, dtype=dtype, device=DEVICE) * 0.8
    v = torch.randn(b, hk, kv_len, head_dim, dtype=dtype, device=DEVICE) * 0.8

    anchors = torch.randint(1, ctx_len + 1, (b, num_anchors), dtype=torch.int32, device=DEVICE)

    keep = torch.ones(b, num_anchors, dtype=torch.int8, device=DEVICE)
    if with_dummy:
        keep[:, -1] = 0          # dummy block：验证 keep=False 的自注意力对角线分支
        keep[:, -2] = 0

    scale = head_dim ** -0.5
    return q, k, v, anchors, keep, scale


def run_case(idx, b, hq, hk, num_anchors, block_size, ctx_len, head_dim, dtype,
             with_dummy, use_autotune, head_chunks=1, block_n=64):
    print(f"\n===== case {idx}: B={b} HQ={hq} HK={hk} (GQA={hq // hk}) "
          f"NUM_ANCHORS={num_anchors} BLOCK_SIZE={block_size} CTX={ctx_len} "
          f"HEAD_DIM={head_dim} dtype={dtype} dummy={with_dummy} "
          f"head_chunks={head_chunks} autotune={use_autotune} =====")

    q, k, v, anchors, keep, scale = make_inputs(
        b, hq, hk, num_anchors, block_size, ctx_len, head_dim, dtype, with_dummy
    )

    o_ref, lse_ref = ref_forward(q, k, v, anchors, keep, ctx_len, scale, block_size)

    if use_autotune:
        o, lse = dflash_attention_forward(
            q, k, v, anchors, keep, ctx_len=ctx_len, scale=scale,
            block_size=block_size, head_chunks=head_chunks,
        )
    else:
        o, lse = dflash_attention_forward(
            q, k, v, anchors, keep, ctx_len=ctx_len, scale=scale,
            block_size=block_size, head_chunks=head_chunks, block_n=block_n,
        )

    tol = 1e-2 if dtype == torch.float16 else 2e-2  # bf16 尾数更短，放宽
    torch.testing.assert_close(o.float(), o_ref, atol=tol, rtol=tol,
                               msg=f"case {idx}: O 对拍失败")
    torch.testing.assert_close(lse, lse_ref, atol=tol, rtol=tol,
                               msg=f"case {idx}: LSE 对拍失败")
    print(f"[PASSED] max|O-Ref|={ (o.float() - o_ref).abs().max().item():.3e}  "
          f"max|LSE-Ref|={(lse - lse_ref).abs().max().item():.3e}")


def test_correctness_fixed_block_n():
    run_case(0, 1, 8, 8, 4, 16, 256, 64, torch.float16, with_dummy=True, use_autotune=False)


def main():
    parser = argparse.ArgumentParser(description="DFlash draft attention NPU 正确性对拍")
    parser.add_argument("--autotune", action="store_true", help="走 autotune 路径（首次较慢）")
    parser.add_argument("--case", type=int, default=None, help="只跑指定 case（1 起）")
    parser.add_argument("--head-chunks", type=int, default=1, help="query head 拆分份数（需整除 GROUPS）")
    args = parser.parse_args()

    for i, (b, hq, hk, na, bs, ctx, hd, dt, bn) in enumerate(CASES, start=1):
        if args.case is not None and i != args.case:
            continue
        with_dummy = (i >= 4) or (args.case is not None)  # 覆盖 keep=False 分支
        run_case(i, b, hq, hk, na, bs, ctx, hd, dt, with_dummy=with_dummy,
                 use_autotune=args.autotune, head_chunks=args.head_chunks, block_n=bn)

    print("\n全部对拍通过。")


if __name__ == "__main__":
    main()
