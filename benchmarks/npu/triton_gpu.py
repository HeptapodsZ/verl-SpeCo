import triton
import triton.language as tl


@triton.jit
def _forward_kernel(
    Q,
    K,
    V,
    ANCHORS,
    KEEP,
    O,
    LSE,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    CTX_LEN: tl.constexpr,
    NUM_ANCHORS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUPS: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    HEAD_CHUNKS: tl.constexpr,
    ACTIVE_ROWS: tl.constexpr,
    QUERY_ROWS: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    anchor_id = tl.program_id(0)
    batch_kv_chunk = tl.program_id(1)
    batch_id = batch_kv_chunk // (HK * HEAD_CHUNKS)
    kv_chunk = batch_kv_chunk - batch_id * HK * HEAD_CHUNKS
    kv_head = kv_chunk // HEAD_CHUNKS
    head_chunk = kv_chunk - kv_head * HEAD_CHUNKS

    offs_m = tl.arange(0, QUERY_ROWS)
    head_lane = offs_m // BLOCK_SIZE
    token_offset = offs_m - head_lane * BLOCK_SIZE
    query_head = kv_head * GROUPS + head_chunk * HEADS_PER_PROGRAM + head_lane
    offs_n_local = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    query_index = anchor_id * BLOCK_SIZE + token_offset
    query_valid = (offs_m < ACTIVE_ROWS) & (query_head < (kv_head + 1) * GROUPS)
    q_ptrs = (
        Q
        + ((batch_id * HQ + query_head[:, None]) * Q_LEN + query_index[:, None])
        * HEAD_DIM
        + offs_d[None, :]
    )
    q = tl.load(q_ptrs, mask=query_valid[:, None], other=0.0)
    anchor = tl.load(ANCHORS + batch_id * NUM_ANCHORS + anchor_id)
    keep = tl.load(KEEP + batch_id * NUM_ANCHORS + anchor_id) != 0

    row_max = tl.where(query_valid, -float("inf"), 0.0).to(tl.float32)
    row_sum = tl.zeros((QUERY_ROWS,), tl.float32)
    acc = tl.zeros((QUERY_ROWS, HEAD_DIM), tl.float32)

    # The loop bound is a runtime anchor, so 65K contexts do not get unrolled.
    for start_n in tl.range(0, anchor, BLOCK_N, num_stages=PIPELINE_STAGES):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        key_valid = offs_n < anchor
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
        allowed = query_valid[:, None] & key_valid[None, :] & keep
        scores = tl.where(allowed, scores, -float("inf"))
        tile_max = tl.max(scores, axis=1)
        # A dummy block cannot see context.  Keep its online-softmax state at
        # the empty identity until the finite self-only local tile is visited;
        # otherwise (-inf) - (-inf) produces NaNs that survive tl.where.
        has_context = query_valid & keep
        new_max = tl.where(has_context, tl.maximum(row_max, tile_max), row_max)
        alpha = tl.where(
            has_context,
            tl.exp2((row_max - new_max) * _LOG2E),
            1.0,
        )
        probs = tl.where(
            allowed,
            tl.exp2((scores - new_max[:, None]) * _LOG2E),
            0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(probs.to(Q.dtype.element_ty), v)
        row_sum = row_sum * alpha + tl.sum(probs, axis=1)
        row_max = new_max

    local_index = CTX_LEN + anchor_id * BLOCK_SIZE + offs_n_local
    local_valid = offs_n_local < BLOCK_SIZE
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
    k = tl.load(k_ptrs, mask=local_valid[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=local_valid[:, None], other=0.0)
    scores = tl.dot(q, tl.trans(k)) * SCALE
    valid_rows_cols = query_valid[:, None] & local_valid[None, :]
    local_allowed = tl.where(
        keep,
        valid_rows_cols,
        valid_rows_cols & (token_offset[:, None] == offs_n_local[None, :]),
    )
    scores = tl.where(local_allowed, scores, -float("inf"))
    tile_max = tl.max(scores, axis=1)
    new_max = tl.maximum(row_max, tile_max)
    alpha = tl.exp2((row_max - new_max) * _LOG2E)
    probs = tl.exp2((scores - new_max[:, None]) * _LOG2E)
    acc = acc * alpha[:, None] + tl.dot(probs.to(Q.dtype.element_ty), v)
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
    tl.store(o_ptrs, output, mask=query_valid[:, None])
    tl.store(lse_ptrs, row_max + tl.log(row_sum), mask=query_valid)