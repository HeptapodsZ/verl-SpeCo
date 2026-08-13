# `run_qwen3-8b_drafter_dflash_vllm.sh` 的 Drafter Training 参数说明

## 1. 本文回答的问题

本文解释脚本 `verl-SpeCo/examples/run_qwen3-8b_drafter_dflash_vllm.sh` 中显式设置的全部 19 个
`actor_rollout_ref.rollout.drafter.training.*` 参数，包括：

- 参数控制什么；
- 参数在当前 SpeCo 实现中怎样生效；
- 为什么该示例这样配置；
- 参数之间的联动、实际边界与容易误解之处。

本文描述的是 **SpeCo overlay 行为**，不是 upstream VeRL v0.8.0 的原生功能。Rollout 权重热更新部分则通过
SpeCo 对 **vLLM rollout runtime** 的适配实现。

## 2. 一句话心智模型

这份脚本每隔 5 个 RL trainer global step，复用 actor 的 old-logprob 前向计算采集一批目标模型 hidden states，
随后对 DFlash drafter 尝试执行 10 个优化器 step；只要训练成功，就把 drafter 权重转为 CPU BF16 snapshot，
通过 512 MiB 分桶异步发布给 vLLM，在更新期间暂停生成，并在更新后清空 KV cache。

需要特别注意：

1. `step=10` 是每次触发时尝试执行的 **drafter optimizer step 数**，不是 RL step 数。
2. `dflash_max_window=65536` 不是本脚本实际采集 65K hidden-state token 的保证。old-logprob 采集仍继承
   `hidden_state_window_tokens_per_sample=512`，因此默认计划是每个入选样本采集 513 行 hidden states，
   用于形成 512 个训练行。
3. `dflash_hard_sample_ratio=0.3` 在当前 old-logprob 样本构造路径上通常退化为全随机抽样，因为该路径没有把
   hard score、sample loss 或 speculative acceptance 字段写进 drafter sample。
4. 脚本没有设置 `publish_interval_steps`；它继承默认值 `0`，含义是每次产生成功训练结果时都发布。

## 3. 实际执行与数据流

在满足 `global_step > 0` 且 `global_step % 5 == 0` 的 RL step 上，执行流如下：

1. **生成阶段**：vLLM 使用当前 DFlash drafter 做 speculative rollout，但不从 SGLang rollout 路径采集 hidden states。
2. **old-logprob 阶段**：actor worker 计算 old log probability；SpeCo 在同一次目标模型前向中用 forward hooks
   捕获 DFlash 所需的目标层 hidden states，只保留采集计划选中的连续位置。
3. **样本分发**：采集结果连同 token ids、位置和布局元数据，通过 Ray/object refs 分发给对应 drafter training worker。
4. **训练触发**：SpeCo 在 actor update 周围挂接 drafter 流程。它先同步与这些 hidden states 对应的 target LM head，
   再执行 actor update，之后尝试 10 次 drafter training step。
5. **DFlash loss**：每个 training step 默认最多取 4 个样本；每条样本均匀无放回采最多 64 个有效 anchor，
   每个 anchor 使用默认 16-token block，offset 0 是已知 anchor token且不计损失，offset 1..15 为预测目标。
6. **发布**：若至少一个 drafter optimizer step 成功，默认立即准备 snapshot，转为 BF16 CPU contiguous tensors，
   再通过 vLLM bucketed weight-transfer 路径更新 rollout drafter。
7. **一致性边界**：异步发布可以和当前 RL step 的后续主机端工作重叠，但下一次 `generate_sequences` 会等待尚未完成的发布，
   因此下一轮 rollout 不会读取一半新、一半旧的 drafter 权重。

## 4. 参数总览

| 参数后缀 | 脚本值 | 直接作用 |
|---|---:|---|
| `collect_hidden_states_from_sgl` | `False` | 禁用 SGLang rollout hidden-state 采集路径 |
| `collect_hidden_states_from_old_logprob` | `True` | 在 actor old-logprob 前向中采集监督特征 |
| `old_logprob_hidden_capture_impl` | `forward_hook` | 用目标层 forward hooks 捕获所需 hidden states |
| `dflash_num_anchors` | `64` | 每条样本最多采样 64 个 DFlash block anchor |
| `dflash_max_window` | `65536` | DFlash 后端对已经收集的单样本序列进行二次裁剪的上限 |
| `dflash_loss_mode` | `restricted_ce` | 只在当前训练 batch 出现过的 token 子词表上计算 CE |
| `dflash_loss_decay_gamma` | `7` | block 内较远预测位置的指数衰减尺度 |
| `dflash_front_position_weight` | `2.0` | 对 block 前部预测位置额外乘 2 倍权重 |
| `dflash_front_position_count` | `3` | 额外加权 offset 1、2、3 三个位置 |
| `dflash_hard_sample_ratio` | `0.3` | 计划让 batch 的约 30% 来自困难样本池 |
| `step` | `10` | 每次训练触发尝试执行 10 个 optimizer steps |
| `collect_interval_steps` | `5` | 每 5 个正整数 RL global steps 才允许采集 |
| `training_interval_steps` | `5` | 每 5 个正整数 RL global steps 才允许训练 |
| `publish_async` | `True` | 不在发起权重更新 RPC 后立刻阻塞 driver |
| `publish_dtype` | `bf16` | 发布 snapshot 转成 CPU BF16 |
| `draft_update_weights_bucket_megabytes` | `512` | vLLM 权重传输分桶大小为 512 MiB |
| `draft_update_pause_generation` | `True` | 更新前 abort/pause，finally 中恢复生成 |
| `draft_update_flush_before` | `False` | 更新前不要求重置 prefix/KV cache |
| `draft_update_flush_after` | `True` | 更新成功后清空 KV cache |

## 5. 逐组说明：What / How / Why

### 5.1 监督数据来源

#### `collect_hidden_states_from_sgl=False`

- **What**：关闭从 SGLang rollout runtime 直接收集 hidden states 的路径。
- **How**：本参数与 `collect_hidden_states_from_old_logprob=True` 组合后，监督特征只来自 actor old-logprob 前向。
  当前实现明确禁止同时开启这两条路径。
- **Why**：该脚本的 rollout runtime 是 vLLM，不是 SGLang；使用 old-logprob 路径还能复用 RL 本来就要执行的
  old-logprob 计算，避免再为教师 hidden states 单独跑一次完整 target forward。

虽然参数名带有 `sgl`，它不是“是否从任意 rollout engine 采集”的通用开关，而是明确指向 SGLang 路径。

#### `collect_hidden_states_from_old_logprob=True`

- **What**：把 actor old-logprob 前向同时用作 DFlash 的教师特征采集前向。
- **How**：只有 collect interval 和 training interval 同时命中时才创建采集计划。当前默认还会应用以下未在脚本中覆盖的限制：
  `collection_sample_rate=1.0`、每个 replica 每个 step 最多 16 个样本、最多 16384 个采集行、
  `hidden_state_window_mode=front`、每个样本 512 个训练行。
- **Why**：复用已有 actor forward 降低额外计算成本，同时让 hidden states 与该 RL step 使用的 actor/target head 版本一致。

当前实现的兼容性门槛是 `actor.strategy` 必须为 `fsdp` 或 `fsdp2`，且 `use_logits=False`。脚本设置了
`actor.strategy=fsdp2`，并继承 `use_logits=false`，因而满足约束。

#### `old_logprob_hidden_capture_impl=forward_hook`

- **What**：选择通过 forward hooks 捕获指定 transformer layer/final norm 的输出。
- **How**：SpeCo 只给 DFlash 所需模块注册 hook，并在前向后拼接所需层；另一合法值是
  `output_hidden_states`，它让模型通过标准输出返回整组 hidden states，再从中选择。
- **Why**：hook 路径意在避免物化全部层输出。它在理论上可减少峰值暂存，但本文没有硬件测量，不能据此声称
  已获得端到端加速或显存下降。

该实现是 SpeCo 的、带版本依赖的 runtime patch；如果目标模型模块结构无法定位指定层或 final norm，会 fail closed，
而不是静默使用错误监督。

#### 重要继承参数：`hidden_state_window_tokens_per_sample=512`

这个参数没有在示例脚本中显式出现，而是从 `speco_base.yaml` 继承默认值 512。它是理解
`dflash_max_window=65536` 和 DFlash 训练样本形状的上游关键参数。

##### What：它到底限制什么

`hidden_state_window_tokens_per_sample=W` 指定 old-logprob 或 SGLang hidden-state 采集阶段，计划为每个入选请求保留的
**连续训练窗口规模**。在本脚本采用的 old-logprob 路径中，实现把它读成 `train_rows=W`，然后实际请求：

```text
hidden_rows = W + 1
```

多出来的一行用于保留 token/hidden 的边界与 next-token 对齐空间。它不表示“从一个请求中随机抽 W 个互不连续的 token”，
也不表示“直接产生 W 个独立 DFlash block”。它先产生一个连续的、带位置元数据的 collected window；DFlash optimizer
之后才从这个窗口内动态采样 anchors 和 block targets。

需要区分三种“样本”：

1. **RL request/sample**：一条完整的 prompt + response。
2. **Drafter collected item**：从这条完整序列中截出的一个连续 ids/hidden/mask 窗口，由本参数控制。
3. **DFlash block training target**：每次 optimizer step 从 collected item 中重新抽 anchor 后形成的 16-position block；
   一个 collected item 可以产生最多 64 个这样的 block target。

##### How：old-logprob 采集计划如何使用它

对 batch 中每条完整样本，trainer 先计算：

```text
P = 有效 prompt 长度
R = 有效 response 长度
W = hidden_state_window_tokens_per_sample
N = W + 1                  # 实际计划的 hidden rows
```

随后依次应用以下规则。

**1. 长度资格检查**

```text
P > 0 且 R >= N
```

只有满足该条件的样本才是候选。当前 `W=512`，所以要求 `R>=513`。这意味着 response 只有 200 tokens 的请求不会被
缩短为 200-row drafter sample，而是直接不进入 old-logprob hidden collection。该行为由采集计划决定，发生在
`dflash_max_window` 生效之前。

**2. 样本级抽样与 replica 预算**

候选样本还要通过：

- `collection_sample_rate`；
- `max_collect_samples_per_step_per_replica`；
- `max_collect_tokens_per_step_per_replica`。

其中 token budget 按 `N=W+1` 计费，而不是按后续 DFlash 实际留下的行数计费。当前默认每 replica 每 step 的 token budget
是 16384，因此仅看 token budget 时最多可容纳：

```text
floor(16384 / (W + 1))
```

| `W` | 每个 collected item 计费行数 `W+1` | 16384-row budget 最多容纳的样本数 | 再考虑默认 sample cap=16 |
|---:|---:|---:|---:|
| 128 | 129 | 127 | 16 |
| 512 | 513 | 31 | 16 |
| 4096 | 4097 | 3 | 3 |
| 8192 | 8193 | 1 | 1 |
| 16384 | 16385 | 0 | 0 |

所以单独把本参数调到 16384，会因为默认 token budget 小于单样本需求而导致零采集；必须同步调整 budget。
入选样本按 owner/replica 分桶，hidden states 由 actor old-logprob worker 产生，再以 tensor 或 Ray object refs 交给对应
drafter training worker。该参数因此同时影响生产端选择、传输负载和消费端样本长度。

**3. 连续窗口起点**

窗口位置由 `hidden_state_window_mode` 决定。

当默认 `mode=front` 时：

```text
start = P - 1
positions = [start, start + 1, ..., start + W]
```

也就是从最后一个 prompt token 开始，随后覆盖 response 前部。保留最后一个 prompt 位置是为了给第一个 response token
提供边界上下文。

当 `mode=random` 时：

```text
max_start_offset = R - (W + 1)
offset ∈ [0, max_start_offset]
start = P - 1 + offset
positions = [start, ..., start + W]
```

offset 不是每次调用 Python RNG 随意产生，而是根据 step、batch index、prompt/response 长度等信息做确定性 hash；
默认 `hidden_state_random_seed_by_step=true`，同一 step 与同一样本描述会得到稳定窗口，不同 step 可以变化。

**4. 捕获目标层 hidden states**

本脚本使用 DFlash，hidden layout 为 `dflash_aux`。对每个计划位置，forward hooks 捕获 DFlash 配置要求的多个 target
context layers，再沿最后一个维度拼接。设：

```text
C = num_context_layers，当前默认 dflash_num_target_layers=5
H = target model 每层 hidden size
N = W + 1
```

则单条 collected hidden tensor 的逻辑形状为：

```text
hidden_states: [N, C * H]
hidden_positions: [N]
```

在 BF16 下，仅这个 hidden tensor 的原始字节量近似为：

```text
2 * (W + 1) * C * H bytes
```

例如仅作量级演示，若 `H=4096`、`C=5`、`W=512`，单条 raw BF16 hidden tensor 约为 20 MiB；
若把 `W` 提到 4096，则约为 160 MiB。实际峰值还包括模型输出暂存、位置、ids、Ray/object-store、batch padding
与训练副本，不能只用这一个张量估算总显存或主存。

##### 例 1：`front` 模式怎样从完整序列构造 collected item

假设采用零基位置，并且：

```text
prompt 长度 P = 100      # prompt 位于全局位置 0..99
response 长度 R = 1000   # response 位于全局位置 100..1099
W = 512
N = W + 1 = 513
mode = front
```

采集计划得到：

```text
start = P - 1 = 99
hidden_positions = [99, 100, ..., 611]  # 共 513 行
```

语义上，它包含：

```text
位置 99：最后一个 prompt token 的 target hidden
位置 100..611：response 前 512 个位置的 target hidden
```

old-logprob forward 只把这些位置上 DFlash 所需的 C 层 hidden 捕获并拼接。进入 drafter worker 后，位置对齐逻辑会建立：

```text
input_ids：从全局位置 99 开始的连续 token 窗口
hidden_states：与 hidden_positions 99..611 对应的 513 行 aux hidden
loss_mask：位置 99 为 0；有效 response 位置为 1
```

对齐阶段会临时多保留一个 token id 作为右边界；DFlash backend 随后以 ids、hidden、mask 的共同长度对齐，当前路径
通常形成长度 513 的局部训练序列：

```text
local index:       0       1       2                512
global position:  99      100     101               611
token role:       P_last  R_0     R_1      ...      R_511
loss_mask:         0       1       1                  1
hidden:           h_99    h_100   h_101             h_611
```

由于脚本设置 `dflash_max_window=65536`，这个 513-row collected window 不会被第二次裁剪。

##### 例 1 继续：一个 collected item 如何派生 DFlash block targets

脚本继承 `dflash_block_size=16`，并设置 `dflash_num_anchors=64`。在每次 optimizer step 中，DFlash 从该局部序列
满足以下条件的位置里均匀无放回抽 anchor：

```text
loss_mask[anchor] > 0
anchor + block_size <= local_sequence_length
```

假设这一次抽到局部 anchor `a=100`：

```text
局部 anchor index = 100
对应全局 token position = 99 + 100 = 199
block 覆盖全局 token positions = 199..214
```

DFlash 为该 block 构造：

```text
noise/draft input:
  [token_199, MASK, MASK, ..., MASK]   # 总长 16

labels:
  [token_199, token_200, ..., token_214]

loss:
  offset 0 被排除
  实际监督 token_200..token_214
```

与此同时，drafter 把 collected window 的多层 target hidden 当作 context features。一次 optimizer step 可以从同一个
collected item 抽最多 64 个不同 anchors，因此 `W=512` 的意义是提供一个可重复采样的连续上下文与候选位置池，
而不是只生成一个 512-token causal-LM 样本。

下一次 optimizer step 会重新调用 anchor sampler，所以即使仍选中同一个 collected item，也可能派生不同的 64 个 blocks。
这就是脚本 `step=10` 能够多次复用同一批 hidden-state 数据的方式之一。

##### 例 2：`random` 模式怎样改变覆盖区域

仍取 `P=100`、`R=1000`，但设：

```text
W = 128
N = 129
mode = random
max_start_offset = 1000 - 129 = 871
```

假设确定性 hash 得到 `offset=300`：

```text
start = 99 + 300 = 399
hidden_positions = [399, ..., 527]  # 129 行
```

该窗口位于 response 中段，而不是 response 前部。它覆盖的 response 零基 offset 大约是 299..427；局部 loss mask
通常全为 1。DFlash 仍然在这个局部窗口内抽 16-position blocks，并且任何 anchor 都不能跨出位置 527。

因此：

- `front` 更稳定地训练 response 开头及 prompt→response 边界；
- `random` 在多个 RL steps 上有机会覆盖 response 中后部；
- `W` 决定每次可见的连续跨度；mode 决定这段跨度放在哪里。

##### 它如何影响 DFlash 训练成本与数据多样性

增大 `W` 会带来几类不同影响：

- **候选资格下降**：要求 `response_len >= W+1`，短 response 会被过滤。
- **每 step 样本数下降**：固定 token budget 下，每条样本消费 `W+1` 行，可能使 collected items 数量减少。
- **位置池扩大**：可供 anchor 抽样的位置更多；但 anchor 数仍最多为 64。
- **context 扩大**：即使 anchors 已饱和为 64，DFlash attention 的 context K/V 仍会随窗口变长。
- **传输和存储线性增长**：aux hidden 的主体字节量近似随 `(W+1)*C*H` 增长。
- **覆盖方式变化**：较大的 front window 覆盖更多 response 前部；较小 random window 配合跨 step 变化可能用较低单步成本覆盖不同区域。

当有效位置已经足以稳定采到 64 个 anchors 后，继续扩大 `W` 不会增加每次 optimizer step 的 block 数；它主要增加
候选多样性、上下文长度和 hidden-state 成本。因此不能仅因为长序列上限很大就把该参数设到最大。

##### 与其他参数的联动

| 参数 | 与 `hidden_state_window_tokens_per_sample` 的关系 |
|---|---|
| `hidden_state_window_mode` | 决定连续窗口位于 response 前部还是一个确定性随机位置 |
| `hidden_state_window_min_rows` | 当本参数为 `null` 时，old-logprob 路径用它作为 fallback；不是同时取二者最大值 |
| `hidden_state_random_max_offset` | 当前 old-logprob 计划没有使用它限制 random offset；它主要出现在 SGLang runtime 的窗口规划中 |
| `max_collect_tokens_per_step_per_replica` | 必须至少大于 `W+1` 才可能在该 replica 收到一个样本 |
| `max_collect_samples_per_step_per_replica` | 与 token budget 共同决定每 replica 实际收集多少样本 |
| `dflash_max_window` | 对已经采集的窗口做第二次上限裁剪，不能扩大窗口 |
| `dflash_num_anchors` | 限制每个 collected item 每次 forward 实际派生的 block 数 |
| `dflash_block_size` | 决定每个 anchor 需要的连续目标长度；窗口必须足够容纳完整 block |
| `batch_size_per_gpu` | 决定一次 optimizer step 最多选多少个 collected items，不等于采集样本上限 |

##### 针对当前脚本的调参判断

当前组合是：

```text
hidden_state_window_tokens_per_sample = 512  # 继承默认
dflash_max_window = 65536
dflash_num_anchors = 64
dflash_block_size = 16
max_collect_samples_per_step_per_replica = 16
max_collect_tokens_per_step_per_replica = 16384
batch_size_per_gpu = 4
```

其含义是：每个满足 `response_len>=513` 的入选请求，通常产生一个约 513-row 的 DFlash collected item；
每个 replica 每次最多收 16 个这种 item；每个 optimizer step 最多选 4 个 item；每个 item 动态抽最多 64 个
16-position blocks。`dflash_max_window=65536` 在这一组合中只是不再裁剪 513-row item。

若要把 `W` 从 512 提到 4096，至少要重新检查：

1. 实际 response 长度分布中有多少样本达到 4097 tokens；
2. 默认 16384-row budget 每 replica 只允许 3 个样本，是否足够支撑 batch size 4；
3. 单样本多层 BF16 hidden payload、object-store 与训练设备内存是否可承受；
4. anchors 已经封顶为 64 后，更长 context 是否带来可测的 acceptance/throughput收益；
5. `dflash_max_window` 是否至少为 4097，否则采集后又会被训练后端裁短。

调参时建议记录实际 `candidate_count`、`selected_count`、collected rows/payload MiB、训练 batch 的
`packed_tokens_before_shift`、成功 optimizer steps、DFlash forward 时间、显存峰值和 speculative acceptance，
避免只看到配置中的 `W` 而误判真实样本数量和训练长度。

### 5.2 DFlash 样本与损失

#### `dflash_num_anchors=64`

- **What**：控制每条训练样本最多选多少个 block 起点。
- **How**：在满足 loss mask 且能容纳完整 block 的位置中均匀无放回采样；有效位置少于 64 时只使用实际可用数量。
  这是 **每条样本** 的上限，不是整个 GPU batch 的总数。
- **Why**：anchor 越多，单次 optimizer step 覆盖的上下文位置越多，但 draft forward、attention 和 loss 成本近似随 anchor 数增加。

脚本未覆盖 `dflash_block_size`，因此继承默认值 16。每个有效 anchor 产生一个 16-position block，但 offset 0
仅提供已知 anchor token，loss 从 offset 1 开始。

#### `dflash_max_window=65536`

- **What**：DFlash backend 在单个样本进入训练 batch 前，对已经收集并完成位置对齐的
  `input_ids`、`hidden_states` 和 `loss_mask` 同时施加的最大连续窗口长度。
- **How**：它是训练侧的第二级裁剪，不参与 old-logprob 采集计划，也不改变 rollout/old-logprob 阶段已经生成的
  Ray payload。裁剪后，DFlash 才构造 padded batch、采样 block anchors 并执行 forward/loss。
- **Why**：限制单个长样本进入 DFlash attention 的上下文长度和 batch padding 长度，同时尽量保留首个监督 token
  前后的上下文。脚本设为 65536，等价于允许 DFlash backend 尽可能完整保留上游交给它的窗口，而不是主动退回默认 512。

##### 三级窗口关系

理解该参数最重要的是区分三个不同长度：

1. **原始序列长度** `L_raw`：有效 prompt tokens 与 response tokens 的总数。本脚本配置上限为
   `32768 + 16384 = 49152`，但实际长度由数据与生成终止位置决定。
2. **采集/对齐窗口长度** `L_collected`：old-logprob 只为计划位置捕获 hidden states。脚本没有覆盖
   `hidden_state_window_tokens_per_sample`，所以默认 `train_rows=512`，并请求额外一行形成
   `hidden_rows=513`。采集计划通常从原序列位置 `prompt_len - 1` 开始。
3. **DFlash 训练窗口长度** `L_train`：先对 ids、hidden 和 mask 做共同长度对齐，再由
   `dflash_max_window=W` 二次裁剪。因此大致有：

```text
L_aligned = min(L_input_ids, L_hidden_states, L_loss_mask)
L_train   = min(L_aligned, W)
```

这里的 `W` 只能缩短已经采集的窗口，不能令 `L_train` 超过 `L_collected`，更不能恢复未采集的原始序列位置。

##### 精确裁剪算法

设共同对齐后的长度为 `L`，`W=dflash_max_window`，`r_start` 是 `loss_mask` 中第一个非零位置。
当前实现计算：

```text
start = clamp(r_start - floor(W / 2), 0, max(0, L - W))
end   = min(start + W, L)
```

然后同步切片：

```text
input_ids    = input_ids[start:end]
hidden_states = hidden_states[start:end]
loss_mask    = loss_mask[start:end]
```

若整个 `loss_mask` 都为 0，则保留序列尾部 `W` 行，即 `[max(0, L-W):L]`。因此它不是随机窗口，也不是
固定从序列头部截断；一般情况下，它试图让第一个监督位置前面最多保留约半个窗口，剩余部分留给监督区域。
靠近序列头尾时会被边界 clamp。

##### 当前脚本中的实际样本形状

本脚本走 old-logprob 路径，默认采集过程可以简化为：

```text
完整有效序列：[prompt tokens | response tokens]
                         ^ response 起点

采集起点：prompt_len - 1
计划 hidden rows：513
初始局部窗口：[最后 1 个 prompt token | 前部 response tokens ...]
```

`collect_online_data()` 根据 hidden positions 做对齐时，构造约 514 个局部 token ids 和 513 行 hidden states；
DFlash 预处理先取三者共同长度，所以 `L_aligned` 通常是 513。此时：

```text
W = 65536
L_aligned = 513
L_train = min(513, 65536) = 513
```

因此当前值不会产生任何二次裁剪。它只是完整保留默认 old-logprob 窗口。若使用配置默认值 `W=512`，
通常只会把这个 513-row 窗口裁成 512 rows；它仍然不会从完整的 49K 上限序列中直接选择 512 rows，
因为完整序列在更早的 old-logprob 采集阶段已经变成局部窗口。

另一个容易忽略的影响是：old-logprob 采集计划要求 `response_len >= hidden_rows`。在当前默认值下，response
少于 513 个有效 token 的样本不会进入这个采集计划；调大 `dflash_max_window` 不会放宽这一门槛，因为候选选择发生在
DFlash backend 看到样本之前。

##### 如果把 `dflash_max_window` 调小

假设上游仍交给 DFlash 一个 513-row 局部窗口，并且局部 `loss_mask` 的 index 0 是最后一个 prompt token、
index 1 开始是 response：

| `W` | DFlash 保留的典型局部范围 | 结果 |
|---:|---|---|
| `65536` | `[0:513]` | 不裁剪 |
| `512` | `[0:512]` | 少 1 行尾部 response/context |
| `256` | `[0:256]` | 保留边界 token 与前 255 个 response 位置 |
| `128` | `[0:128]` | anchor 候选和 attention context 进一步缩小 |

因为该 old-logprob 局部窗口只在 response 前保留约一个 token，公式中的“首个 loss token前半个窗口”在这里无法真正
保留半窗 prompt：`r_start` 通常约为 1，`start` 会被 clamp 到 0。该居中策略在上游样本本身包含较长 prompt 前缀时
才会明显体现。

##### 对后续训练样本构造的影响

裁剪后的每个样本保持独立边界，不会与另一个样本拼成一条 DFlash 序列。batch 构造会：

1. 对每条样本保留裁剪后的 `ids / hidden / loss_mask`；
2. 以该 minibatch 中最长的 `L_train` 为长度，把其他样本右侧补零；
3. 用 `loss_mask` 保证 padding 和非监督位置不会成为有效 loss；
4. 在每条 padded row 内独立采样 anchors，anchor 不跨样本边界。

因此一个更大的 `W` 可能让某个长样本抬高整个 minibatch 的 padding 长度。对于当前默认
`batch_size_per_gpu=4`，训练 batch 的主要张量形状近似为：

```text
input_ids:     [B, L_batch]
hidden_states: [B, L_batch, num_context_layers * target_hidden_size]
loss_mask:     [B, L_batch]

B = 实际样本数（最多 4）
L_batch = max(每条样本裁剪后的 L_train)
```

##### 对 anchor 与 loss 的影响

`dflash_max_window` 不直接指定 anchor 数，但会改变可用位置集合。默认 `dflash_block_size=16` 时，长度为 `L_train`
的样本只在能够容纳完整 16-position block、且 anchor 处 `loss_mask>0` 的位置中采样，最多再受
`dflash_num_anchors=64` 限制：

```text
max_anchor_position = L_train - dflash_block_size
actual_anchor_count = min(64, valid_anchor_positions)
```

若窗口仍有足够多的有效 response 位置，anchor 数会很快饱和到 64；此后继续增大 `W` 不增加单步 anchor 数，
但会扩大每个 anchor 可以 attend 的 context K/V 长度。若 `W <= dflash_block_size`，实现没有可训练的完整 block，
所以应把 `dflash_max_window` 保持为严格大于 block size 的正整数。

裁剪还会改变 `restricted_ce` 的 batch 内候选词集合，因为 loss 只看裁剪后 `input_ids` 中出现的 token 与 active targets。
窗口越小，受限词表通常也越小。不过当前 target LM-head 的 row-restricted 同步集合是在更早阶段从 collected items 构造的，
所以仅调小 `dflash_max_window` 不一定同比减少 target-head 同步通信量。

##### 它能节省和不能节省的成本

调小 `dflash_max_window` 会直接减少：

- DFlash 训练 batch 的 padded sequence length；
- DFlash context attention 的 K/V 长度；
- 保留在最终 batch 中的 hidden-state 张量和部分 loss/logits 工作量；
- 可能的 restricted vocabulary 大小。

但按当前代码顺序，backend 在裁剪前已经执行了 `input_ids.to(device)` 和整块 hidden states 的
`to(device, dtype=bf16)`，所以它**不会**减少这一次初始 CPU→训练设备传输量，也不会减少更早的 old-logprob
hidden capture、CPU/object-ref 存储或 Ray payload。若目标是降低采集和传输成本，应优先调整
`hidden_state_window_tokens_per_sample` 及每 step 的采集预算；若目标只是限制 DFlash forward/batch 成本，
`dflash_max_window` 才是直接旋钮。

##### 调参原则

- 先决定需要采集多少连续监督行：调 `hidden_state_window_tokens_per_sample`。
- 再决定 DFlash 每次训练最多消费其中多少行：调 `dflash_max_window`。
- 保证 `dflash_max_window > dflash_block_size`，并检查裁剪后仍有足够的 `loss_mask>0` 位置。
- 与 `dflash_num_anchors` 联合看：当 anchors 已稳定达到 64 时，扩大窗口主要增加 context cost，而不是增加每步监督 block 数。
- 用 `packed_tokens_before_shift`、`packed_loss_tokens`、实际 anchor 数、DFlash forward 时间和显存峰值验证，
  不要把配置值本身当作实际训练长度。

若要做 65K 训练，必须同时扩大采集窗口与 token budget，并重新评估 hidden-state 传输、batch padding、DFlash attention
后端和显存；只设置 `dflash_max_window=65536` 不会形成 65K 训练样本。

#### `dflash_loss_mode=restricted_ce`

- **What**：把完整词表交叉熵改成当前训练 batch 的受限词表交叉熵。
- **How**：候选集合为该 batch 的 `input_ids` 与所有 active target ids 的去重并集；只索引 target LM head 的这些行，
  再在这个子词表上做 softmax/CE。它不是 top-k teacher distillation，也不是从完整词表随机负采样。
- **Why**：显著降低大词表 logits 的矩阵乘与显存/传输量。代价是优化目标发生变化：分母不再包含完整词表中的其他 token，
  所以该 loss 数值和 `full_vocab` CE 不可直接比较，也不能称为完整词表 CE 的无损等价物。

当前默认 `target_lm_head_row_restricted_sync=true` 会配合这一模式，只同步候选 token 对应的 target head 行；
若无法构造安全的行集合，实现会回退到更保守的同步路径。

#### `dflash_loss_decay_gamma=7`

- **What**：控制 block 内随预测距离增加的 loss 衰减速度。
- **How**：对 block offset `k` 使用
  `exp(-max(k - 1, 0) / gamma)`。offset 0 本来就被排除；offset 1 权重为 1，offset 2 为 `exp(-1/7)`，
  之后继续衰减。
- **Why**：让更靠近 anchor、通常更容易且对短 speculative acceptance 更直接的预测承担更高权重。
  gamma 越大衰减越慢；小于或等于 0 时不应用该衰减。

#### `dflash_front_position_weight=2.0` 与 `dflash_front_position_count=3`

- **What**：共同提高 block 前 3 个可训练预测位置的重要性。
- **How**：满足 `0 < offset <= 3` 的位置再乘 `2.0`；它与上述指数衰减相乘，而不是替换衰减权重。
- **Why**：进一步偏向 speculative block 的前部，因为前部错误会更早终止接受链。

组合后的相对权重为：offset 1 是 `2.0`，offset 2 是 `2*exp(-1/7)`，offset 3 是
`2*exp(-2/7)`；从 offset 4 起只有指数衰减。最终 loss 仍按有效加权 token 数归一化，因此这些是相对权重。

#### `dflash_hard_sample_ratio=0.3`

- **What**：期望每个 drafter batch 约 30% 的样本从困难样本池选取，其余随机选取。
- **How**：困难度优先使用显式 hard score/sample loss；否则用 acceptance length 或 acceptance rate 的负值，
  即接受越短越困难。实现先从分数最高的一小池中随机取 hard 部分，再从剩余样本随机补齐。
- **Why**：把一部分训练预算集中到 drafter 当前表现较差的请求，同时保留随机样本的覆盖面。

**当前脚本路径的实际效果**：old-logprob sample 构造代码只写入 token、hidden states/refs、位置、布局、step 和
replica 元数据，没有写入上述任一困难度字段。因此 `_dflash_hard_sample_score()` 返回 `None`，hard 部分为空，
随后整个 batch 都由随机抽样补齐。也就是说，在当前 old-logprob 数据路径上，这个 `0.3` 通常不产生 hard-mining 效果。
要让它生效，需要显式把 rollout acceptance 或可比较的 sample loss 关联回 old-logprob sample；这是后续设计工作，
不是改一个比例即可完成的配置调整。

### 5.3 采集与训练调度

#### `collect_interval_steps=5`

- **What**：每 5 个 RL trainer global steps 允许一次特征采集。
- **How**：判定是 `global_step > 0 and global_step % 5 == 0`，所以命中 5、10、15……，不会在 step 0 触发。
- **Why**：降低每个 RL step 都采集和搬运 hidden states 的成本。

#### `training_interval_steps=5`

- **What**：每 5 个 RL trainer global steps 允许一次 drafter 训练周期。
- **How**：使用与采集相同的整除规则。在 old-logprob 模式下，采集计划还要求 training interval 同时命中，
  所以这两个 interval 并不是完全独立的两个异步时钟。
- **Why**：让训练频率与数据刷新频率保持一致，避免在默认 `use_data_buffer=false` 时命中训练却没有当前 step 样本。

本脚本把两者都设为 5，因此预期训练周期是 step 5、10、15……。如果把它们设成不相同的数，old-logprob 实际采集
只发生在两者共同命中的 step，即近似由两者最小公倍数控制；若没有启用 data buffer，其他训练命中通常也没有可训练样本。

#### `step=10`

- **What**：一次训练周期内尝试的 drafter training-step 次数。
- **How**：worker 做 10 次循环，每次重新准备 batch、前向、反向、梯度裁剪、optimizer step 和 scheduler step。
  只有存在有效 batch、有限 loss、有限 gradient 的迭代才算成功 optimizer step。
- **Why**：用一个采集周期的数据做多次更新，提高每次 hidden-state 采集的复用率；代价是数据重复使用、训练耗时和过拟合风险增加。

脚本未覆盖 `batch_size_per_gpu`，因此默认每个 training step 最多取 4 个样本。默认 `use_data_buffer=false`，
所以只从当前 RL step 的已采集样本中取数据；10 次迭代可能反复抽到相同样本，但 anchor 会在每次 DFlash forward 中重新随机采样。

### 5.4 权重 snapshot 与 vLLM 热更新

#### `publish_async=True`

- **What**：driver 发起 rollout worker 的权重更新 RPC 后不立即 `ray.get` 等待。
- **How**：返回的 refs 被保存；发起下一次发布之前、进入下一次 `generate_sequences` 之前，以及训练流程退出时都会等待它们完成。
- **Why**：允许传输/加载与当前 RL step 的其他可用工作重叠，同时在下一轮生成前保住版本一致性。

它并不意味着 vLLM 能一边生成同一批请求一边安全地改权重：本脚本还设置了
`draft_update_pause_generation=True`。异步化的是 driver 等待方式，不是取消 runtime 内部的一致性边界。

#### `publish_dtype=bf16`

- **What**：把准备发布的 trainable state tensors 转成 BF16。
- **How**：drafter training worker 生成 detached、CPU、contiguous 的 BF16 snapshot，再交给 rollout 更新路径。
- **Why**：相对 FP32 大约减半 snapshot 内存与传输字节数，并与常见 BF16 推理权重匹配；代价是发布时舍弃 FP32 精度。
  optimizer state 不会发布给 vLLM。

#### `draft_update_weights_bucket_megabytes=512`

- **What**：设置 bucketed weight sender 的单桶目标大小为 512 MiB。
- **How**：值被换算为 MiB 字节数，传给 VeRL vLLM bucketed weight-transfer sender。它控制传输分块，不是总权重上限。
- **Why**：大桶可减少分桶/消息调度开销，但会增加单桶暂存、共享内存或 IPC 压力，并可能减弱流水化粒度。
  512 MiB 是否最优必须在目标节点、传输模式与 drafter 大小上测量，不能从代码静态判断。

#### `draft_update_pause_generation=True`

- **What**：权重更新前暂停生成请求，更新结束或异常退出时恢复。
- **How**：vLLM adapter 的 rollout rank 0 调用 `abort_all_requests(reset_prefix_cache=flush_before)`；更新放在
  `try/finally` 中，finally 调用 `resume_generation`。
- **Why**：避免活跃请求在一次 speculative sequence 内混用两个 drafter 版本。代价是更新期间形成明确的 rollout bubble。

#### `draft_update_flush_before=False`

- **What**：不要求在加载新 drafter 权重前清空 prefix/KV cache。
- **How**：因为 pause 为 true，该值作为 `reset_prefix_cache=False` 传给 `abort_all_requests`；若 pause 为 false，
  它才会单独控制更新前的 `clear_kv_cache`。
- **Why**：减少一次更新前 cache 清理开销。它不是“完全不清 cache”，因为本脚本仍启用了更新后 flush。

#### `draft_update_flush_after=True`

- **What**：新权重加载完成后清空 vLLM KV cache。
- **How**：bucket transfer 和 runtime weight load 完成后，rollout rank 0 调用 `clear_kv_cache`，之后设置新的 global step，
  最终恢复生成。
- **Why**：防止新 drafter 复用旧模型版本生成的缓存，是较保守的一致性选择；代价是下一轮请求失去可复用 cache，
  增加重新 prefill 的成本。

## 6. 这些参数组合起来意味着什么

### 6.1 预期节奏

假设 1-based `global_step`，且每次都有足够有效样本：

| RL global step | 采集 old-logprob hidden | 尝试 drafter optimizer steps | 成功后发布 |
|---:|---:|---:|---:|
| 1–4 | 否 | 0 | 否 |
| 5 | 是 | 10 | 是 |
| 6–9 | 否 | 0 | 否 |
| 10 | 是 | 10 | 是 |

由于 `publish_interval_steps=0`（继承默认值），发布跟随每次有成功 optimizer step 的训练周期；不是每 5 步无条件发布。

### 6.2 单个 DFlash target 的相对 loss 权重

在默认 `dflash_block_size=16` 下，anchor offset 0 不训练，offset `k=1..15` 的基础权重为：

```text
w(k) = exp(-(k - 1) / 7) * (2.0 if 1 <= k <= 3 else 1.0)
```

该权重还会乘 block keep mask、有效 label mask 和原始 response/loss mask，最终用有效加权 token 数归一化。

### 6.3 性能与正确性取舍

- **节省项**：复用 old-logprob forward、restricted vocabulary CE、受限 target-head 行同步、BF16 snapshot、异步 driver RPC。
- **新增项**：hidden-state 选择/拷贝/Ray 传输、每 5 步 10 次 drafter update、snapshot 构造、512 MiB 分桶传输、
  vLLM pause、weight load、更新后 cache flush。
- **正确性边界**：hidden layout 或模块定位不兼容时 fail closed；下一轮生成等待 pending publish；更新后清 cache。
- **不能从配置直接得出的结论**：该组合是否提高端到端 RL throughput。必须同时测量 rollout 节省与 drafter
  训练/发布/cache 失效成本。

## 7. 建议关注的指标

解释或调参时，至少联合观察：

- `timing_s/drafter`、`timing_s/drafter_train_rpc`；
- `timing_s/drafter_prepare_batch`、forward/backward/optimizer 细分；
- `timing_s/drafter_publish_wait_pending`、`drafter_publish_fetch_snapshot`、`drafter_publish_update_weights`；
- `drafter/collected_samples`、`drafter/train_successful_steps_max`、`drafter/published`；
- DFlash loss、top-1/top-5、按 block position 的 loss/accuracy；
- speculative mean acceptance length、rollout token/s 和完整 RL step wall time；
- 更新前后 GPU memory peak，以及 cache flush 导致的 prefill 变化。

最高 ROI 的首个核对动作是：确认每个命中 step 实际采集行数是否约为每样本 513，并确认
`dflash_hard_sample_ratio=0.3` 对当前样本没有产生 hard-selection；这两点会直接改变对 65K window 和 hard mining 的理解。

## 8. 未决问题与下一步阅读目标

1. 该示例为何同时把 `dflash_max_window` 提到 65536、却保留 512-row old-logprob 采集窗口？这是有意的后端上限，
   还是遗漏了 `hidden_state_window_tokens_per_sample` 覆盖？需要结合运行日志或示例作者意图确认。
2. 是否应把 vLLM rollout acceptance 元数据稳定关联到 old-logprob batch item，使 `dflash_hard_sample_ratio` 真正生效？
   这需要明确 request/sample identity 和跨阶段对齐契约。
3. 512 MiB bucket、异步发布以及更新后 cache flush 的净收益尚未在目标硬件上测量。
4. 若扩大采集窗口，需要先做显存预估，并分别测量 hidden capture、Ray object transfer、DFlash attention 与 target-head
   row sync 的增长，不能在本地 16 GB RTX 5080 上直接尝试全尺寸 65K RL 训练。

建议下一步阅读：

- `verl_speco/trainer/speco_ray_trainer.py`：采集/训练/发布的 trainer 级调度；
- `verl_speco/integration/oldlogprob_runtime.py`：forward-hook 捕获、位置选择和 object-ref 传输；
- `verl_speco/trainer/base_trainer.py`：样本对齐、hard sampling 与 optimizer step；
- `verl_speco/backends/dflash_trainer_backend.py`：anchor 采样、窗口裁剪和 loss；
- `verl_speco/integration/vllm_runtime.py`：vLLM drafter 热更新、pause/flush 与 bucket transfer。

## 9. 来源与版本上下文

本文基于 2026-08-11 本地工作区的当前内容：

- Git HEAD：`c9846b27604c15a2dc8dd7b4bfd083af76e7ef86`；
- 工作树当时存在未提交修改，且示例脚本、`speco_base.yaml`、DFlash backend 等相关文件均有本地修改；
  因此本文刻意描述的是 **当前工作树快照**，不保证等同于该 HEAD 的干净版本；
- 支持基线按仓库约定视为 VeRL v0.8.0；本文涉及的 drafter online training 与 runtime patch 均标注为 SpeCo overlay 行为。

主要源码证据（行号对应上述工作区快照）：

- 示例参数：`examples/run_qwen3-8b_drafter_dflash_vllm.sh:63-88`；
- 默认配置：`verl_speco/config/speco_base.yaml:49-117`；
- interval 与 old-logprob 约束：`verl_speco/trainer/speco_ray_trainer.py:821-905`；
- 512-row old-logprob 采集计划：`verl_speco/trainer/speco_ray_trainer.py:949-953,1122-1224`；
- old-logprob sample 字段：`verl_speco/trainer/speco_ray_trainer.py:1440-1467`；
- DFlash hidden layout 与 target layer 选择：
  `verl_speco/integration/oldlogprob_layer_ids.py:76-93,165-177,269-293`；
- collected item 的 token/hidden/loss-mask 对齐：`verl_speco/trainer/base_trainer.py:2354-2642,2835-2874`；
- DFlash minibatch 的独立样本 padding：`verl_speco/trainer/base_trainer.py:3379-3504,3769-3852`；
- 训练触发与发布：`verl_speco/trainer/speco_ray_trainer.py:1847-1906,2271-2319`；
- DFlash anchor、位置权重与 restricted CE：`verl_speco/backends/dflash_trainer_backend.py:160-218,271-287,300-454`；
- DFlash 后端窗口：`verl_speco/backends/dflash_trainer_backend.py:1052-1127`；
- hard sampling：`verl_speco/trainer/base_trainer.py:3220-3288`；
- optimizer-step 语义：`verl_speco/workers/speco_worker.py:986-1032` 与
  `verl_speco/trainer/base_trainer.py:4259-4428`；
- BF16 snapshot：`verl_speco/trainer/base_trainer.py:4474-4506`；
- vLLM pause/flush/bucket update：`verl_speco/integration/vllm_runtime.py:2272-2350`。

## 10. 自检问题

1. 为什么 `collect_interval_steps=5` 与 `training_interval_steps=5` 不代表两个互相独立的后台周期？
2. `dflash_max_window=65536` 为什么不等价于每个样本训练 65536 个 hidden-state rows？
3. offset 2 的 loss 权重如何同时受到 gamma 与 front-position 参数影响？
4. `publish_async=True` 为什么仍然不会允许下一轮 rollout 使用尚未完成更新的 drafter？
5. 当 `hidden_state_window_tokens_per_sample=4096`、采集 token budget 仍为 16384 时，每个 replica 最多能收几个样本？
6. 为什么一个 513-row collected item 可以在连续 10 个 optimizer steps 中派生出不同的 DFlash block targets？
