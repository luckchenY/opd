# process_reward_weighted_distill 算法流程笔记

对应脚本：`tools/process_reward_weighted_distill.py`
依赖共享：`tools/process_reward_distill.py`（提供 segmentation / vLLM lifecycle / dataset helpers）

---

## 一、核心思想

把 **process reward (PR) 信号** 嵌进 **GKD（generalized knowledge distillation, 学生 ↔ 老师 token 级 JSD）** 的 loss 里，做 **段落级加权蒸馏**。

- **Student**: Qwen3-1.7B（待训练）
- **Teacher**: Qwen3-8B（冻结）
- **池子**: Phase 0 一次性筛出「teacher 能做对」的题，**整个训练全程冻结**，不增不减
- **每 epoch 重做一次**: student rollout + 段落级 PR 评估 + 加权 JSD 训练 + held-out 评测

**与 process_reward_distill 的关键区别**：

| 维度 | process_reward_distill | process_reward_weighted_distill |
|---|---|---|
| 池子 | 动态（学会的剔除、续写补充） | 静态冻结 |
| 训练记录 | 三分支（new_response / continuation / replay） | 单分支（一条 student 完整 rollout + 段权重） |
| 每题记录数 | 每 epoch 多条 | 每 epoch 一条 |
| 训练框架 | swift rlhf（GKD trainer） | 自定义 HF Trainer + 自写 weighted JSD |
| 信号利用方式 | 用 PR 决定**采样什么样本** | 用 PR 决定**每个 token 的 loss 权重** |

---

## 二、关键超参与环境变量

```text
PR_ALPHA            权重公式 weight = max(0, 1 + α·pr).  默认 1.0.
MERGE_SEGMENTS      是否折叠单调 PR run.  默认 true.
N_EPOCHS            外循环 epoch 数.  默认 5.

GKD_LR              基础 peak LR.  默认 1e-6.
GKD_BETA            JSD 中 β（β=0:KL(s||t), β=1:KL(t||s), 中间值:JSD）. 默认 0.5.
GKD_TEMPERATURE     softmax 温度.  默认 1.0.
GKD_BATCH_SIZE      per-device batch.  默认 1.
GKD_GRAD_ACCUM      梯度累积.  默认 4.
GKD_MAX_LEN         token 上限.  默认 MAX_NEW_TOKENS + 1024.
GKD_WARMUP          warmup 比例.  默认 0.05.
GKD_LR_SCHEDULE     cosine|linear|constant. 默认 cosine（外层 peak 衰减）.

TEST_EVAL_EVERY     test 评测频率.  默认 1（每 epoch）.
TEST_EVAL_SAMPLES   test 子采样上限.  默认 500（最后一 epoch 用全量）.
N_SAMPLES           池上限（0 = 全量）.

POOL_DATASET        hendrycks_math (默认) | dapo_math_17k.
DAPO_SHARDS         冒号分隔的 arrow shard 路径（覆盖默认）.
OUTPUT_DIR          产出目录. 默认 tools/weighted_distill_output.
```

**LR 双重调度的设计**：
- 外层 `_lr_for_epoch` 跨 epoch 衰减 peak LR（cosine/linear/constant）
- 内层 Trainer 用 `constant_with_warmup`（warmup 到 peak 后保持）
- 这样避免「外 cosine × 内 cosine」的双重衰减

---

## 三、Phase 0：构建静态 T✓ 池（仅首次运行）

`phase0_build_pool()` 流程：

1. **若 `pool.jsonl` 已存在** → 直接 load 返回（断点续跑友好）
2. **加载原始题目**（通过 `_load_pool_samples_normalized`）：
   - `POOL_DATASET=hendrycks_math` → 走 `load_dataset_samples()` + `extract_boxed_answer` 提取 gt
   - `POOL_DATASET=dapo_math_17k` → 走 `_load_dapo_samples_normalized`，从 arrow shard 读 row，用 `_extract_problem_text` / `_extract_gt_answer` 兼容多种 schema
   - 统一归一化为 `[{"problem": str, "gt_answer": str}, ...]`
3. **Teacher rollout**（`run_isolated_rollout` 起独立子进程跑 vLLM）：
   - temperature=0（贪心），max_tokens=MAX_NEW_TOKENS
   - 子进程跑完即销毁 CUDA context，GPU 干净
4. **T✓ 过滤**：用 `extract_boxed_answer` + `answers_match` 判断 teacher 答案是否对，对的留下：
   ```json
   {"problem": "...", "gt_answer": "..."}
   ```
5. **写 `pool.jsonl`**（之后整个训练只读不改）
6. **Test set**：仅 MATH 时保存 `test_set.jsonl`（官方 test split 全量），DAPO 跳过（无 test split）
7. **Train-eval 快照**：把 pool 复制一份存 `train_eval_set.jsonl`，并初始化 `train_eval_solved.json = []`（已解出题目的索引集合）

---

## 四、Per-epoch 主循环（`run_epoch`）

每个 outer epoch 由 orchestrator 串行 spawn **三个独立子进程**：

```
Step 0–4 (inference) → Step 5 (training, torchrun) → Step 6 (eval)
```

子进程隔离的目的：vLLM 和 HF Trainer 共用 GPU 时显存/CUDA 状态会打架，spawn → exit 是最干净的隔离。

### Step 0–4：`_epoch_inference_worker`（vLLM 推理子进程）

加载 `current_ckpt` 进 vLLM，按 `INFERENCE_BATCH_SIZE` 分 mini-batch 处理整个池子：

#### Step 1: Student rollout
- prompt: `build_filter_prompt(problem, tokenizer)` → `[sys: EPISODE_EVAL_SYSTEM][user: problem]` + `add_generation_prompt`
- 采样: temperature=ROLLOUT_TEMPERATURE, top_k=50, n=1, max_tokens=MAX_NEW_TOKENS
- 通过 `safe_generate`（带 prompt 长度过滤、失败兜底）

#### Step 2: 段落切分（CPU）
`segment_response_text(response)`：
- 在 trigger word（如 "wait", "let me check", "actually"...）处切，且当前段累计句号数 ≥ `SEGMENT_MIN_PERIODS` 才切
- 返回 `[(start_char, end_char), ...]`，episode i 的前缀 = `response[:end_char]`

#### Step 3: K-fold PR 评估
对每条 student response，构建一组 forced-answer prompt：
- **Empty-prefix baseline**: `build_episode_eval_prompt(problem, "", tok)` → 学生不思考、直接强制出 boxed 答案的命中率
- **每个 episode 边界**: `build_episode_eval_prompt(problem, response[:end_c], tok)` → 截到第 i 段末尾后强制出答案的命中率

每条 prompt 采样 `K_EVAL` 次（temperature=EVAL_TEMPERATURE, max_tokens=MAX_ANSWER_TOKENS, stop=`<|im_end|>`）。
每条 prompt 的 acc_rate = K 次中命中 gt 的比例。

`build_episode_eval_prompt` 关键细节：如果 prefix 在 `<think>` 块内，先补 `</think>` 再加 `EPISODE_EVAL_SUFFIX`（强制收敛到 boxed 答案）。

K-fold 结果存进 `acc_per_li[li] = {-1: empty_acc, 0: acc_0, 1: acc_1, ...}`。

#### Step 4: 计算 PR + 段权重 + 写训练记录

`compute_segments_for_response(episodes, ep_accs, empty_acc, do_merge)`：

1. **原始 PR**: 对每段 i，`pr_i = acc_rate_i - acc_rate_{i-1}`（i=0 时上一段 = `empty_acc`）
2. **可选 merge**: 若 `MERGE_SEGMENTS=true`，调 `merge_monotonic_runs` 折叠 PR 单调 run（连续上升或连续下降 → 合并为最末端那个点）
3. **重算 PR**（合并后）: 用「相对 empty_acc 的累积变化」重算，即 `pr_merged_i = acc_merged_i - acc_merged_{i-1}`
4. **段权重**: `weight_i = max(0, 1 + PR_ALPHA · pr_i)`
   - 正 PR 段（推理帮助）→ 上权（最大 1+α）
   - 负 PR 段（推理拉跨）→ 下权，可能为 0
   - α=1 时范围 [0, 2]，安全；α>1 时 max-clamp 防负权翻转 loss 方向

每条 student response 产出一条训练记录写到 `epoch_NNN_train.jsonl`：
```json
{
  "problem": "...",
  "gt_answer": "...",
  "student_response": "<完整 student rollout>",
  "empty_acc": 0.42,
  "segments": [
    {"end_char": 320, "acc_rate": 0.6, "process_reward": 0.18, "weight": 1.18},
    {"end_char": 850, "acc_rate": 0.3, "process_reward": -0.30, "weight": 0.70},
    ...
  ]
}
```

子进程退出前 `destroy_llm` + 写 `inf_out.json`（含 `n_train_records`）。

### Step 5：`_train_worker`（torchrun DDP 训练子进程）

orchestrator 通过 `torchrun --nproc_per_node=N_GPUS` 起进程。每 rank：

#### 5.1 Dataset：`WeightedGKDDataset`
对每条记录调 `_encode`：
1. 构 prompt：`apply_chat_template([sys: EPISODE_EVAL_SYSTEM, user: problem], add_generation_prompt=True)` → `prompt_ids`
2. tokenize student_response，**带 `return_offsets_mapping`**（这是关键 —— 把每个 token 映射回字符 offset）
3. **token → segment 分配**：双指针扫，每个 token 的 `tok_start` 落在哪段就拿那段的 weight
4. 末尾 append `<|im_end|>` token，权重固定 1.0（终止符与内容无关）
5. 拼接：
   ```
   input_ids = prompt_ids + response_ids + [im_end]
   labels    = [-100]*len(prompt_ids) + response_ids + [im_end]
   weights   = [0.0]*len(prompt_ids)  + tok_weights  + [1.0]
   ```
6. 长度 > `max_length` 截断；若截到 labels 全 -100 则丢弃

#### 5.2 Collator
batch 内动态 pad，`weights` 列也 pad（pad 处填 0）。

#### 5.3 Loss：`weighted_jsd`
逻辑（process_reward_weighted_distill.py:633-692）：
```python
s_logits = student_logits[:, :-1, :]   # 用 i 处 logits 预测 i+1
t_logits = teacher_logits[:, :-1, :]
sh_lbl   = labels[:, 1:]
sh_wts   = weights[:, 1:]

mask = sh_lbl != -100
s = s_logits[mask]    # [n, V]
t = t_logits[mask]
w = sh_wts[mask]       # [n]
```

按 `chunk_size=512` token 分块（防显存爆），每块：

| β | 公式 |
|---|---|
| β = 0 | `KL(student_log || teacher_log)` per token，加权求和 |
| β = 1 | `KL(teacher_log || student_log)` per token，加权求和 |
| 0<β<1 | `m = logsumexp(s + log(1-β), t + log(β))`<br>`jsd = β·KL(m||t) + (1-β)·KL(m||s)` |

最终 `total / n`（n 是有效 token 数）。

⚠️ **重要细节**：teacher forward 用的是 `student_input_ids`（即 student 自己产出的 rollout）。所以 teacher 是被 student 的前缀 condition 出来的，这是「在 student 路径上做 token 级蒸馏」的本质。负 PR 段下权就是为了不在「错误前缀」上加力度学 teacher。

#### 5.4 Trainer 配置
- `num_train_epochs=1`（每个 outer epoch 内只过一遍数据）
- `lr_scheduler_type=constant_with_warmup`（warmup 到 peak 后保持）
- `gradient_checkpointing=True`，`use_reentrant=False`
- `bf16=True`，`save_strategy=no`（手动 `save_model(output_dir)`）
- `remove_unused_columns=False`（保留 `weights` 列）
- `seed = SEED + epoch_idx`（每 epoch 不同 shuffle）

teacher 在 trainer init **之后**才 load，确保 LOCAL_RANK 已设置好；teacher 放在每 rank 自己的 device，frozen + eval 模式。

保存：`save_model(output_dir)` + `tokenizer.save_pretrained` + `_copy_args_json`（拷 SWIFT 兼容用的 args.json）。

### Step 6：`_epoch_eval_worker`（vLLM 评测子进程）

加载训练完的新 ckpt：

- **Test eval**（仅 `TEST_SET_JSONL` 存在时）：
  - 满足 `(epoch+1) % TEST_EVAL_EVERY == 0` 或最后一 epoch
  - 中间 epoch：从 `test_records` 中按 SEED 子采样到 `TEST_EVAL_SAMPLES` 条
  - 最后一 epoch：用全量
- **Train eval**（始终跑）：
  - skip 已经解出的题（来自 `train_eval_solved.json`）
  - 解出新的，索引追加进 solved 集合
- 评测都是 temperature=0、`build_filter_prompt`（free-thinking）、命中标准 = `extract_boxed_answer + answers_match`

写 `eval_out.json`，orchestrator 拿到后追加一条到 `eval_log.jsonl`：
```json
{
  "epoch": 3,
  "test_correct": 348, "test_total": 500, "test_acc": 0.696,
  "train_correct": 412, "train_total": 1000, "train_acc": 0.412,
  "newly_solved_train": 39,
  "n_train_records": 987
}
```

---

## 五、Resume 与缓存

`_find_latest_checkpoint` 扫 `checkpoints/epoch_NNN/config.json`，找最大的 NNN，从 `last_done_idx + 1` 续跑。

每个 epoch 每个 step 都有缓存检查：

| 文件 | 存在则跳过 |
|---|---|
| `epoch_NNN_inf_out.json` + `epoch_NNN_train.jsonl` | Step 0-4 推理 |
| `checkpoints/epoch_(N+1)/config.json` | Step 5 训练 |
| `epoch_NNN_eval_out.json` | Step 6 评测 |

—— 任何 step 跑挂了重启都能从断点继续。

---

## 六、目录结构

```
OUTPUT_DIR/
├── pool.jsonl                       # 静态 T✓ 池，{problem, gt_answer}
├── test_set.jsonl                   # MATH 才有
├── train_eval_set.jsonl             # = pool 快照（用于 in-distribution 评测）
├── train_eval_solved.json           # 已解出题的索引集合
├── eval_log.jsonl                   # 每 epoch 一条评测记录
├── epoch_000_inf_cfg.json
├── epoch_000_inf_out.json
├── epoch_000_train.jsonl            # 该 epoch 的加权训练数据
├── epoch_000_train_cfg.json
├── epoch_000_eval_cfg.json
├── epoch_000_eval_out.json
├── ...
└── checkpoints/
    ├── epoch_001/                   # 第 1 epoch 训完的 student
    │   ├── config.json
    │   ├── *.safetensors
    │   ├── tokenizer.json ...
    │   └── args.json                # 从 student_ckpt 拷过来（SWIFT 兼容）
    ├── epoch_002/
    └── ...
```

---

## 七、数据集切换

### MATH（默认）
```bash
python tools/process_reward_weighted_distill.py
```
全套流程包含 test eval。

### DAPO-Math-17k
```bash
POOL_DATASET=dapo_math_17k \
OUTPUT_DIR=$PWD/tools/weighted_distill_output_dapo \
    python tools/process_reward_weighted_distill.py
```

注意：
- 必须换 `OUTPUT_DIR`（否则会 reuse MATH 的 `pool.jsonl` 缓存）
- DAPO 没 test split，`eval_log.jsonl` 中 `test_acc` 始终为 None
- AIME 类评测用 evalscope 在 `checkpoints/epoch_NNN/` 上单独跑
- `_load_dapo_samples_normalized` 兼容 `prompt`/`messages`/`problem` 多种字段；gt 优先 `reward_model.ground_truth`，回退到 row 顶层的 `ground_truth`/`answer`/`solution`/`gt_answer`
- **HF 上的 DAPO-Math-17k 文件实际有 1,791,700 行，但只有 17,398 个独立问题**（每题约重复 103 次，推测是为 RL 训练的 multi-rollout 格式预生成的）。loader 里按 `problem` 文本去重，pool 落到 ~17k 上。
- 抽取失败时（0 样本）会打印 schema 和首行让你 debug

---

## 八、CLI dispatch

`__main__` 根据 argv 分派：

| 模式 | 命令 |
|---|---|
| Orchestrator | `python process_reward_weighted_distill.py` |
| Inference 子进程 | `python process_reward_weighted_distill.py --_epoch-worker <cfg> <out>` |
| Eval 子进程 | `python process_reward_weighted_distill.py --_eval-worker <cfg> <out>` |
| Train 子进程 | `(torchrun ...) process_reward_weighted_distill.py --_train-worker <cfg>` |

子进程模式都从 cfg JSON 读参数，写结果到 out JSON / `train.jsonl` / `checkpoints/epoch_NNN/`。

---

## 九、设计要点回顾

1. **静态池**：避免「学会 → 剔除 → 越学越难 → 池子被难题主导」的偏移
2. **段权重而非样本权重**：信号粒度细到 token 级，能精准下权某段「check-then-change」式的破坏性推理
3. **`weight = 1 + α·pr` 而非 `1 + α·max(0,-pr)`**：原版的 `max(0,-pr)` 会在 student 走错的前缀上**加大**JSD 力度，而 teacher 在该前缀下的条件分布很可能也是同款 check-then-change，反而强化坏行为；新公式直接下权坏段、上权好段，跟「不学坏样本」一致
4. **PR baseline 用 empty-prefix**：学生不思考直接答的命中率，作为「思考的边际收益」基准
5. **monotonic merge**：同方向的 PR 微抖动合并，减少「同质段反复出现」的噪声，并让长段的累积影响更显著（合并后 pr 是从 empty_acc 起的累积量）
6. **三子进程隔离**：vLLM 和 HF Trainer 之间显存/CUDA 状态干净切换
7. **断点续跑**：每个 step 都缓存中间产物
