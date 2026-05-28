# `process_reward_consistency_opd.py` 技术报告

> 文档版本：基于 `process_reward_consistency_opd.py`（约 3000 行）  
> 路径：`chenyang/opd/tools/process_reward_consistency_opd.py`

---

## 目录

1. [概述](#1-概述)
2. [核心假设](#2-核心假设core-hypothesis)
3. [六项关键设计选择](#3-六项关键设计选择)
4. [整体技术路线](#4-整体技术路线)
5. [Phase 0：静态 T✓ 样本池](#5-phase-0静态-t-样本池)
6. [每 Epoch 七步流水线](#6-每-epoch-七步流水线)
7. [一致性评分算法详解](#7-一致性评分算法详解)
8. [Retry 与 Workset 机制](#8-retry-与-workset-机制)
9. [并行与工程架构](#9-并行与工程架构)
10. [主要输出文件](#10-主要输出文件)
11. [关键超参数](#11-关键超参数)
12. [与相关脚本的关系](#12-与相关脚本的关系)
13. [方法动机与预期行为](#13-方法动机与预期行为)
14. [使用方式](#14-使用方式)
15. [总结](#15-总结)

---

## 1. 概述

`process_reward_consistency_opd.py` 是一个 **面向 On-Policy Distillation (OPD)** 的完整训练流水线脚本。其核心目标不是直接优化 Process Reward (PR)，而是利用 PR 与 KL 散度之间的 **序关系（ordering）** 来识别 **不可靠的 KL 监督信号**，从而在 OPD 训练中：

- **保留** 一致性高的 response 用于训练；
- **屏蔽（mask）** 段落级不一致的 token，使其不参与 loss；
- **丢弃或重试** 一致性低且答错的样本。

脚本实现了从数据池构建 → 学生 rollout → 过程奖励评估 → 师生 KL 计算 → 一致性筛选 → OPD 训练 → 评估 的 **多 epoch 闭环**，并针对大 teacher（如 32B）做了 **Teacher Top-K 磁盘缓存**，训练阶段 GPU 上只加载 student。

**依赖关系**：仅从 `process_reward_distill.py` 复用函数（答案匹配、分段、K-fold prompt 等），所有超参与路径在本脚本内独立定义，避免基线脚本变更产生隐式影响。

复用的函数包括：

- `extract_boxed_answer`, `answers_match`
- `segment_response_text`, `merge_monotonic_runs`
- `write_jsonl`, `load_jsonl`
- `build_filter_prompt`, `build_episode_eval_prompt`
- `make_llm`, `destroy_llm`, `safe_generate`, `run_isolated_rollout`

---

## 2. 核心假设（Core Hypothesis）

在同一条 student response 内，若某段推理的 **Process Reward 更高**，则 student 与 teacher 在该段上的分布应 **更一致**（即 **KL 更低**）。

当这一序关系被违反时，该段上的 KL 监督对 **当前这条 response** 不可靠，应在 OPD loss 中被 **降权（置零）**。

形式化地，对相邻 rank 对 \((A, B)\)，其中 \(PR_A > PR_B\)：

```
violation(A, B) = max(0, KL_A - KL_B) × max(0, PR_A - PR_B)
```

段落的不一致性分数 = 其参与的所有 violation 之和。

**一致性判定**（邻接对）：高 PR 段的 KL 应 ≤ 低 PR 段的 KL；KL 相等视为一致，不产生 violation。

---

## 3. 六项关键设计选择

| # | 设计 | 说明 |
|---|------|------|
| 1 | **不做 PR–KL 回归/线性拟合** | PR 噪声大（K=8 时 std≈0.25），KL 量级随推理阶段漂移；只关心 **排序** |
| 2 | **Per-response 邻接对比较** | 按 PR 降序排列 merged 段，只看相邻 rank 对 (1-2, 2-3, …)，类似 Kendall-Tau |
| 3 | **数据集级 response 筛选** | 全局按 consistency ratio 排序，保留 top X%；其余按正误分流 |
| 4 | **段级 mask，非删除** | 不一致段 token weight=0，仍保留在序列中作上下文；labels 不为 -100 |
| 5 | **Top-K support KL** | 离线 KL 与训练 loss 均在 teacher Top-K 支撑集上重归一化后算 KL |
| 6 | **Teacher Top-K 磁盘缓存** | Step 4 前向 teacher 后序列化 indices+logits；Step 6 训练不加载 teacher |

---

## 4. 整体技术路线

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 0: 静态池构建                                             │
│  加载 MATH/DAPO → Teacher greedy rollout → 保留 T✓ → pool.jsonl │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  每个 Epoch (× N_EPOCHS)                                         │
│                                                                  │
│  Workset = pool + retry_pool                                     │
│       │                                                          │
│       ▼                                                          │
│  Step 1-3  Student vLLM: rollout + 分段 + K-fold PR             │
│       │                                                          │
│       ▼                                                          │
│  Step 4      HF Student+Teacher: Top-K KL + teacher cache       │
│       │                                                          │
│       ▼                                                          │
│  Step 5      CPU: 一致性评分 + 筛选 + mask + retry 路由          │
│       │                                                          │
│       ▼                                                          │
│  Step 6      torchrun: OPD 训练 (cached top-K KL + weights)     │
│       │                                                          │
│       ▼                                                          │
│  Step 7      Student vLLM: held-out + train-eval                │
│       │                                                          │
│       ▼                                                          │
│  删除 epoch teacher cache → 更新 checkpoint → 下一 epoch         │
└─────────────────────────────────────────────────────────────────┘
```

**训练 KL 目标**（在 teacher Top-K 支撑集 \(S_t\) 上重归一化）：

```
L = β · KL(p_s || p_t) + (1 - β) · KL(p_t || p_s)
```

默认 `β = OPD_REVERSE_KL_BETA = 0.7`（reverse-leaning，mode-seeking）。

---

## 5. Phase 0：静态 T✓ 样本池

**函数**：`phase0_build_pool()`

**流程**：

1. 若 `pool.jsonl` 已存在则直接加载（可断点复用）。
2. 从数据集加载题目：
   - **hendrycks_math**（别名 `math`）：本地 7 个子集 `load_from_disk`，确定性 shuffle + holdout 测试集（默认 500 题）。
   - **dapo_math_17k**：从 Arrow shard 读取（需 `DAPO_SHARDS` 环境变量）。
3. **Teacher 贪心过滤**（temperature=0）：只保留 teacher `\boxed{}` 答案与 GT 匹配的题目 → 静态 pool。
4. 写出 `test_set.jsonl`（MATH holdout）、`train_eval_set.jsonl`（pool 全量，用于训练集内 eval）。

**意义**：保证 pool 中每题 teacher 可解，OPD 监督有上界参考；pool 在整个训练过程中 **冻结**。

**相关函数**：

- `_load_pool_samples_normalized()`：数据集入口
- `_load_math_local_split()`：MATH 本地加载与 train/test 划分
- `_load_dapo_samples_normalized()`：DAPO Arrow shard 加载

---

## 6. 每 Epoch 七步流水线

Orchestrator 函数：`run_epoch(epoch_idx, current_ckpt, pool)`

### Step 1–3：Student Worker（`_epoch_student_worker`）

**并行**：每 GPU 一个 vLLM worker（DP，TP=1），modulo 分片，`_orig_idx` 保证合并顺序。

| 子步 | 内容 |
|------|------|
| **Step 1 Rollout** | 当前 student checkpoint 对 workset 每题生成完整推理（max 8192 tokens, T=0.7） |
| **Step 2 分段** | `segment_response_text()`：触发词 + 句号计数切 episode |
| **Step 3 K-fold PR** | 对每个 segment endpoint 做 K=8 次 forced-answer eval，得 `acc_rate`；另测 empty prefix baseline |

**Process Reward 定义**（per-segment）：

```
PR_i = acc_rate(prefix_i) - acc_rate(prefix_{i-1})
```

首段 prior 为 `empty_acc`（空 prefix 的 K-fold 准确率）。

**Rollout 输出 schema**（`epoch_NNN_rollouts.jsonl`）：

```json
{
  "work_idx": 0,
  "problem": "...",
  "gt_answer": "...",
  "retry_count": 0,
  "first_epoch": -1,
  "source": "pool",
  "response_text": "...",
  "response_token_ids": [1, 2, 3],
  "student_correct": false,
  "student_answer": "...",
  "segments": [{"start_char": 0, "end_char": 100}],
  "ep_accs": {"0": 0.625, "1": 0.75},
  "empty_acc": 0.125
}
```

Skipped 记录（空 response / 无 segments）在后续路由中视同低一致性错题。

---

### Step 4：KL Worker（`_epoch_kl_worker`）

**并行**：`N_GPUS // KL_GPUS_PER_WORKER` 个 worker；大 teacher 可 TP 多卡。

**Token 对齐策略**（保证 on-policy）：

- prompt：`tokenize(build_filter_prompt(problem))`
- response：直接使用 rollout 的 `response_token_ids`（vLLM 精确输出）

**Per-token KL**（response 每个位置 \(t\)）：

1. 取 teacher logits 的 Top-K 索引 \(S_t\)
2. gather student logits 于 \(S_t\)
3. 在 K 维支撑集上 log_softmax + softmax
4. 计算 forward KL：`KL(q_hat || pi_hat)` 与 reverse KL：`KL(pi_hat || q_hat)`
5. 混合：`normKL_mix = β·KL_rev + (1-β)·KL_fwd`

**Merged Paragraph 构建**（`_build_merged_paragraphs`）：

1. 计算 per-segment PR（相邻 acc_rate 差分）
2. `merge_monotonic_runs()`：将 PR 单调 run 合并（仅严格升降翻转处切分）
3. merged 段 PR = `acc_end - acc_prior`（累积 ΔAcc）
4. 按字符范围映射 token，**长度加权平均** 得段落级 `normKL_fwd/rev/mix`
5. 记录 `tok_start`, `tok_end` 供 Step 5 mask 使用

**Teacher Cache**（`epoch_NNN_teacher_topk_cache.pt`）：

每条 rollout 存：

```python
{
  "topk_indices": Tensor[resp_len, K],   # int32
  "topk_logits":  Tensor[resp_len, K],   # bfloat16, RAW（未除温度）
  "prompt_len":   int,
  "resp_len":     int,
  "topk":         int,
  "vocab_clip":   int,
}
```

训练时按 `cache_id`（= rollout 的 `_orig_idx`）索引。

---

### Step 5：一致性调度（`_consistency_dispatch`，CPU）

这是本脚本的方法核心，分四个阶段：

#### Phase a：Per-response 评分（`_score_response_paragraphs`）

1. 按 `(PR ↓, KL ↑)` 排序 merged 段落（PR 相同时 KL 升序，避免自比较产生虚假 violation）
2. 对相邻 rank 对检查：高 PR 段是否 `KL ≤` 低 PR 段
3. 计算 **consistency ratio** = 一致邻接对数 / 有效邻接对数
4. 累加 weighted violation 到各段 **inconsistency score**

**Eligible 条件**：≥ `MIN_MERGED_SEGS_PER_RESP`（默认 3）段，且至少 1 个有效 PR 对（PR 不完全相同）。

#### Phase b：全局排序

Eligible responses 按 `(consistency, n_pairs)` 降序，保留 top `TOP_PERCENT_RESPONSES%`（默认 30%）。

#### Phase c：Mask 与路由

**保留的 response（`type: consistency_opd`）**：

- 按 inconsistency 降序，最多 mask `ceil(MASK_TOP_PERCENT_SEGMENTS% × n_segs)` 段
- **仅 mask inconsistency > 0 的段**；全一致则 vanilla OPD（全 weight=1）
- 生成 `response_weights`：masked token → 0.0，其余 → 1.0
- 若全部 weight=0（退化情况），fall through 到 retry/discard 路由

**未保留 response 路由**：

| 条件 | 动作 |
|------|------|
| 段数不足但非 skipped | `vanilla_opd_fallback`（仍训练，不 mask，weight 全 1） |
| 答对 + 低一致性 | 写入 `discarded_correct.jsonl`（已会，丢弃） |
| 答错 + 低一致性 | `retry_count++`，入 retry_pool；≥ MAX_RETRIES 则永久 drop |
| skipped | 走 retry/discard 路径（视同错题） |

**Train record schema**（`epoch_NNN_train.jsonl`）：

```json
{
  "type": "consistency_opd",
  "problem": "...",
  "gt_answer": "...",
  "response_token_ids": [1, 2, 3],
  "response_weights": [1.0, 1.0, 0.0],
  "cache_id": "42",
  "n_paragraphs": 5,
  "n_masked": 1,
  "consistency": 0.75,
  "n_pairs": 4,
  "paragraphs": [...]
}
```

#### Phase d：统计与诊断

输出 `epoch_NNN_dispatch_stats.json`，包含：

- 保留/丢弃/retry 数量
- mask 预算利用率
- consistency / inconsistency 分布（p50/p75/p90/...）
- `frac_kept_paragraphs_zero`：若 ≈1，说明 masking 基本未激活

---

### Step 6：OPD 训练（`_train_worker`）

**启动**：`torchrun --nproc_per_node=N_GPUS`（单卡则直接 subprocess）

**数据集**（`ConsistencyDataset`）：

- `input_ids` = prompt + response + im_end
- `labels`：prompt 与 im_end 为 -100；response token 保留 id（供 forward）
- `weights`：prompt/im_end=0；response 按 mask 设 0/1
- 加载 Step 4 teacher cache，按 `cache_id` 对齐

**Loss**（`OPDConsistencyTrainer.compute_loss`）：

对每个 active position（label ≠ -100）：

```
L_t = β · KL(p̂_s || p̂_t) + (1-β) · KL(p̂_t || p̂_s)
L   = Σ(w_t · L_t) / Σ(w_t)
```

其中 `p̂` 在 teacher Top-K 支撑集上重归一化，温度 `OPD_TEMP`。

**训练配置**：

- bf16、gradient checkpointing
- `constant_with_warmup` LR schedule（外层 epoch 级 cosine/linear 衰减）
- 每 epoch 1 pass，`save_strategy=no`
- Flash Attention 2（可用时）或 SDPA

**关键优势**：32B teacher 仅在 Step 4 前向；Step 6 每 rank 约 ~7–8 GB CPU RAM 读 cache，GPU 只跑 student。

---

### Step 7：评估（`_epoch_eval_worker`）

- **Test set**：MATH holdout（greedy, T=0）
- **Train eval**：pool 题目，跳过已 solved（`train_eval_solved.json`）
- 记录 `eval_log.jsonl`（含 dispatch stats、test/train acc）
- Epoch 结束后 **删除 teacher cache** 控制磁盘（约 10–20 GB/epoch）

---

## 7. 一致性评分算法详解

### 7.1 数据流

```
原始 segments + ep_accs
        │
        ▼
计算 per-segment PR（acc 差分）
        │
        ▼
merge_monotonic_runs（单调 run 合并）
        │
        ▼
Merged paragraphs + normKL（来自 Step 4）
        │
        ▼
按 PR↓, KL↑ 排序
        │
        ▼
相邻 rank 对比较 → violation 累加
        │
        ▼
consistency ratio + per-paragraph inconsistency
```

### 7.2 核心函数

**`_violation(pr_a, kl_a, pr_b, kl_b)`**

- A 为更高 PR rank 的段
- 仅当 `PR_A > PR_B` 且 `KL_A > KL_B` 时产生非零 violation
- `violation = (PR_A - PR_B) × (KL_A - KL_B)`

**`_score_response_paragraphs(paras)`**

- 输入：单 response 的 merged paragraphs（含 `process_reward`, `normKL`）
- 输出：`(inconsistency[], consistency_ratio, n_pairs)`
- 少于 2 段：consistency=NaN，不参与 eligible 筛选

### 7.3 设计 rationale

**为何用邻接对而非全局 Kendall-Tau？**

- 计算简单、可解释
- 只惩罚 **局部序关系颠倒**，对 PR/KL 绝对量级不敏感
- violation 需 **PR gap 与 KL 反转同时显著** 才大，抑制噪声

**为何 monotonic merge？**

- 相邻 segment 的 PR 若在单调 run 内波动，视为同一"推理阶段"
- 减少碎片化，使段落级 PR/KL 更稳定
- merged PR 用累积 ΔAcc，而非单段 delta

**为何 K=8（而非基线的 K=4）？**

- PR 噪声约 std≈0.25（K=8）
- 本方法依赖 **排序信号**，需要更高 SNR 的 PR 估计

---

## 8. Retry 与 Workset 机制

### 8.1 状态流转

```
Pool（静态）
    │
    ▼
每 epoch Workset = pool ∪ retry_pool（按 problem+gt 去重，retry 优先）
    │
    ├── top 30% consistency ──► Train（consistency_opd + mask）
    │
    └── bottom 70%
            ├── student 答对 ──► Discarded（已掌握）
            └── student 答错 ──► retry_count++
                    ├── < MAX_RETRIES ──► 下 epoch 重 rollout
                    └── ≥ MAX_RETRIES ──► 永久 drop
```

### 8.2 关键函数

- `_build_workset_for_epoch(pool, retry_pool, epoch_idx)`：合并 workset
- `_load_retry_pool()` / `_save_retry_pool()`：retry backlog 持久化
- `_append_dropped_retries()` / `_append_discarded()`：日志追加

Retry 键：`(problem, gt_answer)`，与 pool 索引无关。

---

## 9. 并行与工程架构

| 阶段 | 并行模式 | GPU 分配 |
|------|----------|----------|
| Step 1-3 Student | DP，1 GPU/worker | vLLM TP=1 |
| Step 4 KL | DP，`KL_GPUS_PER_WORKER` 卡/worker | Student GPU0, Teacher 其余（可 TP） |
| Step 6 Train | DDP torchrun | 每 rank 全卡 student |
| Step 7 Eval | DP，同 Step 1-3 | vLLM TP=1 |

**分片策略**：modulo sharding（`i % n_shards == shard_idx`），负载均衡优于连续分片（因 skipped 记录处理极快）。

**断点续训**：

- 各 step 产出中间文件（rollouts / kl / train.jsonl / checkpoint），存在则 skip
- `_find_latest_checkpoint()` 从 `checkpoints/epoch_NNN/` 恢复
- Phase 0 pool 亦缓存于 `pool.jsonl`

**子进程入口**（CLI dispatch）：

```bash
python process_reward_consistency_opd.py --_student-worker <cfg> <out>
python process_reward_consistency_opd.py --_kl-worker       <cfg> <out>
python process_reward_consistency_opd.py --_train-worker    <cfg>
python process_reward_consistency_opd.py --_eval-worker     <cfg> <out>
```

---

## 10. 主要输出文件

| 文件 | 含义 |
|------|------|
| `pool.jsonl` | Teacher 过滤后的静态训练池 |
| `test_set.jsonl` | MATH holdout 测试集 |
| `train_eval_set.jsonl` | Pool 全量，用于训练集内 eval |
| `train_eval_solved.json` | 已 solved 的 train-eval 索引 |
| `retry_pool.jsonl` | 当前 epoch 待重试 backlog |
| `dropped_retries.jsonl` | 超过 MAX_RETRIES 永久丢弃 |
| `discarded_correct.jsonl` | 答对但低一致性（无需再训） |
| `epoch_NNN_workset.jsonl` | 本 epoch 待 rollout 题目 |
| `epoch_NNN_rollouts.jsonl` | Student rollout + PR eval |
| `epoch_NNN_kl.jsonl` | 段落级 normKL + merged paragraphs |
| `epoch_NNN_teacher_topk_cache.pt` | Teacher Top-K 缓存（epoch 末删除） |
| `epoch_NNN_train.jsonl` | 训练样本 + weights + cache_id |
| `epoch_NNN_dispatch_stats.json` | 筛选/mask 统计 |
| `checkpoints/epoch_NNN/` | Student checkpoint |
| `eval_log.jsonl` | 各 epoch 准确率日志 |

默认输出目录：`tools/consistency_opd_output/`（可通过 `OUTPUT_DIR` 覆盖）。

---

## 11. 关键超参数

| 环境变量 | 默认 | 作用 |
|----------|------|------|
| `STUDENT_MODEL` | `Qwen/Qwen3-1.7B` | 学生模型 |
| `TEACHER_MODEL` | `Qwen/Qwen3-32B` | 教师模型 |
| `TOP_PERCENT_RESPONSES` | 30 | 保留 consistency 最高的 response 比例 (%) |
| `MASK_TOP_PERCENT_SEGMENTS` | 20 | 每 response 最多 mask 的段落比例上限 (%) |
| `MIN_MERGED_SEGS_PER_RESP` | 3 | 参与 consistency 筛选的最小段落数 |
| `MAX_RETRIES` | 3 | 低一致性答错题最大重试 epoch 数 |
| `K_EVAL` | 8 | 每 segment PR 的 K-fold 次数 |
| `KL_TOPK` | 64 | Teacher Top-K 支撑集大小 |
| `KL_GPUS_PER_WORKER` | 1 | 每个 KL worker 占用 GPU 数 |
| `OPD_REVERSE_KL_BETA` | 0.7 | Reverse-leaning KL 混合系数 β |
| `OPD_TEMP` | 1.0 | KL 计算 softmax 温度 |
| `N_EPOCHS` | 5 | 外循环 epoch 数 |
| `GKD_LR` | 1e-6 | 基础学习率 |
| `GKD_BATCH_SIZE` | 1 | 每 device batch size |
| `GKD_GRAD_ACCUM` | 4 | 梯度累积步数 |
| `GKD_MAX_LEN` | 9216 | 训练最大序列长度 |
| `GKD_LR_SCHEDULE` | cosine | 外层 epoch LR 调度（cosine/linear/constant） |
| `MAX_NEW_TOKENS` | 8192 | Rollout 最大生成长度 |
| `MAX_ANSWER_TOKENS` | 384 | K-fold eval 最大答案长度 |
| `ROLLOUT_TEMPERATURE` | 0.7 | Rollout 采样温度 |
| `INFERENCE_BATCH_SIZE` | 200 | vLLM 批大小（0=整批） |
| `N_SAMPLES` | 0 | 子采样数量（0=全量） |
| `POOL_DATASET` | hendrycks_math | 或 dapo_math_17k |
| `LOCAL_MATH_DATA_DIR` | 见代码 | 本地 MATH 数据路径 |
| `LOCAL_MATH_TEST_HOLDOUT` | 500 | MATH 测试集 holdout 数量 |
| `OUTPUT_DIR` | tools/consistency_opd_output | 输出目录 |
| `SEED` | 42 | 随机种子 |

---

## 12. 与相关脚本的关系

```
process_reward_distill.py              ← 基线 OPD/PR 蒸馏（复用工具函数）
        │
        ├── process_reward_weighted_distill.py    （加权蒸馏变体）
        ├── process_reward_correction_distill.py  （纠错蒸馏变体）
        └── process_reward_consistency_opd.py     ← 本脚本
                PR-KL 一致性过滤 + segment mask
```

| 维度 | `process_reward_distill` | `process_reward_consistency_opd` |
|------|--------------------------|----------------------------------|
| PR 用途 | 直接加权/筛选 | 仅作 **排序探针** |
| KL 计算 | 可能全词表或 top-K | 统一 Top-K support |
| 不可靠信号处理 | 依具体变体 | Response 筛选 + token mask |
| Teacher 训练时 | 可能在线前向 | **磁盘缓存**，训练不加载 |
| K-fold | K=4 | K=8（更高 SNR） |
| Retry 机制 | 无/依变体 | 内置 MAX_RETRIES 闭环 |

本脚本在 OPD 框架上增加了 **可靠性门控（reliability gating）**：

- 不是改 loss 形式，而是 **选哪些 response 训、哪些 token 参与 loss**
- PR 仅作 **序信号**，KL 仅在与 PR 一致时才被信任
- 筛选标准（normKL_mix）与训练 loss 使用相同的 β 混合

---

## 13. 方法动机与预期行为

### 13.1 解决的问题

On-policy 场景下，student 某段推理看似"过程奖励高"，但 teacher 分布与该段并不更一致——此时强行做 KL 蒸馏会引入 **错误监督**。

### 13.2 预期效果

1. 训练集偏向 **PR-KL 序关系自洽** 的 rollout
2. 即使保留的 response 内，仍 **屏蔽局部不一致段落**
3. 答错且整体不可靠的题 **多 epoch 重采样**，而非用坏 signal 硬训
4. 答对但不可靠的题 **直接丢弃**（已掌握，无需 OPD）

### 13.3 诊断指标

运行时应关注 `dispatch_stats` 中的：

| 指标 | 含义 | 异常信号 |
|------|------|----------|
| `n_kept_with_actual_masks` | 真正做了 mask 的保留 response 数 | =0 则 masking 未激活 |
| `frac_kept_paragraphs_zero` | 保留 response 中 inconsistency=0 的段落比例 | ≈1 则等价于 top-X% + vanilla OPD |
| `mask_budget_utilisation` | 实际 mask 数 / 理论预算 | 过低说明 violation 稀少 |
| `kept_consistency` vs `eligible_consistency` | 保留 vs 全部 eligible 的 consistency 分布 | 检查筛选是否有效 |
| `n_vanilla_fallback` | 段数不足 fallback 到 vanilla OPD 的数量 | 过高说明分段/merge 产出偏短 |

---

## 14. 使用方式

### 14.1 基本运行

```bash
cd /path/to/chenyang/opd
python tools/process_reward_consistency_opd.py
```

### 14.2 DAPO-Math-17k

```bash
OUTPUT_DIR=$PWD/tools/consistency_opd_dapo \
POOL_DATASET=dapo_math_17k \
DAPO_SHARDS=/path/to/shard1:/path/to/shard2 \
python tools/process_reward_consistency_opd.py
```

### 14.3 常用调参示例

```bash
# 更激进筛选：只保留 top 20%，mask 30% 段落
TOP_PERCENT_RESPONSES=20 \
MASK_TOP_PERCENT_SEGMENTS=30 \
python tools/process_reward_consistency_opd.py

# 大 teacher KL 阶段用 2 卡 TP
KL_GPUS_PER_WORKER=2 \
TEACHER_MODEL=Qwen/Qwen3-32B \
python tools/process_reward_consistency_opd.py

# 子采样 1000 题快速验证
N_SAMPLES=1000 N_EPOCHS=2 \
python tools/process_reward_consistency_opd.py
```

### 14.4 断点续训

直接重新运行同一命令即可。脚本会：

1. 加载已有 `pool.jsonl`
2. 检测 `checkpoints/epoch_NNN/` 最新 checkpoint
3. Skip 已完成的 step 中间产物

---

## 15. 总结

`process_reward_consistency_opd.py` 实现了一条 **"Process Reward 引导的 OPD 可靠性过滤"** 完整链路：

| 层次 | 内容 |
|------|------|
| **信号** | K-fold 答案准确率差分 → Process Reward；Teacher Top-K KL → 分布一致性 |
| **判据** | 同 response 内 PR 高者应 KL 低（邻接对序关系） |
| **动作** | Response 级 top-X% 筛选 + 段落级 token mask + 错题 retry |
| **训练** | Cached Top-K OPD loss，支持大 teacher、多 GPU 流水线 |
| **闭环** | 多 epoch 滚动，pool 静态、checkpoint 递进、eval 持续监控 |

该方法的核心创新在于 **把 PR 从"回归目标"降级为"排序探针"**，并与 OPD 的 KL 目标在同一 Top-K 支撑集上对齐，使筛选标准与训练 loss 一致，从而构成一套可工程化落地的 on-policy 蒸馏质量控制体系。

---

*文档生成日期：2026-05-27*
