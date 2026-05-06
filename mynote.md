# process_reward_distill.py 笔记

## 算法概览

**Process-Reward Guided On-Policy Distillation**(过程奖励引导的在线策略蒸馏)。
核心思路:让学生模型(Qwen3-1.7B)在做不对的题上一边自己尝试、一边从教师模型(Qwen3-8B)那里学,
但**不是整条答案照抄**——而是用"过程奖励"定位学生自己推理链中真正出错的那一小段,只针对坏段做蒸馏,
其余对的部分保留学生自己的风格。

角色:
- **Student(S)**:当前被训练的小模型,每轮都在更新。
- **Teacher(T)**:固定的大模型,只在 phase 0 做一次 rollout,之后作为回退蒸馏目标。

数据单位:
- **Sample / 条目**:一个 `(problem, gt_answer, student_response, teacher_response, teacher_correct)` 元组。
- **Episode / 段**:把 `student_response` 按触发词(`wait`/`alternatively`/`let me check`/`hmm` 等)切成的连续片段。
- **Process reward(过程奖励)**:某个 episode 的 accuracy rate 相对上一段的增量 `Δacc = acc_i − acc_{i-1}`。
  - `pr > 0`:这段让解题更有把握 → 好段
  - `pr < 0`:这段把推理带偏了 → 坏段,正是要重点蒸馏的地方
  - `pr = 0`:没变化

## 整体流程

### Phase 0 — 构建初始样本池(只跑一次,结果缓存到 `sample_pool.jsonl`)
1. 加载 hendrycks_math 训练集(7 个子集合并)。
2. **Student rollout**:学生以 `temperature=0.7` 对每题采样一次。能直接答对的丢掉(学生已掌握,不值得训)。
3. **Teacher rollout**:对剩下的 S✗ 样本让教师用 `temperature=0`(贪婪)再答一遍,记录 `teacher_correct ∈ {True, False}`。
   - 注意:**T✗ 也保留**,不丢。后面用 `teacher_correct` 作为分支依据。
4. 把 `test_set.jsonl`(官方 MATH test 5000 题)和 `train_eval_set.jsonl`(初始 pool 的快照,永不变)落盘用作后续评测。

### 每个 epoch(重复 N_EPOCHS 次)
所有 vLLM 推理都在一个**隔离子进程**里跑,退出后 CUDA 上下文彻底释放,保证训练子进程能拿到干净显存。

**Eval(held-out 评测)**
- `test_set.jsonl`:greedy 评测,按 `TEST_EVAL_EVERY` 节奏跑,最后一轮一定跑全量。
- `train_eval_set.jsonl`:增量评测,已经答对过的题跳过,新答对的记到 `train_eval_solved.json`,用来观察"初始 S✗ 池逐渐被攻克"的曲线。

**Step 1 — Episode 分段(CPU)**
对 pool 里每条的 `student_response` 按触发词 + 至少 3 个句号的规则切成多个 episode,拿到每段的 `end_char`。

**Step 2 — 单次 vLLM session 做三件事(开 prefix caching)**
请求排列刻意让同一 sample 相邻,让 APC KV 缓存命中最大化:
1. **2a Filter(greedy,T=0)**:学生对整题答一次。答对 → 说明学生最新版本已经会了 → 本条加入 `to_remove`。
2. **2b K-fold PR 评测(T=0.7,K=4)**:对每个 episode 前缀 + 空前缀,各采 K 次**强制收尾**(在 prefix 后拼 `EPISODE_EVAL_SUFFIX` 让模型立刻 `\boxed{}`),统计正确率作为 `acc_rate`。
   - 已被 filter 删掉的 sample 跳过 kfold(反正下游不消费)。
3. **2c Continuation(T=0.7,n=1)**:挑 top-`TOP_K_CONTINUATION` 且 `acc_rate ≥ ACC_RATE_THRESHOLD` 且 `pr > 0` 的 episode,让学生从该前缀继续自由生成。
   - 续写用**逐 prompt 动态 `max_tokens`**,保证 `prompt + new` 总预算不超过 phase-0 budget。

**Step 3 — 过程奖励 + 单调合并(CPU)**
- 每段:`process_reward = acc_i − acc_{i-1}`,`empty_acc_rate`(空前缀的 acc)作为起点。
- `merge_monotonic_runs`:把 PR 序列里**严格单增或严格单减的极大子段**合并成一个代表点(取末端)。flat(`Δ=0`)不触发切分。
- 好处:减少冗余段、抓住真正的拐点,训练样本更少更干净。

**Step 4 — 构建训练样本(三分支)**
对每条 pool sample:
- `max(acc_rate) > 0` → **PR-segment 训练**:
  - `TRAIN_MODE=neg_only`(默认):只取 `pr < 0` 的段。user = `problem + 该段之前的所有 prior_text`,assistant = 当前段文本。**只蒸馏这一小段**,让学生修正"坏拐点"。
  - `TRAIN_MODE=all`:所有 merged 段都纳入。
- `max(acc_rate) == 0` 且 `teacher_correct` → **整条蒸馏教师**:user = problem,assistant = `teacher_response`。学生完全答不到边,直接抄教师。
- `max(acc_rate) == 0` 且 `teacher` 也错 → **丢弃**(真正的无望题,教师都不会)。

**Step 5 — 落盘 `train_segments.jsonl`**

**Step 6 — `swift rlhf --rlhf_type gkd`(子进程)**
- `--lmbda 0`:完全 off-policy,直接用数据集里的 assistant response 当目标,不做 on-policy rollout。
- `--beta 1`:纯 JSD ≈ KL(S‖T),学生向教师分布对齐。
- 每轮产生一个新 checkpoint `checkpoints/epoch_XXX/`。

**Step 7 — 更新 pool**
- 删掉这轮 `to_remove`(学生已学会的)。
- 把 continuation 生成的 `(prefix + 新续写)` **没答对**的那些加回 pool,作为下一轮继续攻克的材料。答对的同样丢弃(因为 continuation 是从"好拐点"生成的,答对说明已经沿对路走通,没有再训的价值)。

## 这个算法的好处

1. **精细粒度定位学生的错误**
   传统 SFT / 蒸馏是把教师整条答案硬灌给学生,学生原本就会的部分也被覆盖,风格被抹平。
   这里用过程奖励找到**学生自己推理链里真正导致正确率下降的那一段**,只修那一段,其他对的部分保留。

2. **多分支样本利用,信息不浪费**
   - 学生答不到边但教师会 → 整条蒸馏教师。
   - 学生部分走对、某段走偏 → 段级 PR 训练。
   - 教师也不会 → 直接丢弃,不引入噪声标签。
   对比只用单一范式(要么全 SFT 要么全 RL),这种混合使用把每个样本的信息榨到最干净。

3. **On-policy + off-policy 的折中**
   Continuation 用的是**当前学生**从好前缀出发继续生成(on-policy 数据增强),
   而训练时 `lmbda=0` 纯 off-policy,稳定性高、便于分布式训练。
   既保留了 on-policy 的数据多样性,又规避了 on-policy 训练的不稳定性。

4. **自动课程(curriculum):难度随学生能力漂移**
   - 每轮 filter 把"已学会"的题从 pool 删掉 → pool 只剩还不会的。
   - Continuation 把"走到一半还没错"的前缀扩成新 pool 条目 → 下一轮从更深的位置重新评 PR。
   - 学生越强,要攻的拐点就越靠后;不需要人工分阶段课程,难度自适应。

5. **GPU 资源严格隔离,避免显存泄漏**
   - vLLM 和 `swift rlhf` 分别在独立子进程里跑,子进程退出 → CUDA 上下文完全销毁。
   - 比 `del llm + torch.cuda.empty_cache()` 更彻底,训练环节不会被推理阶段残留的显存挤爆。

6. **APC(automatic prefix caching)友好的请求编排**
   `[sys][user]` 前缀在 filter / kfold / continuation 三个阶段完全一致,且同一 sample 的请求在 mini-batch 内相邻,KV 缓存复用率最大化 → 推理 token 实际成本远低于请求数的朴素估计。

7. **单调合并降低训练冗余**
   `merge_monotonic_runs` 把连续单增/单减的段压成一个拐点代表,既保留了"走偏的位置",又把训练样本数控制住,避免对同一种错误重复训。

8. **内建评测与早停友好**
   每轮对 test set 和初始 train set 分别做 greedy 评测,`eval_log.jsonl` 保留完整曲线;`train_eval_solved` 增量跳过已解决题,保证 eval 成本不随学生变强线性增长。

## 每 mini-batch 打印的变量含义

每个 mini-batch 末尾会打印:
```
filter_removed=N1  kfold_prompts=N2  cont_prompts=N3  to_remove_total=N4
```

| 变量 | 含义 | 触发阶段 |
|------|------|----------|
| `filter_removed` | **本 mini-batch** 里学生 greedy rollout 能直接做对、被判为"已学会"的样本数 | Step 2a(filter,`temperature=0`,一次采样) |
| `kfold_prompts`  | **本 mini-batch** 送去做 K-fold 过程奖励评估的 prompt 总数 ≈ Σ(每个样本的 episode 数 + 1 条整条评估)。每个 prompt 还会被采 `K_EVAL=8` 次 | Step 2b(kfold,`temperature=0.7`) |
| `cont_prompts`   | **本 mini-batch** 里 `process_reward > 0` 的 episode prefix 数,基于这些 prefix 让学生续写;每 prompt 只采 1 次 | Step 2c(continuation,`temperature=0.7`) |
| `to_remove_total`| **累计**(跨所有已处理 mini-batch)被标记"已学会"、轮末要从 pool 删掉的样本数 | 全局,Step 2a 持续累加 |

## "删除" 的粒度

- `to_remove` 存的是 `segmented`(= 当轮 pool 的 snapshot)里的**单条索引 `si`**。
- 轮末重建 pool:`new_pool = [item for si, item in enumerate(pool) if si not in to_remove]` —— 只丢这一条,不是把整个原问题抹掉。
- **但是**:filter prompt 仅取 `item["problem"]` + `temperature=0`(确定性),所以
  **同一个原问题在 pool 里的多条条目**(一条来自 phase0 初始 rollout,其它来自过去几轮的续写)会**全部通过或全部不通过** filter,最终效果就是"同一原问题的所有相关条目一起删光"。
- 代码上这是隐式的,不是显式 group;每条各算各的 filter。

## `TRAIN_MODE=neg_only` 时 `assistant` 是什么?loss 怎么算?

### `assistant` = 学生自己原回答里的"坏段切片"

`process_reward_distill.py:1301-1319`:

```python
start_char   = segs[m_idx - 1]["end_char"] if m_idx > 0 else 0
prior_text   = item["student_response"][:start_char]
segment_text = item["student_response"][start_char:seg["end_char"]]   # ← "当前段文本"
user_content = item["problem"] + "\n" + prior_text if prior_text else item["problem"]
train_records.append({
    "messages": [
        {"role": "user",      "content": user_content},     # 题目 + 坏段之前所有 prior_text
        {"role": "assistant", "content": segment_text},     # 学生自己原回答里 pr<0 的那一段原文
    ],
})
```

**关键**:`assistant` 不是教师写的,不是真值答案,**是学生自己在 phase 0 / 上轮续写里生成过的、被判为过程奖励 < 0 的那段文本切片本身**。
乍看反直觉:为什么要用学生的"错误输出"作 target?答案在 loss 计算方式里。

### Loss 不是 token CE,是逐位置 KL(S‖T)

`--lmbda 0 --seq_kd false --beta 1` 的组合下,GKDTrainer 走 `ms-swift/swift/rlhf_trainers/gkd_trainer.py:585` 的 else 分支:

1. `_prepare_batch_inputs(inputs, encode_prompt_only=False)` 把 `[user, assistant]` 完整拼成 token 序列。
2. `labels` 里 user/system 部分填 `-100`(ignore),**只有 `segment_text` 对应的 token 位置保留真实 token id** → loss 只在这段 token 上算。
3. 学生 forward → 每个位置 `s_logits[t]`(vocab 维分布);教师 forward(冻结)→ `t_logits[t]`。
4. 对每个 `labels[t] != -100` 的位置,算 `generalized_jsd_loss(s_logits[t], t_logits[t], beta=1)`。

`gkd_trainer.py:791-794` 里 `beta=1` 的具体分支:

```python
elif beta == 1:
    jsd_chunk = F.kl_div(t_log_probs, s_log_probs, reduction='none', log_target=True)
```

`F.kl_div(input=log_q, target=log_p, log_target=True)` = `Σ exp(log_p) · (log_p − log_q)` = **KL(p ‖ q)**。这里 p = 学生分布,q = 教师分布,即:

$$\mathcal{L}_{\text{token}} = \mathrm{KL}(s \,\|\, t) = \sum_v s(v) \cdot \bigl(\log s(v) - \log t(v)\bigr)$$

这是**反向 KL / mode-seeking**:让学生分布收缩到教师分布的主模上。

### `segment_text` 的真实角色

1. **定义条件上下文**:第 t 个位置学生和教师都看到 `[user_content][segment_text[:t]]`,即"学生原先写错的这条轨迹上,前 t−1 个 token"。**学生并不是在学"输出这段文本"**,而是被强制复盘"在这条错误轨迹的每一步,我下一个 token 的分布应该是什么"。
2. **定义 loss 覆盖范围**:label mask 只保留这些 token 位置,KL 只在这里算。

### 直观解释

> 把学生原回答里的 bad segment 当成一段固定的"考试场景",在每一步让学生 re-distribute,使分布向教师分布主模靠拢。训练后,学生**下次再走到同样上下文**时采样分布变了,就很可能不再产生同一个坏 token,从而岔出错误轨迹。

### ⚠️ 几个潜在问题

1. **梯度稀疏**:坏段里绝大多数 token(连词、标点、常见数学短语)教师分布和学生分布本来就类似 → 逐 token KL 接近 0,梯度接近 0。**大部分坏段 token 其实没训到什么**。
2. **关键 token 被稀释**:真正引发错误的可能只是坏段开头 1–2 个决策 token(比如触发词后面跟一个错误代数变形),但整段 loss 被平均 → 关键位置的梯度信号被稀释。
3. **Prior-text 分布偏移**:`prior_text` 是学生写的,教师在训练时条件在**学生风格的 prior 上**预测下一段。教师在这种非其自然分布的上下文下给出的分布本身带噪,KL 对齐的是"学生风格 × 教师参数"的混合空间,不是纯教师信号。
4. **反向 KL 的 mode-seeking 特性**:学生如果对坏段某 token 非常自信(s 尖锐),而教师认为应该是另一个 token,KL 主要由 s 的 mode 支配 —— 学生会朝教师 mode 移动,但只锁单点,错过多条合理替代路径的信息。
5. **Teacher 也可能错**:PR 分支不保证 `teacher_correct=True`(教师兜底分支在 `max_acc==0` 那一支才显式检查),PR 段训练时教师的分布只是"另一种可能的方向",不一定指向正确答案。



  ┌───────┬───────────────────┬────────────────────────┬─────────────────────────────────────────────────────────────────┐                                                                                  
  │ Epoch │ test_acc (5000题) │ train_acc (2071题池内) │                           Checkpoint                            │
  ├───────┼───────────────────┼────────────────────────┼─────────────────────────────────────────────────────────────────┤                                                                                  
  │ base  │ —                 │ —                      │ models/Qwen3-1.7B                                               │
  ├───────┼───────────────────┼────────────────────────┼─────────────────────────────────────────────────────────────────┤                                                                                  
  │ 1     │ 0.7344            │ 0.178                  │ checkpoints/epoch_001/v0-.../checkpoint-?                       │
  ├───────┼───────────────────┼────────────────────────┼─────────────────────────────────────────────────────────────────┤                                                                                  
  │ 2     │ 0.7254            │ 0.237                  │ checkpoints/epoch_002/v0-.../checkpoint-1014                    │
  ├───────┼───────────────────┼────────────────────────┼─────────────────────────────────────────────────────────────────┤                                                                                  
  │ 3     │ 0.7188            │ 0.272                  │ checkpoints/epoch_003/v0-20260422-120202/checkpoint-1931 ← 最新 │