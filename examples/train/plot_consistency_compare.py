import pandas as pd
import matplotlib.pyplot as plt
import os

# --- wandb 训练日志 ---
wandb_df = pd.read_csv("/data/chenyang/wandb_dl/opd_consistency.csv")
wandb_df = wandb_df.sort_values("_step").reset_index(drop=True)

# --- eval_consistency_output 里的 4 个点 ---
eval_rows = []
eval_root = "/data/chenyang/OPD/eval_consistency_output"
for d in sorted(os.listdir(eval_root)):
    p = os.path.join(eval_root, d, "consistency_summary.tsv")
    if not os.path.isfile(p):
        continue
    # 跳过以 # 开头的注释行
    df = pd.read_csv(p, sep="\t", comment="#")
    eval_rows.append(df.iloc[0])
eval_df = pd.DataFrame(eval_rows).sort_values("run_step").reset_index(drop=True)
print("Eval points:")
print(eval_df[["checkpoint", "run_step", "mean_consistency"]])

# 在 step 1 处加一个起点，与 wandb 训练曲线的起点对齐
start_val = float(wandb_df.loc[wandb_df["_step"] == 1, "opd_consistency/mean_consistency"].iloc[0])
eval_df = pd.concat([
    pd.DataFrame([{"checkpoint": "global_step_0", "run_step": 1, "mean_consistency": start_val}]),
    eval_df,
], ignore_index=True)

# --- 画图 ---
fig, ax = plt.subplots(figsize=(10, 6))

# 训练曲线
ax.plot(
    wandb_df["_step"],
    wandb_df["opd_consistency/mean_consistency"],
    color="#1f77b4",
    linewidth=1.6,
    alpha=0.9,
    label="wandb training (opd_consistency/mean_consistency)",
)

# eval 点用红色折线连起来 + 散点
ax.plot(
    eval_df["run_step"],
    eval_df["mean_consistency"],
    color="#d62728",
    marker="o",
    markersize=9,
    linewidth=2,
    linestyle="--",
    label="eval_consistency_output (offline eval)",
)

# 在 eval 点旁边标注数值
for _, r in eval_df.iterrows():
    ax.annotate(
        f'{r["mean_consistency"]:.4f}',
        xy=(r["run_step"], r["mean_consistency"]),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=9,
        color="#d62728",
    )

ax.set_xlabel("global step")
ax.set_ylabel("mean consistency")
ax.set_title("OPD consistency: training log vs offline eval (JustRL-1.5B / R1-Distill-Qwen-1.5B)")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")

# x 轴刻度对齐到 eval 点
ticks = sorted(set(list(wandb_df["_step"][::20]) + list(eval_df["run_step"])))
ax.set_xticks(ticks)
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
out = "/data/chenyang/wandb_dl/consistency_compare.png"
plt.savefig(out, dpi=150)
print(f"\nSaved figure to {out}")
