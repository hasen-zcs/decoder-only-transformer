# -*- coding: utf-8 -*-
"""
用 matplotlib 画训练 loss 曲线
数据来源: train_mini.py 记录的 log/training_log.csv
输出:     output/loss_curve.png
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

# 中文字体（Windows 自带微软雅黑）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

LOG_FILE = Path("log") / "training_log.csv"

if not LOG_FILE.exists():
    raise SystemExit(f"找不到日志文件 {LOG_FILE}，请先运行 train_mini.py 训练。")

steps, train_losses = [], []
val_steps, val_losses = [], []
with open(LOG_FILE, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        step = int(row["step"])
        if row["train_loss"]:
            steps.append(step)
            train_losses.append(float(row["train_loss"]))
        if row["val_loss"]:
            val_steps.append(step)
            val_losses.append(float(row["val_loss"]))

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(steps, train_losses, color="tab:blue", lw=1.2, label="train loss")
ax.semilogy(val_steps, val_losses, color="tab:red", lw=1.2, marker="o",
            ms=4, label="val loss")

ax.set_xlabel("训练步数 (step)")
ax.set_ylabel("loss（对数刻度）")
ax.set_title("decoder-only transformer 训练曲线")
ax.grid(alpha=0.3, which="both")
ax.legend()

plt.tight_layout()
out = Path("output") / "loss_curve.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=150)
print(f"已保存: {out.resolve()}")
plt.show()
