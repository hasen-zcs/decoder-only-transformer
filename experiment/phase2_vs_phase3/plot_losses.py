"""
把 phase2 和 phase3 的 loss 曲线画在同一张图里：
- 数据：各 phase 的 logs/training_log.csv（train.py 每 100 步自动写）
- 输出：experiment/phase2_vs_phase3/loss_curve.png（train / val 两个子图，对数纵轴）
执行：& $PY experiment/phase2_vs_phase3/plot_losses.py
（train.py 结束时会自动调本脚本；两边都训完再跑一次就是完整对比图）
图内标签用英文：服务器 matplotlib 默认字体没有中文字形，中文会变方框。
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASES = [
    ("phase2", "phase2 (Post-LN + LN + pos-table)"),
    ("phase3", "phase3 (Pre-LN + RMSNorm + RoPE)"),
]


def load_log(exp_dir: Path, phase: str):
    p = exp_dir / phase / "logs" / "training_log.csv"
    if not p.exists():
        return None
    steps, train_l, val_l = [], [], []
    for ln in p.read_text(encoding="utf-8").strip().splitlines()[1:]:
        s, t, v, lr, g = ln.split(",")
        steps.append(int(s))
        train_l.append(float(t))
        val_l.append(float(v))
    return steps, train_l, val_l


def plot_all(exp_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, key, title in zip(axes, (1, 2), ("train loss", "val loss")):
        for phase, label in PHASES:
            data = load_log(exp_dir, phase)
            if data is None:
                continue
            steps, train_l, val_l = data
            ys = train_l if key == 1 else val_l
            ax.plot(steps, ys, marker=".", lw=1.2, label=label)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(title)
        ax.set_yscale("log")     # 初始自尖峰 ~100 → 后期 ~1.5，线性轴会把后期压没
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.tight_layout()
    out = exp_dir / "loss_curve.png"
    fig.savefig(out, dpi=130)
    print(f"saved → {out}")
    return out


if __name__ == "__main__":
    plot_all(Path(__file__).resolve().parent)
