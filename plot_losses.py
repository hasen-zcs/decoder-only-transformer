"""
Loss曲线（根目录主线）：数据是 logs/training_log.csv（train_mini.py 每100步自动写）
输出：loss_curve.png（train / val 两个子图，对数纵轴）
执行：& $PY plot_losses.py
图内标签用英文：服务器 matplotlib 默认字体没有中文字形，中文会变方框。
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_log(base: Path):
    p = base / "logs" / "training_log.csv"
    if not p.exists():
        return None
    steps, train_l, val_l = [], [], []
    for ln in p.read_text(encoding="utf-8").strip().splitlines()[1:]:
        s, t, v, lr, g = ln.split(",")
        steps.append(int(s))
        train_l.append(float(t))
        val_l.append(float(v))
    return steps, train_l, val_l


def plot_all(base: Path):
    data = load_log(base)
    if data is None:
        print(f"没有日志: {base / 'logs' / 'training_log.csv'}（先跑 train_mini.py）")
        return None
    steps, train_l, val_l = data

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, ys, title in zip(axes, (train_l, val_l), ("train loss", "val loss")):
        ax.plot(steps, ys, marker=".", lw=1.2, color="tab:blue")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(title)
        ax.set_yscale("log")     # 初始自尖峰 ~100 → 后期 ~1.5，线性轴会把后期压没
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = base / "loss_curve.png"
    fig.savefig(out, dpi=130)
    print(f"saved → {out}")
    return out


if __name__ == "__main__":
    plot_all(Path(".").resolve())
