"""
阶段二vs三消融实验 · 训练（自包含，单文件可跑）
本目录 model.py / structure.py = phase2 架构；本文件与 phase3 侧逐项一致。
执行（项目根目录）：
& D:\\Users\\17743\\anaconda3\\envs\\minimind\\python.exe experiment/phase2_vs_phase3/phase2/train.py
"""
import sys
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn

BASE = Path(__file__).resolve().parent      # phase2/
ROOT = BASE.parents[2]                      # 项目根（Dataset.py、data/）
sys.path.append(str(ROOT))                  # append 在末尾：本目录 model.py 先命中，不与根目录 model/ 包冲突

from model import GPT
from Config import GPTConfig
from Dataset import MyDataset

SEED = 42


def get_lr(step, max_steps, base_lr=1e-3, warmup_steps=200, min_lr=1e-4):
    """
    学习率调度：
    step < warmup_steps:  从 0 线性升到 base_lr
    step >= warmup_steps: 按余弦曲线从 base_lr 降到 min_lr
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


# 播种：torch = 模型初始化 + dropout；numpy = 数据随机窗口顺序
# 两侧同种子 → 两边看到的 (x, y) 窗口序列完全一致
torch.manual_seed(SEED)
np.random.seed(SEED)

config = GPTConfig()
gpt = GPT(config).to(config.device)
optimizer = torch.optim.AdamW(gpt.parameters(), lr=config.base_lr)

train_dataset = MyDataset(str(ROOT / "data" / "train.bin"), config.block_size, config.batch_size)
val_dataset = MyDataset(str(ROOT / "data" / "val.bin"), config.block_size, config.batch_size)

params = sum(p.numel() for p in gpt.parameters())
total_tokens = config.epochs * config.batch_size * config.block_size
print(f"[phase2] params={params:,} | seed={SEED} | steps={config.epochs} | total tokens={total_tokens:,}")
print(f"[phase2] 期望初始 loss 90~110（自尖峰），盯前 100 步输出", flush=True)


@torch.no_grad()
def evaluate(model, dataset, eval_steps, device):
    model.eval()
    losses = []
    for _ in range(eval_steps):
        x, y = dataset.get_batch(device)
        logits, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(max_steps, start_step=0):
    for step in range(start_step, max_steps):
        # 每步按调度器设置学习率
        for group in optimizer.param_groups:
            group["lr"] = get_lr(step, max_steps, config.base_lr, config.warmup_steps, config.min_lr)

        optimizer.zero_grad()
        x, y = train_dataset.get_batch(config.device)
        logits, loss = gpt(x, y)
        loss.backward()
        # 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(gpt.parameters(), config.grad_clip)
        optimizer.step()

        if (step + 1) % 100 == 0:
            val_loss = evaluate(gpt, val_dataset, eval_steps=20, device=config.device)
            print(
                f"step {step+1}, "
                f"train loss {loss:.4f}, "
                f"val loss {val_loss:.4f}, "
                f"lr {optimizer.param_groups[0]['lr']:.2e}, "
                f"grad_norm {grad_norm:.2f}",
                flush=True
            )
            # 写 CSV（plot_losses.py 的数据源）
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{step+1},{loss.item():.6f},{val_loss:.6f},"
                        f"{optimizer.param_groups[0]['lr']:.6e},{grad_norm:.4f}\n")

        if (step + 1) % 1000 == 0:
            ckpt_dir = BASE / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save({
                "step": step,
                "model_state_dict": gpt.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            }, ckpt_dir / "step_latest.pt")

    # 训练结束：最终权重
    ckpt_dir = BASE / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(gpt.state_dict(), ckpt_dir / "model_final.pth")
    print(f"[phase2] done. final → {ckpt_dir / 'model_final.pth'}", flush=True)


# 断点续训：存在 step_latest.pt 则从上次继续（会重跑最后一个已存 step 一次，无害）
start_step = 0
latest = BASE / "checkpoints" / "step_latest.pt"
if latest.exists():
    checkpoint = torch.load(latest, map_location=config.device)
    gpt.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_step = checkpoint["step"]
    print(f"[phase2] resume from step {start_step}", flush=True)

# —— loss 日志：每 100 步一行写 CSV；续训时保留旧日志中 step < start_step 的行（去重、不断线）——
log_path = BASE / "logs" / "training_log.csv"
log_path.parent.mkdir(exist_ok=True)
kept = []
if start_step > 0 and log_path.exists():
    for ln in log_path.read_text(encoding="utf-8").strip().splitlines()[1:]:
        if int(ln.split(",")[0]) < start_step:
            kept.append(ln)
log_path.write_text("step,train_loss,val_loss,lr,grad_norm\n" + "".join(l + "\n" for l in kept), encoding="utf-8")

train(config.epochs, start_step)

# —— 训练结束：自动更新合并曲线（phase2 + phase3 同图 → loss_curve.png）——
try:
    sys.path.append(str(BASE.parents[1]))
    from plot_losses import plot_all
    plot_all(BASE.parents[1])
except Exception as e:
    print(f"[phase2] plot skipped: {e}", flush=True)
