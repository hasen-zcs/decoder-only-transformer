from model.model import GPT, GPTConfig
from model.Tokenizer import MyTokenizer
import torch
from torch import nn
from Dataset import MyDataset
from model.Config import GPTConfig
from pathlib import Path
import csv
import math

def get_lr(step, max_steps, base_lr=1e-3, warmup_steps=200, min_lr=1e-4):
    """
    学习率调度：
    step < warmup_steps:  从 0 线性升到 base_lr
    step >= warmup_steps: 按余弦曲线从 base_lr 降到 min_lr
    """
    if step < warmup_steps:
        return base_lr * (step+1) /warmup_steps
    progress = (step - warmup_steps)/max(max_steps-warmup_steps, 1)
    cosine = 0.5*(1 + math.cos(math.pi*progress))
    return min_lr + (base_lr-min_lr)*cosine



# 实例化配置，模型，分词器，优化器
config = GPTConfig()
tokenizer = MyTokenizer(config.VOCAB_PATH)
gpt = GPT(config).to(config.device)
optimizer = torch.optim.AdamW(
    gpt.parameters(),
    lr=1e-3
)



# 导入数据
train_dataset = MyDataset(
    "data/train.bin", 
    config.block_size,
    config.batch_size
    )

val_dataset = MyDataset(
    "data/val.bin", 
    config.block_size,
    config.batch_size
    )

@torch.no_grad()
def evaluate(model, dataset, eval_steps, device="cuda"):
    model.eval()

    losses = []
    for _ in range(eval_steps):
        x, y = dataset.get_batch(device)

        logits, loss = gpt(x, y)

        losses.append(loss.item())
    model.train()

    return sum(losses) / len(losses)


# ---------- 训练日志（供 matplotlib 画 loss 曲线） ----------
LOG_FILE = Path("log") / "training_log.csv"

def log_metrics(step, train_loss, val_loss=None, lr=None, grad_norm=None):
    LOG_FILE.parent.mkdir(exist_ok=True)
    new = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["step", "train_loss", "val_loss", "lr", "grad_norm"])
        w.writerow([
            step,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}" if val_loss is not None else "",
            f"{lr:.6e}" if lr is not None else "",
            f"{grad_norm:.4f}" if grad_norm is not None else "",
        ])

def train_epochs(max_steps, start_step=None):
    if start_step is None:
        start_step = 0
    for step in range(start_step, max_steps):
        # 按调度器获取学习率
        for group in optimizer.param_groups:
            group["lr"] = get_lr(step, max_steps,
                                 config.base_lr,
                                 config.warmup_steps,
                                 config.min_lr)

        optimizer.zero_grad()
        # 取数据
        x, y = train_dataset.get_batch(config.device)
        # 前向传播
        logits, loss = gpt(x, y)
        # 反向传播
        loss.backward()
        # 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(gpt.parameters(), config.grad_clip)

        # 更新优化器
        optimizer.step()

        if (step+1) % 100 == 0:

            val_loss = evaluate(
                gpt,
                val_dataset,
                eval_steps=20
            )
            print(
                f"step {step+1}, "
                f"train loss {loss:.4f}, "
                f"val loss {val_loss:.4f},"
                f"lr {optimizer.param_groups[0]['lr']:.2e},"
                f"grad_norm {grad_norm:.2f}"
            )
            log_metrics(
                step + 1, loss.item(), val_loss,
                optimizer.param_groups[0]["lr"], grad_norm
            )

        elif (step+1) % 10 == 0:
            # 每10步记一次 train loss，曲线更平滑
            log_metrics(
                step + 1, loss.item(), None,
                optimizer.param_groups[0]["lr"], grad_norm
            )

        if (step+1) % 1000 == 0:
            checkpoint = {
                        "step" : step,
                        "model_state_dict": gpt.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item()
                    }
            
            torch.save(
                        checkpoint,
                        "checkpoints/step_latest.pt"
                    )
        if step+1 == max_steps:
            torch.save(gpt.state_dict(), "checkpoints/model.pth")



# 测试是否可用
# print("train batch:", train_dataset.get_batch()[0].shape)
start_step = 0
# checkpoints断点重续
p = Path("checkpoints/step_latest.pt")
if p.exists():
    checkpoint = torch.load(
        "checkpoints/step_latest.pt",
        map_location=config.device
    )
    gpt.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_step = checkpoint["step"]

train_epochs(config.epochs, start_step)





