from model.model import GPT, GPTConfig
from model.Tokenizer import MyTokenizer
import torch
from torch import nn
from Dataset import MyDataset
from model.Config import GPTConfig
from pathlib import Path
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

def train_epochs(max_steps, start_step=None):
    if start_step is None:
        start_step = 0
    for step in range(start_step, max_steps):
        # 按调度器获取学习率，每一个epoch进行一次取
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
            # 写CSV（plot_losses.py 的数据源）
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{step+1},{loss.item():.6f},{val_loss:.6f},"
                        f"{optimizer.param_groups[0]['lr']:.6e},{grad_norm:.4f}\n")

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

# —— loss日志：每100步一行写CSV；续训时保留旧日志中 step < start_step 的行（去重、不断线）——
log_path = Path("logs/training_log.csv")
log_path.parent.mkdir(exist_ok=True)
kept = []
if start_step > 0 and log_path.exists():
    for ln in log_path.read_text(encoding="utf-8").strip().splitlines()[1:]:
        if int(ln.split(",")[0]) < start_step:
            kept.append(ln)
log_path.write_text("step,train_loss,val_loss,lr,grad_norm\n" + "".join(l + "\n" for l in kept), encoding="utf-8")

train_epochs(config.epochs, start_step)

# —— 训练结束：自动刷新loss曲线（train/val 双子图，对数纵轴 → loss_curve.png）——
try:
    from plot_losses import plot_all
    plot_all(Path(".").resolve())
except Exception as e:
    print(f"plot skipped: {e}")





