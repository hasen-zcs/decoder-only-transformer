class GPTConfig:
    # —— 阶段二vs三消融 · phase3 侧（Pre-LN + RMSNorm + RoPE）——
    # 本文件必须与 phase2 侧逐项一致；两侧唯一允许不同的是 model.py / structure.py
    vocab_size = 6027
    block_size = 128
    n_embd = 128
    b_head = 4
    n_layer = 8
    dropout = 0.1
    batch_size = 64

    # 训练步数（10000 × 64 × 128 = 81.9M token ≈ 新训练集 2.17 epoch）
    epochs = 10000

    # ------训练技巧超参数-------
    base_lr = 1e-3        # 学习率峰值
    min_lr = 1e-4         # 学习率衰减下限
    warmup_steps = 200    # warmup步数
    grad_clip = 1.0       # 梯度裁剪阈值

    # 配置
    device = "cuda"

    VOCAB_PATH = "data/vocab.json"
