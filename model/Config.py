class GPTConfig:
    vocab_size = 4242
    block_size = 128
    n_embd = 128
    ffn_dim = 256
    b_head = 4
    n_kv_head = 2
    n_layer = 8
    dropout = 0.1
    batch_size = 64

    # 训练次数
    epochs = 1000

    # ------训练技巧超参数-------
    base_lr = 1e-3        # 学习率峰值 
    min_lr = 1e-4         # 学习率衰减下限
    warmup_steps = 200    # warmup步数
    grad_clip = 1.0       # 梯度裁剪阈值

    # 配置
    device = "cuda"

    VOCAB_PATH = "data/vocab.json"