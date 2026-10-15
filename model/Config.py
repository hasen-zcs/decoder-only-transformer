class GPTConfig:
    vocab_size = 4242
    block_size = 128
    n_embd = 128
    ffn_dim = 256
    b_head = 4
    n_layer = 8
    dropout = 0.1
    batch_size = 64

    # 训练次数
    epochs = 1000

    # 配置
    device = "cuda"

    VOCAB_PATH = "data/vocab.json"