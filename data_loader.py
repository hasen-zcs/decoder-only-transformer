import numpy as np
import torch


def get_batch(data, batch_size, block_size):
    ix = np.random.randint(
        0,
        len(data)-block_size,
        size=batch_size
    )

    x = np.stack([
        data[i:i+block_size]
        for i in ix
    ])

    y = np.stack([
            data[i+1:i+block_size+1]
            for i in ix
        ])

    x = torch.from_numpy(x.astype(np.int64))
    y = torch.from_numpy(y.astype(np.int64))

    return x, y

if __name__=="__main__":    
    # 读取
    train_data = np.fromfile("data/train.bin", dtype=np.uint16)
    val_data = np.fromfile("data/val.bin", dtype=np.uint16)

    print("train:", train_data.shape)
    print("val:", val_data.shape)

    block_size = 128
    batch_size = 64
    x, y = get_batch(train_data, batch_size, block_size)
    print("x shape:", x.shape)
    print("y shape:", y.shape)

    print(x[0])
    print(y[0])
    

