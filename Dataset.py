import numpy as np
import torch

class MyDataset:
    def __init__(self, data_path, block_size, batch_size):
        self.block_size = block_size
        self.batch_size = batch_size

        self.data = np.fromfile(data_path, dtype=np.uint16)

    def get_batch(self, device="cpu"):
        ix = np.random.randint(
            0,
            len(self.data)-self.block_size,
            size=self.batch_size
        )

        x = np.stack([
            self.data[i:i+self.block_size]
            for i in ix
        ])

        y = np.stack([
                self.data[i+1:i+self.block_size+1]
                for i in ix
            ])

        x = torch.from_numpy(x.astype(np.int64)).to(device=device)
        y = torch.from_numpy(y.astype(np.int64)).to(device=device)

        return x, y       
