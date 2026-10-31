import torch

train_data = torch.load("train.pt",weights_only=True)
val_data = torch.load("val.pt",weights_only=True)

block_size = 64
batch_size = 16
def get_batch(data):
    starts = torch.randint(0,len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x, y

def get_eval_batch(data, batch_index):
    start = batch_index * batch_size * block_size
    if start + batch_size * block_size + 1 > len(data):
        return None
    x = torch.stack([data[i:i + block_size] for i in range(start,start + batch_size * batch_size,block_size)])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in range(start,start + batch_size * batch_size,block_size)])
    return x, y

x, y = get_batch(train_data)
print("Input shape:",x.shape)
print("Target shape:",y.shape)