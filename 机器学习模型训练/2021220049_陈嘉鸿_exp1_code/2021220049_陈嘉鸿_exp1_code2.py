import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 解决中文显示问题,固定格式，直接复制下面俩行代码就行
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

# 加载波士顿房价数据集
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# 将数据转换为 PyTorch 张量
data = torch.tensor(data, dtype=torch.float32)
target = torch.tensor(target, dtype=torch.float32)

# 创建 DataLoader 对象，用于批处理数据
batch_size = 8
dataset = TensorDataset(data, target)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 检查训练数据的形状
print(data.shape, target.shape)

torch.manual_seed(1432)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13, 16)
        self.fc2 = nn.Linear(16, 8)
        self.bn = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.bn(x)
        x = self.fc3(x)
        return x


def run_optimizer(opts, init_mode='he_normal', lr=None, momentum=None, num_rounds=5):
    mses_rounds = []
    for _ in range(num_rounds):
        mses = []
        for opt in tqdm(opts, desc='优化器'):
            model = Net()
            if init_mode == 'he_normal':
                nn.init.kaiming_normal_(model.fc1.weight)
                nn.init.kaiming_normal_(model.fc2.weight)
                nn.init.kaiming_normal_(model.fc3.weight)
            else:
                nn.init.zeros_(model.fc1.weight)
                nn.init.zeros_(model.fc2.weight)
                nn.init.zeros_(model.fc3.weight)

            criterion = nn.MSELoss()
            if lr is None and momentum is None:
                if opt == 'SGD':
                    optimizer = optim.__dict__[opt](model.parameters(), lr=0.01)
                else:
                    optimizer = optim.__dict__[opt](model.parameters())
            elif lr is not None and momentum is None:
                optimizer = optim.__dict__[opt](model.parameters(), lr=lr)
            elif lr is None and momentum is not None:
                if opt == 'Adam' or opt == 'Adamax':
                    optimizer = optim.__dict__[opt](model.parameters())
                elif opt == 'RMSprop':
                    optimizer = optim.__dict__[opt](model.parameters(), momentum=momentum)
                elif opt == 'Adagrad':
                    optimizer = optim.__dict__[opt](model.parameters())
                else:
                    optimizer = optim.__dict__[opt](model.parameters(), lr=0.01, momentum=momentum)

            model.train()
            for epoch in tqdm(range(50), desc='训练轮次', leave=False):
                for x, y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()
            model.eval()
            with torch.no_grad():
                mse = criterion(model(x), y).item()
            mses.append(mse)
        mses_rounds.append(mses)
    return mses_rounds

# Default optimizers
opt_names = ["Adam", "SGD", "RMSprop", "Adagrad", "Adamax"]
# Run optimization for 3 rounds
mses_rounds = run_optimizer(opt_names, num_rounds=20,lr=0.01)

# Print model performance for each optimizer
for i, mses in enumerate(mses_rounds):
    print(f"Model performance for Round {i+1}:")
    for j, opt_name in enumerate(opt_names):
        print(f"{opt_name}: {mses[j]:.2f}")
    print()
# Plotting optimizer performance across rounds
plt.figure(figsize=(10, 6))
for i, opt_name in enumerate(opt_names):
    optimizer_performance = [rounds[i] for rounds in mses_rounds]
    plt.plot(range(1, len(mses_rounds) + 1), optimizer_performance, marker='o', label=opt_name)

plt.title("Optimizer Performance Across Rounds")
plt.xlabel("Round")
plt.ylabel("Mean Squared Error")
plt.xticks(range(1, len(mses_rounds) + 1))
plt.legend()
plt.grid(True)
plt.show()

