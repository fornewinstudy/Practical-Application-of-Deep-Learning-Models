import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# 解决中文显示问题,固定格式，直接复制下面俩行代码就行
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False

# 加载波士顿房价数据集
data_url = datasets.load_diabetes()
data = data_url.data
target = data_url.target

# 设置神经网络模型
model = nn.Sequential(
    nn.Linear(10, 24),  # 根据数据集的特征数量修改输入层的大小
    nn.ReLU(),
    nn.Linear(24, 32),
    nn.ReLU(),
    nn.BatchNorm1d(32),
    nn.Linear(32, 1)
)

# 实例化优化器、损失函数
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.01)

# 创建训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

# 转换为PyTorch张量
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 创建DataLoader对象
batch_size = 32
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)

# 训练模型
loss_values, val_loss_values = [], []
mae, val_mae = [], []
epoch_nums = 300  # 总epoch
for epoch in range(epoch_nums):
    model.train()
    total_mae, num_batches = 0, 0
    for x, y in train_loader:
        y_pred = model(x)
        loss = criterion(y_pred[:, 0], y)
        batch_mae = torch.abs(y_pred - y).mean()
        total_mae += batch_mae.item()
        num_batches += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_values.append(loss.item())
    mae.append(total_mae / num_batches)

    # 验证模型
    model.eval()
    total_mae, num_batches = 0, 0
    with torch.no_grad():  # 禁止梯度计算
        for x, y in test_loader:
            y_pred = model(x)
            loss = criterion(y_pred[:, 0], y)
            batch_mae = torch.abs(y_pred - y).mean()
            total_mae += batch_mae.item()
            num_batches += 1
        val_loss_values.append(loss.item())
        val_mae.append(total_mae / num_batches)

# 绘图
epochs = range(1, len(loss_values) + 1)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# 绘制模型的MAE vs Epochs
ax[0].plot(epochs, mae, 'r', label='Training MAE')
ax[0].plot(epochs, val_mae, 'b', label='Validation MAE')
ax[0].set_title('Training & Validation MAE', fontsize=16)
ax[0].set_xlabel('Epochs', fontsize=16)
ax[0].set_ylabel('MAE', fontsize=16)
ax[0].legend()

# Plot Loss vs Epochs
ax[1].plot(epochs, loss_values, 'g', label='Training Loss')
ax[1].plot(epochs, val_loss_values, 'k', label='Validation Loss')
ax[1].set_title('Training & Validation Loss', fontsize=16)
ax[1].set_xlabel('Epochs', fontsize=16)
ax[1].set_ylabel('Loss', fontsize=16)
ax[1].legend()

plt.show()
