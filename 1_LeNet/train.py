import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
# 数据预处理
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 训练集
# 导入50000张训练图片
train_set = torchvision.datasets.CIFAR10(root='./data', 	 # 数据集存放目录
										 train=True,		 # 表示是数据集中的训练集
                                        download=True,  	 # 第一次运行时为True，下载数据集，下载完成后改为False
                                        transform=transform) # 预处理过程
# 加载训练集，实际过程需要分批次（batch）训练
train_loader = torch.utils.data.DataLoader(train_set, 	  # 导入的训练集
										   batch_size=50, # 每批训练的样本数
                                          shuffle=False,  # 是否打乱训练集
                                          num_workers=0)  # 使用线程数，在windows下设置为0

# 测试集(用作验证集)
# 导入10000张测试图片
test_set = torchvision.datasets.CIFAR10(root='./data',
										train=False,	# 表示是数据集中的测试集
                                        download=False,transform=transform)
# 加载测试集
test_loader = torch.utils.data.DataLoader(test_set,
										  batch_size=10000, # 每批用于验证的样本数
										  shuffle=False, num_workers=0)
# 获取测试集中的图像和标签，用于accuracy计算
test_data_iter = iter(test_loader)
test_image, test_label = test_data_iter.next()

# 训练模型
net = LeNet()  # 定义训练的网络模型
loss_function = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失函数
optimizer = optim.Adam(net.parameters(), lr=0.001)  # 定义优化器（训练参数，学习率）

for epoch in range(5):  # 一个epoch即对整个训练集进行一次训练
    running_loss = 0.0
    time_start = time.perf_counter()

    for step, data in enumerate(train_loader, start=0):  # 遍历训练集，step从0开始计算
        inputs, labels = data  # 获取训练集的图像和标签
        optimizer.zero_grad()  # 清除历史梯度

        # forward + backward + optimize
        outputs = net(inputs)  # 正向传播
        loss = loss_function(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器更新参数

        # 打印耗时、损失、准确率等数据
        running_loss += loss.item()
        if step % 1000 == 999:  # print every 1000 mini-batches，每1000步打印一次
            with torch.no_grad():  # 在以下步骤中（验证过程中）不用计算每个节点的损失梯度，防止内存占用
                outputs = net(test_image)  # 测试集传入网络（test_batch_size=10000），output维度为[10000,10]
                predict_y = torch.max(outputs, dim=1)[1]  # 以output中值最大位置对应的索引（标签）作为预测输出
                accuracy = (predict_y == test_label).sum().item() / test_label.size(0)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %  # 打印epoch，step，loss，accuracy
                      (epoch + 1, step + 1, running_loss / 500, accuracy))

                print('%f s' % (time.perf_counter() - time_start))  # 打印耗时
                running_loss = 0.0

print('Finished Training')

# 保存训练得到的参数
save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)
