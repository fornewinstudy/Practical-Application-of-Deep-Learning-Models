# 预训练的模型参数
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from model import resnet34, resnet101
import torchvision.models.resnet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# 检测是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using  device GPU {} .".format(device))

# 数据预处理（数据增强）
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

# 获取图像数据集的路径
data_root = os.path.abspath(os.getcwd())  		# get data root path 返回本层目录
image_path = data_root + "/dataset_mri/"

# 创建训练集的数据加载器
train_dataset = datasets.ImageFolder(root=image_path+"train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)

dataset_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in dataset_list.items())

# 将类别索引写入json文件
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices34_mri.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 16       # batch_size设为16
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)
# 同理创建验证集的数据加载器
validate_dataset = datasets.ImageFolder(root=image_path + "val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)
# 创建ResNet模型
net = resnet101()
# 加载预训练权重
model_weight_path = "./resnet101-pre.pth"
missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)

# 修改全连接层的结构，适应新的分类任务（4类）
inchannel = net.fc.in_features
net.fc = nn.Linear(inchannel, 4)
net.to(device)

# 定义损失函数和优化器
loss_function = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.0001)
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

best_acc = 0.0
save_path = './resNet101_mri_epoch20_swin.pth'

# 初始化存储每个类别指标的列表
class_metrics = {'accuracy': [[] for _ in range(4)], 'precision': [[] for _ in range(4)],
                 'recall': [[] for _ in range(4)], 'f1_score': [[] for _ in range(4)]}

for epoch in range(20):
    # train
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step+1)/len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")
    print()

    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))  # eval model only have last output layer
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
            all_preds.extend(predict_y.cpu().numpy())
            all_labels.extend(val_labels.cpu().numpy())

        val_accurate = acc / val_num
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds, average=None)
        val_recall = recall_score(all_labels, all_preds, average=None)
        val_f1 = f1_score(all_labels, all_preds, average=None)

        # Append metrics to the dictionary
        for i in range(len(val_precision)):
            TP = cm[i][i]
            TN = np.sum(np.delete(np.delete(cm, i, 0), i, 1))
            FP = np.sum(cm[:, i]) - TP
            FN = np.sum(cm[i, :]) - TP
            acc = (TP + TN) / (TP + TN + FP + FN)
            class_metrics['accuracy'][i].append(acc)
            class_metrics['precision'][i].append(val_precision[i])
            class_metrics['recall'][i].append(val_recall[i])
            class_metrics['f1_score'][i].append(val_f1[i])
            print(f"Class {i}: Epoch {epoch + 1} - Accuracy: {acc}, Precision: {val_precision[i]}, Recall: {val_recall[i]}, F1-Score: {val_f1[i]}")

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))

print('Finished Training')

# Saving the metrics
with open('class_metrics_sgd.json', 'w') as f:
    json.dump(class_metrics, f)

