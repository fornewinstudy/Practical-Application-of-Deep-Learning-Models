import os
import torch
import torchvision.transforms as T
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import numpy as np

# 自定义数据集类
class VOCDataset(VOCSegmentation):
    def __init__(self, root, year, image_set, transforms=None):
        super().__init__(root, year, image_set, download=False)
        self.transforms = transforms

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        if self.transforms is not None:
            image = self.transforms(image)
            target = torch.tensor(np.array(target), dtype=torch.long)
        return image, target

# 定义图像转换
train_transforms = T.Compose([
    T.Resize((520, 520)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = T.Compose([
    T.Resize((520, 520)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载训练和验证数据集
train_dataset = VOCDataset(root='./VOC2012', year='2012', image_set='train', transforms=train_transforms)
val_dataset = VOCDataset(root='./VOC2012', year='2012', image_set='val', transforms=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

################# 模型微调 ####################################
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

# 加载预训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1).to(device)

# 修改模型的最后一层，以适应Pascal VOC数据集
model.classifier[4] = nn.Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1)).to(device)  # Pascal VOC有21个类别

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练和验证函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    return val_loss / len(val_loader)

# 训练和验证模型
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# 保存微调后的模型
torch.save(model.state_dict(), 'finetuned_deeplabv3_resnet50_voc.pth')
