import torch.nn as nn
import torch

'''
为了加快训练，代码只使用了一半的网络参数，相当于只用了原论文中网络结构的下半部分
（正好原论文中用的双GPU，我的电脑只有一块GPU）
（后来我又用完整网络跑了遍，发现一半参数跟完整参数的训练结果acc相差无几）
'''
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        # 用nn.Sequential()将网络打包成一个模块，精简代码
        # 卷积层提取图像特征
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),  # 直接修改覆盖原值，节省运算内存
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
        )
        # 全连接层对图像分类
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # Dropout 随机失活神经元，默认比例为0.5
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    # 前向传播过程
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # 展平后再传入全连接层
        x = self.classifier(x)
        return x

    # 网络权重初始化，实际上 pytorch 在构建网络时会自动初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 若是卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out',  # 用（何）kaiming_normal_法初始化权重
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 初始化偏重为0
            elif isinstance(m, nn.Linear):  # 若是全连接层
                nn.init.normal_(m.weight, 0, 0.01)  # 正态分布初始化
                nn.init.constant_(m.bias, 0)  # 初始化偏重为0
