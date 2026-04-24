# 导入所需的库
import torch
from model import resnet34, resnet101  # 导入自定义的 ResNet-34 模型
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

# 定义数据预处理操作，包括将图像调整大小、居中裁剪、转为张量、以及标准化
data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# 加载待预测的图像
img = Image.open(r"D:\dyy_1\image_ocr\008_054907_L.png")
plt.imshow(img)  # 展示原始图像

# 对图像进行预处理
img = data_transform(img)
# 在第0维度上添加一个维度，将图像扩展为一个 batch
img = torch.unsqueeze(img, dim=0)

# 读取类别标签信息
try:
    json_file = open('./class_indices101_shuang.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# 创建 ResNet-34 模型
model = resnet101(num_classes=2)

# 加载预训练好的模型权重
model_weight_path = "./resNet101_shuang.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()  # 将模型设为评估模式

with torch.no_grad():
    # 输入图像并获取输出
    output = torch.squeeze(model(img))
    # 对输出进行 softmax 操作，得到概率分布
    predict = torch.softmax(output, dim=0)
    # 获取概率最大的类别索引
    predict_cla = torch.argmax(predict).numpy()

# 打印预测结果（类别名称和概率值）
print(class_indict[str(predict_cla)], predict[predict_cla].numpy())
if class_indict[str(predict_cla)] == '1':
    print(123)
plt.show()  # 展示预测结果
