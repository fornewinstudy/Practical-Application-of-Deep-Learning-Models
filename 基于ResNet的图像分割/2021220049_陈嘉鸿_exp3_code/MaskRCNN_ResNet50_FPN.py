import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from PIL import Image
import matplotlib.pyplot as plt

# 加载预训练模型并将其移至GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
model = maskrcnn_resnet50_fpn(weights=weights).to(device)
model.eval()


# 图像预处理
def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(device)


# 进行图像分割
def segment(image_path):
    input_image = preprocess(image_path)
    with torch.no_grad():
        output = model(input_image)[0]
    return output


# 显示结果
def visualize(image_path, output):
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    for i in range(len(output['masks'])):
        mask = output['masks'][i, 0].mul(255).byte().cpu().numpy()
        plt.imshow(mask, cmap='jet', alpha=0.5)

    plt.title("Segmented Image")
    plt.show()


# 测试
image_path = "b1f672549b20c2ccba7305e8a230f9ab.jpeg"
output = segment(image_path)
visualize(image_path, output)
