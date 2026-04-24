import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# 加载预训练模型并将其移至GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
model = deeplabv3_resnet50(weights=weights).to(device)
# SegFormer（Hugging Face）
# from transformers import SegformerForSemanticSegmentation
# model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
print(torch.cuda.memory_summary(device=None, abbreviated=False))
model.eval()

# 图像预处理
def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess_transforms = weights.transforms()
    return preprocess_transforms(image).unsqueeze(0).to(device)

# 进行图像分割
def segment(image_path):
    input_image = preprocess(image_path)
    with torch.no_grad():
        output = model(input_image)['out'][0]
    return output.argmax(0).byte().cpu().numpy()

# 显示结果
def visualize(image_path, seg_map):
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(seg_map, cmap='jet', alpha=0.5)
    plt.title("Segmented Image")
    plt.show()

# 测试
image_path = "891.jpeg"
seg_map = segment(image_path)
cv2.imwrite('seg.png',seg_map)
visualize(image_path, seg_map)


