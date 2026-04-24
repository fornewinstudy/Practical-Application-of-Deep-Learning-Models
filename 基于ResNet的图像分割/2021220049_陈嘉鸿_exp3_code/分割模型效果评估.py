import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from PIL import Image
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载预训练模型
weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
model = deeplabv3_resnet50(weights=weights).to(device)
# # 加载微调后的模型权重
# model.load_state_dict(torch.load('finetuned_deeplabv3_resnet50.pth'))

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


# 调整掩码大小以匹配目标掩码
def resize_mask(pred, target_shape):
    pred_tensor = torch.tensor(pred).unsqueeze(0).unsqueeze(0).float()
    pred_resized = F.interpolate(pred_tensor, size=target_shape, mode='nearest')
    return pred_resized.squeeze().byte().cpu().numpy()


# 评估函数
def dice_coefficient(pred, target):
    intersection = np.sum(pred * target)
    dice = (2. * intersection) / (np.sum(pred) + np.sum(target))
    return dice


def iou(pred, target):
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    iou_score = intersection / union
    return iou_score


def hausdorff_distance(pred, target):
    pred_points = np.transpose(np.nonzero(pred))
    target_points = np.transpose(np.nonzero(target))
    forward_hausdorff = directed_hausdorff(pred_points, target_points)[0]
    backward_hausdorff = directed_hausdorff(target_points, pred_points)[0]
    return max(forward_hausdorff, backward_hausdorff)


def evaluate(pred, target):
    pred_binary = (pred > 0).astype(int)
    target_binary = (target > 0).astype(int)
    dice = dice_coefficient(pred_binary, target_binary)
    iou_score = iou(pred_binary, target_binary)
    hausdorff = hausdorff_distance(pred_binary, target_binary)
    return dice, iou_score, hausdorff


# 显示结果
def visualize(image_path, seg_map, gt_mask):
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask, cmap='gray')
    plt.title("Ground Truth Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(seg_map, cmap='jet', alpha=0.5)
    plt.title("Segmented Image")

    plt.show()


# 测试
image_path = "b1f672549b20c2ccba7305e8a230f9ab.jpeg"
gt_mask_path = "ground_truth_mask.png"  # 替换为实际金标准mask路径

# 获取预测和金标准mask
seg_map = segment(image_path)
gt_mask = np.array(Image.open(gt_mask_path).convert('L'))

# 调整预测掩码大小以匹配金标准掩码
seg_map_resized = resize_mask(seg_map, gt_mask.shape)

# 评估预测结果
dice, iou_score, hausdorff = evaluate(seg_map_resized, gt_mask)
print(f"Dice Coefficient: {dice:.4f}")
print(f"IoU: {iou_score:.4f}")
print(f"Hausdorff Distance: {hausdorff:.4f}")

# 可视化结果
visualize(image_path, seg_map_resized, gt_mask)

##########################################################################################3
# from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
#
# # 加载预训练模型
# weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
# model = maskrcnn_resnet50_fpn(weights=weights).to(device)
# model.eval()
#
# # 图像预处理
# def preprocess(image_path):
#     image = Image.open(image_path).convert("RGB")
#     preprocess_transforms = weights.transforms()
#     return preprocess_transforms(image).unsqueeze(0).to(device)
#
# # 进行图像分割
# def segment(image_path):
#     input_image = preprocess(image_path)
#     with torch.no_grad():
#         output = model(input_image)[0]
#     return output['masks'][0, 0].byte().cpu().numpy()
#
# # 测试
# image_path = "snapshot0001.png"
# gt_mask_path = "ground_truth_mask.png"  # 替换为实际金标准mask路径
#
# # 获取预测和金标准mask
# seg_map = segment(image_path)
# gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
#
# # 评估预测结果
# dice, iou_score, hausdorff = evaluate(seg_map, gt_mask)
# print(f"Dice Coefficient: {dice:.4f}")
# print(f"IoU: {iou_score:.4f}")
# print(f"Hausdorff Distance: {hausdorff:.4f}")
#
# # 可视化结果
# visualize(image_path, seg_map, gt_mask)
