import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff

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


# 计算 Dice 系数
def dice_coefficient(pred, target):
    intersection = np.sum(pred * target)
    return 2.0 * intersection / (np.sum(pred) + np.sum(target))


# 计算 IoU
def iou(pred, target):
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    return intersection / union


# 计算 Hausdorff 距离
def hausdorff_distance(pred, target):
    pred_points = np.argwhere(pred)
    target_points = np.argwhere(target)
    forward_hausdorff = directed_hausdorff(pred_points, target_points)[0]
    backward_hausdorff = directed_hausdorff(target_points, pred_points)[0]
    return max(forward_hausdorff, backward_hausdorff)


# 显示结果
def visualize(image_path, output, gt_mask=None):
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")

    if gt_mask is not None:
        plt.subplot(1, 3, 2)
        plt.imshow(image)
        plt.imshow(gt_mask, cmap='jet', alpha=0.5)
        plt.title("Ground Truth Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(image)
    for i in range(len(output['masks'])):
        mask = output['masks'][i, 0].mul(255).byte().cpu().numpy()
        plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.title("Segmented Image")
    plt.show()


# 评估函数
def evaluate(pred_mask, gt_mask):
    pred_binary = (pred_mask > 0).astype(int)
    target_binary = (gt_mask > 0).astype(int)
    dice = dice_coefficient(pred_binary, target_binary)
    iou_score = iou(pred_binary, target_binary)
    hausdorff = hausdorff_distance(pred_binary, target_binary)
    return dice, iou_score, hausdorff


# 测试
image_path = "b1f672549b20c2ccba7305e8a230f9ab.jpeg"
gt_mask_path = "ground_truth_mask.png"  # 替换为实际金标准mask路径

output = segment(image_path)
# 获取预测的第一个mask作为示例
pred_mask = output['masks'][0, 0].mul(255).byte().cpu().numpy()
gt_mask = np.array(Image.open(gt_mask_path).convert('L'))

# 调整预测掩码大小以匹配金标准掩码
pred_mask_resized = np.array(Image.fromarray(pred_mask).resize(gt_mask.shape[::-1]))

# 评估预测结果
dice, iou_score, hausdorff = evaluate(pred_mask_resized, gt_mask)
print(f"Dice Coefficient: {dice:.4f}")
print(f"IoU: {iou_score:.4f}")
print(f"Hausdorff Distance: {hausdorff:.4f}")

# 可视化结果
visualize(image_path, output, gt_mask)
