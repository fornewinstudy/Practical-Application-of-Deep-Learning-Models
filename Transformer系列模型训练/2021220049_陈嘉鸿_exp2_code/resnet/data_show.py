import json
import matplotlib.pyplot as plt

# 读取class_metrics.json文件
with open('class_metrics_sgd.json', 'r') as f:
    class_metrics = json.load(f)

# 定义类别和指标
classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
metrics = ['accuracy', 'precision', 'recall', 'f1_score']

# 绘制不同指标上，横坐标为epoch，纵坐标为不同类别的指标
for metric in metrics:
    plt.figure()
    plt.title(f'{metric.capitalize()} vs Epoch for Different Classes')
    plt.xlabel('Epoch')
    plt.ylabel(f'{metric.capitalize()}')
    for i, cls in enumerate(classes):
        plt.plot(range(1, len(class_metrics[metric][i])+1), class_metrics[metric][i], label=cls)
    plt.legend()
    plt.show()

# 绘制不同类别上，横坐标为epoch，纵坐标为各的指标数值
for i, cls in enumerate(classes):
    plt.figure()
    plt.title(f'Metrics vs Epoch for {cls}')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    for metric in metrics:
        plt.plot(range(1, len(class_metrics[metric][i])+1), class_metrics[metric][i], label=metric.capitalize())
    plt.legend()
    plt.show()
