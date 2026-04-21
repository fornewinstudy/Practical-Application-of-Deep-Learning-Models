import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cv2
import os, glob, shutil
file = r"D:\CJH_D\数据集\dogvscat\train.zip"
# 1.获取所有图像
input_dir = 'datasetsCatDog'
glob_dir = file + '/*.jpg'
#opencv读取图像，并将图像大小 resize 为（224，224），以匹配模型输入层的大小以进行特征提取。
images = [cv2.resize(cv2.imread(file), (224, 224)) for file in glob.glob(glob_dir)]
paths = [file for file in glob.glob(glob_dir)]
# 图像数组转换为 float32 类型并reshape，然后做归一化。
images = np.array(np.float32(images).reshape(len(images), -1) / 255)
#绘制数据分布图
plt.scatter(images[:, 0], images[:, 1], c = "red", marker='o', label='origin')
plt.xlabel('length')
plt.ylabel('width')
plt.legend(loc=2)
plt.show()

# 2.加载预先训练的模型MobileNetV2来实现图像分类
model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3)) #'imagenet' (pre-training on ImageNet),
predictions = model.predict(images.reshape(-1, 224, 224, 3))
pred_images = predictions.reshape(images.shape[0], -1)

# 3. 使用轮廓法寻找K值
sil = []
kl = []
kmax = 4 #设置最大的K值
for k in range(2, kmax + 1):
    kMeans = KMeans(n_clusters=k).fit(pred_images)#构造聚类器 聚类
    labels = kMeans.labels_ #获取聚类标签
    sil.append(silhouette_score(pred_images, labels, metric='euclidean')) # 计算所有样本的平均剪影系数。
    kl.append(k)

bestK = kl[sil.index(max(sil))]
print(bestK)
plt.plot(kl, sil)
plt.ylabel('Silhoutte Score')
plt.ylabel('K')
plt.show()

# 4. 使用最合适的K值进行聚类
k = bestK
kMeansModel = KMeans(n_clusters=k,  random_state=888)#构造聚类器
kMeansModel.fit(pred_images)#聚类
label_pred = kMeansModel.labels_  # 获取聚类标签
kPredictions = kMeansModel.predict(pred_images)
print(kPredictions)
#绘制 k-means结果
for j in range(0,k):
    imagesRes = images[label_pred == j]
    plt.scatter(imagesRes[:, 0], imagesRes[:, 1],  label=('label'+str(j)))
    plt.xlabel(' length')
    plt.ylabel(' width')
    plt.legend(loc=2)
plt.show()



# 5. 保存图像到不同类别的文件夹
for i in range(1,k+1):
    name="datasetsCatDog/class" + str(i)
    if os.path.isdir(name):
        #os.rmdir(name)# 删除目录 如果该目录非空则不能删除
        shutil.rmtree(name)# 删除目录 如果该目录非空也能删除
    os.mkdir("datasetsCatDog/class" + str(i))
for i in range(len(paths)):
    for j in range(0,k):
        if kPredictions[i] == j:
            shutil.copy(paths[i], "datasetsCatDog/class"+str(j+1))

