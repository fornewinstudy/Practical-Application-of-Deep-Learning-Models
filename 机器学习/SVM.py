#!/usr/bin/env python
import cv2 as cv
import numpy as np
SZ=20
bin_n = 16 # Number of bins

affine_flags = cv.WARP_INVERSE_MAP|cv.INTER_LINEAR

def deskew(img):
    # 计算图像的矩
    m = cv.moments(img)
    # 如果mu02接近于0，则返回原始图像
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    # 计算斜率
    skew = m['mu11']/m['mu02']
    # 定义变换矩阵
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    # 应用仿射变换来去斜图像
    img = cv.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

def hog(img):
    # 计算图像的梯度
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    # 将梯度转换为大小和方向
    mag, ang = cv.cartToPolar(gx, gy)
    # 将角度量化为16个bin值
    bins = np.int32(bin_n*ang/(2*np.pi))
    # 将每个bin值分配给相应的bin
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    # 计算每个bin值的直方图
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    # 将所有直方图连接成一个64位向量
    hist = np.hstack(hists)
    return hist


'''
使用HOG特征和SVM分类器来识别手写数字的例子。
它首先读取digits_1.png图像，然后将其分成50x100个单元格。
前50个单元格用于训练，后50个单元格用于测试。
接下来，它使用deskew函数去除图像中的斜率，并使用hog函数计算每个单元格的HOG特征。
然后，它将所有HOG特征连接成一个64位向量，并使用SVM分类器进行训练。
最后，它使用测试数据集对分类器进行测试，并计算准确率。
'''
img = cv.imread('digits_1.png',0)
if img is None:
    raise Exception("we need the digits.png image from samples/data here !")

cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]

# 前半部分是trainData，其余部分是testData
train_cells = [ i[:50] for i in cells ]
test_cells = [ i[50:] for i in cells]

deskewed = [list(map(deskew,row)) for row in train_cells]

hogdata = [list(map(hog,row)) for row in train_cells]
trainData = np.float32(hogdata).reshape(-1,64)
responses = np.repeat(np.arange(10),250)[:,np.newaxis]

svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)

svm.train(trainData, cv.ml.ROW_SAMPLE, responses)
svm.save('svm_data.dat')

deskewed = [list(map(deskew,row)) for row in test_cells]

hogdata = [list(map(hog,row)) for row in test_cells]
testData = np.float32(hogdata).reshape(-1,bin_n*4)
result = svm.predict(testData)[1]

mask = result==responses
correct = np.count_nonzero(mask)
print('准确率:',correct*100.0/result.size)
