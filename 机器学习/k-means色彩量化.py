import numpy as np
import cv2 as cv

img = cv.imread('82.jpg')
Z = img.reshape((-1,3))

# 转换为np.float32
Z = np.float32(Z)

# 定义标准、簇数(K)并应用kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

# 转换回uint8，原始图像
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv.imshow('res2',res2)
cv.imwrite('color quantification.jpg',res2)
cv.waitKey(0)
cv.destroyAllWindows()
