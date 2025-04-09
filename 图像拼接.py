import cv2
import numpy as np

img1 = cv2.imread('photo/pinjie1.png')
img2 = cv2.imread('photo/pinjie2.png')

# 设置同一个尺寸
img1 = cv2.resize(img1, (640, 480))
img2 = cv2.resize(img2, (640, 480))

# 灰度变换
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)
if len(good) > 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

# 计算单适应性矩阵
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSSAC, 5)


