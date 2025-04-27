import cv2
import numpy as np

img = cv2.imread('photo/chess.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.cornerHarris(img_gray, 2, 3, 0.04)
# 显示角点
img[dst > 0.01 * dst.max()] = [0, 0, 255]
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
