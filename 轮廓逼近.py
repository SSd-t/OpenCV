import cv2
import numpy as np
img = cv2.imread('photo/hand.png')
img_copy = img.copy()
img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
thresh, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_copy, contours, 0, (0, 0, 255), 2)
# 使用多边形逼近
approx = cv2.approxPolyDP(contours[0], 10, closed=True)
cv2.drawContours(img_copy, [approx], 0, (0, 255, 0), 2)
hull = cv2.convexHull(contours[0])
cv2.drawContours(img_copy, [hull], 0, (255, 0, 0), 2)
cv2.imshow('img', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()