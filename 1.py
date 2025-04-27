import cv2
import numpy as np
img=cv2.imread("photo/zuomian.png")
cv2.imshow("img",img)
key=cv2.waitKey(0)
while True:
    if key==ord('q'):
        break
    elif key==ord('s'):
        cv2.imwrite("photo/suib.jpg", img)
        break
cv2.destroyAllWindows()
cv2.waitKey(0)
