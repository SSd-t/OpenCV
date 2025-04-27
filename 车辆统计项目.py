import cv2
import numpy as np
# 加载视频
cap = cv2.VideoCapture('车流量训练视频/test_1.mp4')
# 去背景
mog = cv2.bgsegm.createBackgroundSubtractorMOG()
# 闭运算kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# 设置最小矩阵长宽
min_w = 120
min_h = 100
# 判定线高
line_high = 540
# 计算外接矩形中心点
def center(x, y, w, h):
     cx = int(x + w / 2)
     cy = int(y + h / 2)
     return cx, cy
# 偏移量
offest = 10
# 初始车的数量
car_num = 0
carc = []
# 循环读取视频帧
while True:
    ret, frame = cap.read()
    if ret == True:
        # 把原始帧灰度化
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 高斯去噪
        blur = cv2.GaussianBlur(frame_gray, (3,3), sigmaX=5)
        fgmask = mog.apply(blur)
        # 闭运算
        close = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        # 划判定线
        cv2.line(frame, (0, line_high), (1600, line_high), (0, 0, 255), 2)
        # 查找轮廓
        contours, _ = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 画出轮廓（最大外接矩形）
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # 判断长宽
            is_valid = (w >= min_w) & (h >= min_h)
            if not is_valid:
                continue
            cv2.rectangle(frame, (x, y), ((x+w), (y+h)), (0, 255, 0), 2)
            # 判断矩形中心点是否过判定线
            cpoint = center(x, y, w, h)
            carc.append(cpoint)
            cv2.circle(frame, (cpoint), 5, (255, 255, 0), -1)
            for (x, y) in carc:
                if y > (line_high - offest) and y < (line_high + offest):
                    car_num += 1
                    carc.remove((x, y))
                    print("车的数量：", car_num)
        cv2.putText(frame, 'car_num:' + str(car_num), (10, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 255, 255))
        cv2.imshow('video', frame)
    key = cv2.waitKey(50)
    # 按esc键退出播放
    if key == 27:
        break
# 释放资源
cap.release()
cv2.destroyAllWindows()

# 形态学识别车辆
