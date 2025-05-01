class Button:
    def __init__(self, pos, width, height, value):
        self.pos = pos
        self.width = width
        self.height = height
        self.value = value
    def draw(self, frame):
        cv2.rectangle(frame, (self.pos[0], self.pos[1]), (self.pos[0]+self.width, self.pos[1]+self.height), (225, 225, 225), -1)
        cv2.rectangle(frame, (self.pos[0], self.pos[1]), (self.pos[0]+self.width, self.pos[1]+self.height), (0, 0, 0), 2)
        cv2.putText(frame, self.value, (int(self.pos[0]+self.width/3), int(self.pos[1]+self.height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (25, 25, 25), 2)


import cv2
import numpy as np

button_value = [['7', '8', '9', '+'],
                ['4', '5', '6', '-'],
                ['1', '2', '3', '*'],
                ['0', '.', '=', '/']]

button_list = []
for x in range(4):
    for y in range(4):
        button_list.append(Button((x*50+900, y*50+280), 50, 50, button_value[y][x]))
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    Button((900, 280), 50, 50, '2').draw(frame)
    if ret == True:
        for button in button_list:
            button.draw(frame)
        # 画出计算器显示框
        cv2.rectangle(frame, (900, 220), (1100, 280), (225, 225, 225), -1)
        cv2.rectangle(frame, (900, 220), (1100, 280), (0, 0, 0,), 3)
        cv2.imshow('live', frame)
    key = cv2.waitKey(20)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()