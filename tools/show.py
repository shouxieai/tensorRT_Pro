
# 配合/data/sxai/tensorRT/src/application/app_fall_recognize.cpp中的zmq remote实现远程显示服务器画面的效果
# pip install zmq
import zmq
import sys
import numpy as np
import cv2

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://192.168.16.109:15556")

while True:
    socket.send(b"a")
    message = socket.recv()
    if len(message) == 1 and message == b'x':
    	break

    image = np.frombuffer(message, dtype=np.uint8)
    image = cv2.imdecode(image, 1)
    
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
    	break