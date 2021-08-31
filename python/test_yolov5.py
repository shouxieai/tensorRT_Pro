import os
import cv2
import numpy as np
import trtpy as tp

# 把工作目录切换过去
os.chdir("../workspace/")

engine_file = "yolov5m.fp32.trtmodel"
if not os.path.exists(engine_file):
    tp.compile_onnx_to_file(5, tp.onnx_hub("yolov5m"), engine_file)

yolo   = tp.Yolo(engine_file, type=tp.YoloType.V5)
image  = cv2.imread("inference/car.jpg")
bboxes = yolo.commit(image).get()
print(f"{len(bboxes)} objects")

for box in bboxes:
    left, top, right, bottom = map(int, [box.left, box.top, box.right, box.bottom])
    cv2.rectangle(image, (left, top), (right, bottom), tp.random_color(box.class_label), 5)

saveto = "single_inference/yolov5.car.jpg"
print(f"Save to {saveto}")

cv2.imwrite(saveto, image)