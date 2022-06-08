import os
import cv2
import numpy as np
import pytrt as tp

# change current workspace
os.chdir("../workspace/")

# 如果执行出错，请删掉 ~/.pytrt 的缓存模型
# rm -rf ~/.pytrt，重新下载
engine_file = "yolox_m.fp32.trtmodel"
if not os.path.exists(engine_file):
    tp.compile_onnx_to_file(5, tp.onnx_hub("yolox_m"), engine_file)

yolo   = tp.Yolo(engine_file, type=tp.YoloType.X)
image  = cv2.imread("inference/car.jpg")
bboxes = yolo.commit(image).get()
print(f"{len(bboxes)} objects")

for box in bboxes:
    print(f"{box}")
    left, top, right, bottom = map(int, [box.left, box.top, box.right, box.bottom])
    cv2.rectangle(image, (left, top), (right, bottom), tp.random_color(box.class_label), 5)

os.makedirs("single_inference", exist_ok=True)
saveto = "single_inference/yolox.car.jpg"
print(f"Save to {saveto}")

cv2.imwrite(saveto, image)
