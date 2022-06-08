import os
import cv2
import numpy as np
import pytrt as tp

# change current workspace
os.chdir("../workspace/")

# 如果执行出错，请删掉 ~/.pytrt 的缓存模型
# rm -rf ~/.pytrt，重新下载
engine_file = "centernet_r18_dcn.fp32.trtmodel"
if not os.path.exists(engine_file):
    tp.compile_onnx_to_file(5, tp.onnx_hub("centernet_r18_dcn"), engine_file)

model   = tp.CenterNet(engine_file)
image  = cv2.imread("inference/car.jpg")
bboxes = model.commit(image).get()
print(f"{len(bboxes)} objects")

for box in bboxes:
    print(box)
    left, top, right, bottom = map(int, [box.left, box.top, box.right, box.bottom])
    cv2.rectangle(image, (left, top), (right, bottom), tp.random_color(box.class_label), 5)

os.makedirs("single_inference", exist_ok=True)
saveto = "single_inference/centernet.car.jpg"
print(f"Save to {saveto}")

cv2.imwrite(saveto, image)

