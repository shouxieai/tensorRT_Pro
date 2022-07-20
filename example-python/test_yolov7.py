import os
import numpy as np
import pytrt as tp
import cv2

# change current workspace
os.chdir("../workspace/")

# 如果执行出错，请删掉 ~/.pytrt 的缓存模型
# rm -rf ~/.pytrt，重新下载
device_id = 2
engine_file = "yolov7.fp32.trtmodel"
if not os.path.exists(engine_file):
    tp.compile_onnx_to_file(5, tp.onnx_hub("yolov7"), engine_file, device_id=device_id)

yolo   = tp.Yolo(engine_file, type=tp.YoloType.V7, device_id=device_id)
image  = cv2.imread("inference/car.jpg")
bboxes = yolo.commit(image).get()
print(f"{len(bboxes)} objects")
print(bboxes)

for box in bboxes:
    left, top, right, bottom = map(int, [box.left, box.top, box.right, box.bottom])
    cv2.rectangle(image, (left, top), (right, bottom), tp.random_color(box.class_label), 5)

os.makedirs("single_inference", exist_ok=True)
saveto = "single_inference/yolov7.car.jpg"
print(f"Save to {saveto}")

cv2.imwrite(saveto, image)

# 提交图像数组
futs = yolo.commit_array([image, image])
for fut in futs:
    print(fut.get())

# try:
#     import torch  # 如果出现invalid pointer错误，可以考虑把import torch放到import pytrt上面
#     image  = cv2.imread("inference/car.jpg")
#     device = 0

#     gpu_image = torch.from_numpy(image).to(device)
#     bboxes = yolo.commit_gpu(
#         pimage = gpu_image.data_ptr(),
#         width = image.shape[1],
#         height = image.shape[0],
#         device_id = device,
#         imtype = tp.ImageType.GPUBGR,
#         stream = torch.cuda.current_stream().cuda_stream
#     ).get()
#     print(f"{len(bboxes)} objects")
#     print(bboxes)

#     for box in bboxes:
#         left, top, right, bottom = map(int, [box.left, box.top, box.right, box.bottom])
#         cv2.rectangle(image, (left, top), (right, bottom), tp.random_color(box.class_label), 5)

#     os.makedirs("single_inference", exist_ok=True)
#     saveto = "single_inference/yolov7-gpuptr.car.jpg"
#     print(f"Save to {saveto}")

#     cv2.imwrite(saveto, image)

# except Exception as e:
#     print("GPUPtr test failed.", e)
