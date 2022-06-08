import os
import cv2
import numpy as np
import pytrt as tp

# change current workspace
os.chdir("../workspace/")

def compile_model(width, height):

    width  = tp.upbound(width)
    height = tp.upbound(height)
    index_of_reshape_layer = 0
    def hook_reshape(name, shape):
        # print(name)
        # layerset = [
        #     "Reshape_100", "Reshape_104", "Reshape_108", 
        #     "Reshape_113", "Reshape_117", "Reshape_121", 
        #     "Reshape_126", "Reshape_130", "Reshape_134"
        # ]
        nonlocal index_of_reshape_layer
        strides = [8, 16, 32, 8, 16, 32, 8, 16, 32]
        index  = index_of_reshape_layer
        index_of_reshape_layer += 1

        stride = strides[index]
        return [-1, height * width // stride // stride * 2, shape[2]]

    engine_file = f"retinaface.{width}x{height}.fp32.trtmodel"
    if not os.path.exists(engine_file):

        tp.set_compile_hook_reshape_layer(hook_reshape)
        tp.compile_onnx_to_file(
            5, tp.onnx_hub("mb_retinaface"), engine_file, 
            inputs_dims=np.array([
                [1, 3, height, width]
            ], dtype=np.int32)
        )
    return engine_file


engine_file = compile_model(640, 640)
detector    = tp.Retinaface(engine_file, nms_threshold=0.4)
image       = cv2.imread("inference/group.jpg")
faces       = detector.commit(image).get()

for face in faces:
    left, top, right, bottom = map(int, [face.left, face.top, face.right, face.bottom])
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 255), 5)

    for x, y in face.landmark.astype(int):
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1, 16)

os.makedirs("single_inference", exist_ok=True)
saveto = "single_inference/retinaface.group.jpg"
print(f"{len(faces)} faces, Save to {saveto}")
cv2.imwrite(saveto, image)