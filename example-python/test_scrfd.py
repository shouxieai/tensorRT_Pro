import os
import cv2
import numpy as np
import pytrt as tp

# change current workspace
os.chdir("../workspace/")
tp.set_log_level(tp.LogLevel.Verbose)

def compile_model(width, height):

    def hook_reshape(name, shape):
        layerset = [
            "Reshape_108", "Reshape_110", "Reshape_112", 
            "Reshape_126", "Reshape_128", "Reshape_130", 
            "Reshape_144", "Reshape_146", "Reshape_148"
        ]
        strides = [8, 8, 8, 16, 16, 16, 32, 32, 32]

        if name in layerset:
            index  = layerset.index(name)
            stride = strides[index]
            return [-1, height * width // stride // stride * 2, shape[2]]

        return shape

    engine_file = f"scrfd.{width}x{height}.fp32.trtmodel"
    if not os.path.exists(engine_file):

        tp.set_compile_hook_reshape_layer(hook_reshape)
        tp.compile_onnx_to_file(
            5, tp.onnx_hub("scrfd_2.5g_bnkps"), engine_file, 
            inputs_dims=np.array([
                [1, 3, height, width]
            ], dtype=np.int32)
        )
    return engine_file


engine_file = compile_model(640, 640)
detector    = tp.Scrfd(engine_file, nms_threshold=0.5)
image       = cv2.imread("inference/group.jpg")
faces       = detector.commit(image).get()

for face in faces:
    left, top, right, bottom = map(int, [face.left, face.top, face.right, face.bottom])
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 255), 5)

    for x, y in face.landmark.astype(int):
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1, 16)

os.makedirs("single_inference", exist_ok=True)
saveto = "single_inference/scrfd.group.jpg"
print(f"{len(faces)} faces, Save to {saveto}")
cv2.imwrite(saveto, image)