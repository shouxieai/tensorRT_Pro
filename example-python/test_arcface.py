import os
import cv2
import numpy as np
import pytrt as tp

# change current workspace
os.chdir("../workspace/")

def compile_detector_model(width, height):

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

def compile_feature_model():
    
    engine_file = "arcface_iresnet50.FP32.trtmodel"
    if not os.path.exists(engine_file):
        tp.compile_onnx_to_file(5, tp.onnx_hub("arcface_iresnet50"), engine_file)
    return engine_file

def extract_feature_one(detector_model, feature_model, image, save_debug_name=None):
    faces = detector_model.commit(image).get()
    
    if len(faces) == 0:
        print("Can not detect any face")
        return None
    
    max_face = max(faces, key=lambda item: item.width * item.height)
    crop_image, face = detector_model.crop_face_and_landmark(image, max_face)
    feature = feature_model.commit(crop_image, face.landmark).get()
    
    if save_debug_name is not None:
        left, top, right, bottom = map(int, [face.left, face.top, face.right, face.bottom])
        cv2.rectangle(crop_image, (left, top), (right, bottom), (255, 0, 255), 5)

        for x, y in face.landmark.astype(int):
            cv2.circle(crop_image, (x, y), 3, (0, 255, 0), -1, 16)
        
        print(f"Save debug image to {save_debug_name}")
        cv2.imwrite(save_debug_name, crop_image)
    return feature


def cosine_distance(a, b):
    return float(a @ b.T)

detect_file    = compile_detector_model(640, 640)
arcface_file   = compile_feature_model()
detector_model = tp.Retinaface(detect_file, nms_threshold=0.4)
feature_model  = tp.Arcface(arcface_file)
image_a        = cv2.imread("face/library/2ys2.jpg")
image_b        = cv2.imread("face/library/2ys3.jpg")
image_c        = cv2.imread("face/library/male.jpg")

feature_a = extract_feature_one(detector_model, feature_model, image_a, "image_a.jpg")
feature_b = extract_feature_one(detector_model, feature_model, image_b, "image_b.jpg")
feature_c = extract_feature_one(detector_model, feature_model, image_c, "image_c.jpg")

print("a == b", cosine_distance(feature_a, feature_b))
print("a != c", cosine_distance(feature_a, feature_c))
print("b != c", cosine_distance(feature_b, feature_c))