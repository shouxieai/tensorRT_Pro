import cv2
import numpy as np
import pycuda_resize as pr

image = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
opencv_resize = cv2.resize(image, (5, 10))
my_resize = pr.resize(image, (5, 10))
abs_diff = np.abs(opencv_resize.astype(int) - my_resize.astype(int))

print("===========================abs diff==============================")
print(abs_diff)

print("\n\n")
print("===========================OpenCV Resize==========================")
print(opencv_resize)
print("\n\n")
print("============================My Resize============================")
print(my_resize)

