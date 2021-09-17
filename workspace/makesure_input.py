import numpy as np

def load_tensor(file):
    
    with open(file, "rb") as f:
        binary_data = f.read()

    magic_number, ndims, dtype = np.frombuffer(binary_data, np.uint32, count=3, offset=0)
    assert magic_number == 0xFCCFE2E2, f"{file} not a tensor file."
    
    dims = np.frombuffer(binary_data, np.uint32, count=ndims, offset=3 * 4)

    if dtype == 0:
        np_dtype = np.float32
    elif dtype == 1:
        np_dtype = np.float16
    else:
        assert False, f"Unsupport dtype = {dtype}, can not convert to numpy dtype"
        
    return np.frombuffer(binary_data, np_dtype, offset=(ndims + 3) * 4).reshape(*dims)


tensor = load_tensor("demo.tensor")
ts = [tensor[0, i*3:(i+1)*3].transpose(1, 2, 0) for i in range(4)]
out = np.zeros((640, 640, 3))
out[::2, ::2, :] = ts[0]
out[1::2, ::2, :] = ts[1]
out[::2, 1::2, :] = ts[2]
out[1::2, 1::2, :] = ts[3]
print(out.shape)

import cv2
cv2.imwrite("demo.jpg", (out * 255).astype(np.uint8))


m = np.array([
    [0.5, 0, -8],
    [0, 0.5, -2]
])
print(cv2.invertAffineTransform(m))