import torch
from torch.utils.data import DataLoader
import pycuda_resize as pr
import numpy as np
import cv2

class Dataset:
    def __getitem__(self, index):
        image = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        opencv_resize = cv2.resize(image, (33, 33))
        my_resize = pr.resize(image, (33, 33))
        return opencv_resize, my_resize
    
    def __len__(self):
        return 10
    
# reference: https://stackoverflow.com/a/55812288/8664574
# 如果有spawn，必须保证得有if __name__ == "__main__":作为入口
if __name__ == "__main__":

    # 这一句很关键，否则会报错cuda initialize failed
    torch.multiprocessing.set_start_method('spawn')
    dataset = Dataset()
    dataloader = DataLoader(dataset, 1, True, num_workers=2)
    
    for a, b in dataloader:
        absdiff = torch.abs(a.float() - b.float())
        print(f"sum = {absdiff.sum().item()}, max = {absdiff.max().item()}, shape = {list(absdiff.shape)}")