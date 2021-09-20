from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import torch

import os
import cv2

from opts import opts
from detectors.detector_factory import detector_factory

if __name__ == '__main__':
    opt = opts().init()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    model = detector.model.eval().to(opt.device) # Note that eval() should be set.
    
    dummy_input = torch.zeros((1, 3, 512, 512)).to(opt.device)

    """ 
    note that if you are exporting onnx, you need to use the DCN for onnx rather than the original one(modify the dcn_v2.py).
    comment the original and uncomment the DU. Remember to fill in the args required. Or you can run it in debug mode where
    all required args have been filled in.
    """
    torch.onnx.export(
    model, 
    (dummy_input,), 
    'latest_ctnet_r18_dcn.onnx', 
    input_names=["images"],
    # output_names=["regxy","wh","hm","pool_hm","output"], 
    output_names=["output"], 
    verbose=True, 
    opset_version=11,
    # dynamic_axes={"images": {0:"batch"}, "regxy": {0:"batch"}, "wh": {0:"batch"}, 
                                        #  "hm": {0:"batch"}, "pool_hm": {0:"batch"}, 
                                        #  "output":{0:"batch"}},
    dynamic_axes={"images": {0:"batch"}, "output":{0:"batch"}},
    enable_onnx_checker=False
)



