
try:
    import pytrt as tp
except Exception as e:
    print("Can not import pytrt, please make pyinstall on tensorRT_Pro directory.")
    print(e)
    exit(0)

import libffhdd as ffhdd
import cv2
import numpy as np
import ctypes
import os

print("Create ffmpeg demuxer.")
demuxer = ffhdd.FFmpegDemuxer("exp/fall_video.mp4")
if not demuxer.valid:
    print("Load failed")
    exit(0)

engine_file = "yolox_m.fp32.trtmodel"
if not os.path.exists(engine_file):
    print("Compile onnx to tensorRT engine.")
    tp.compile_onnx_to_file(5, tp.onnx_hub("yolox_m"), engine_file)

print("Create yolox infer.")
gpu_id = 0
yolo = tp.Yolo(engine_file, type=tp.YoloType.X, device_id=gpu_id)

print("Create CUVID decoder.")
output_bgr = True
decoder = ffhdd.CUVIDDecoder(
    bUseDeviceFrame=True,
    codec=demuxer.get_video_codec(),
    max_cache=-1,
    gpu_id=gpu_id,
    output_bgr=output_bgr
)
if not decoder.valid:
    print("Decoder failed")
    exit(0)

pps, size = demuxer.get_extra_data()
decoder.decode(pps, size, 0)
os.makedirs("imgs", exist_ok=True)

print("While loop.")
nframe = 0
while True:
    pdata, pbytes, time_pts, iskeyframe, ok = demuxer.demux()
    nframe_decoded = decoder.decode(pdata, pbytes, time_pts)
    for i in range(nframe_decoded):
        ptr, pts, idx, image = decoder.get_frame(return_numpy=True)
        width    = decoder.get_width()
        height   = decoder.get_height()
        bytesize = decoder.get_frame_bytes()
        nframe += 1

        bboxes = yolo.commit_gpu(
            pimage = ptr,
            width = width,
            height = height,
            device_id = gpu_id,
            imtype = tp.ImageType.GPUBGR if output_bgr else tp.ImageType.GPUYUVNV12,
            stream = decoder.get_stream()
        ).get()

        if not output_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_NV12)

        for box in bboxes:
            left, top, right, bottom = map(int, [box.left, box.top, box.right, box.bottom])
            cv2.rectangle(image, (left, top), (right, bottom), tp.random_color(box.class_label), 5)

        cv2.imwrite(f"imgs/data_{nframe:05d}.jpg", image)
    
    if pbytes <= 0:
        break

print(f"Frame. {nframe}")
print("Program done.")