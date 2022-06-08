
import libffhdd as ffhdd
import cv2
import numpy as np
import ctypes
import os

demuxer = ffhdd.FFmpegDemuxer("exp/number100.mp4")
if not demuxer.valid:
    print("Load failed")
    exit(0)

output_bgr = True
decoder = ffhdd.CUVIDDecoder(
    bUseDeviceFrame=False,
    codec=demuxer.get_video_codec(),
    max_cache=-1,
    gpu_id=0,
    output_bgr=output_bgr
)
if not decoder.valid:
    print("Decoder failed")
    exit(0)

pps, size = demuxer.get_extra_data()
decoder.decode(pps, size, 0)

os.makedirs("imgs", exist_ok=True)

nframe = 0
while True:
    pdata, pbytes, time_pts, iskey, ok = demuxer.demux()
    nframe_decoded = decoder.decode(pdata, pbytes, time_pts)
    for i in range(nframe_decoded):
        # ptr, pts, idx = decoder.get_frame()
        # width    = decoder.get_width()
        # height   = decoder.get_height()
        # bytesize = decoder.get_frame_bytes()
        # nframe += 1

        # if output_bgr:
        #     data = (ctypes.c_ubyte * bytesize).from_address(ptr)
        #     image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        # else:
        #     align_width = (width + 16 - 1) // 16 * 16
        #     assert int(align_width * height * 1.5) == bytesize, "Worng align_width"

        #     data = (ctypes.c_ubyte * bytesize).from_address(ptr)
        #     data = np.frombuffer(data, dtype=np.uint8).reshape(int(height * 1.5), align_width)
        #     image = cv2.cvtColor(data, cv2.COLOR_YUV2BGR_NV12)

        # print(image.shape)
        # cv2.imwrite(f"imgs/data_{nframe:05d}.jpg", image)

        ptr, pts, idx, image = decoder.get_frame(return_numpy=True)
        nframe += 1

        if not output_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_NV12)

        print(image.shape)
        cv2.imwrite(f"imgs/data_{nframe:05d}.jpg", image)
    
    if pbytes <= 0:
        break

print(f"Frame. {nframe}")
