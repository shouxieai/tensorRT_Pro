
import typing
import numpy as np
import requests
import os
from enum import Enum

List  = typing.List
Tuple = typing.Tuple

torch            = None

def lazy_import():
    global torch

    if torch is not None:
        return

    import torch


class LogLevel(Enum):
    Debug   = 5
    Verbose = 4
    Info    = 3
    Warning = 2
    Error   = 1
    Fatal   = 0


class HostFloatPointer(object):
    ptr    : int
    def __getitem__(self, index)->float: ...

class DeviceFloatPointer(object):
    ptr    : int
    def __getitem__(self, index)->float: ...

class DataHead(object):
    Init   = 0
    Device = 1
    Host   = 2

class MixMemory(object):
    cpu    : HostFloatPointer
    gpu    : DeviceFloatPointer
    owner_cpu : bool
    owner_gpu : bool
    cpu_size  : int
    gpu_size  : int
    def __init__(self, cpu=0, cpu_size=0, gpu=0, gpu_size=0): ...
    def aget_cpu(self, size)->HostFloatPointer: ...
    def aget_gpu(self, size)->DeviceFloatPointer: ...
    def release_cpu(self): ...
    def release_gpu(self): ...
    def release_all(self): ...

class Tensor(object):
    shape  : List[int]
    ndim   : int
    stream : int
    workspace : MixMemory
    data   : MixMemory
    numpy  : np.ndarray
    empty  : bool
    numel  : int
    cpu    : HostFloatPointer
    gpu    : DeviceFloatPointer
    head   : DataHead
    def __init__(self, shape : List[int], data : MixMemory=None): ... 
    def to_cpu(self, copy_if_need=True): ...
    def to_gpu(self, copy_if_need=True): ...
    def resize(self, new_shape : List[int]): ...
    def resize_single_dim(self, idim:int, size:int): ...
    def count(self, start_axis:int=0)->int: ...
    def offset(self, indexs : List[int])->int: ...
    def cpu_at(self, indexs : List[int])->HostFloatPointer: ...
    def gpu_at(self, indexs : List[int])->DeviceFloatPointer: ...
    def reference_data(self, shape : List[int], cpu : int, cpu_size : int, gpu : int, gpu_size : int): ...

class Infer(object):
    stream         : int
    num_input      : int
    num_output     : int
    max_batch_size : int
    device         : int
    workspace      : MixMemory
    def __init__(self, file : str): ...
    def forward(self, sync : bool=True): ...
    def input(self, index : int = 0)->Tensor: ...
    def output(self, index : int = 0)->Tensor: ...
    def synchronize(self): ...
    def is_input_name(self, name)->bool: ...
    def is_output_name(self, name)->bool: ...
    def get_input_name(self, index=0)->str: ...
    def get_output_name(self, index=0)->str: ...
    def tensor(self, name)->Tensor: ...
    def print(self): ...
    def set_input(self, index : int, new_tensor : Tensor): ...
    def set_output(self, index : int, new_tensor : Tensor): ...
    def serial_engine(self)->bytes: ...

# 钩子函数的格式是，输入节点名称和shape，返回新的shape
def hook_reshape_layer_func(name : str, shape : List[int]): ...

# 注册编译onnx时的reshapelayer的钩子，一旦执行compileTRT后立即失效
def set_compile_hook_reshape_layer(func : hook_reshape_layer_func): ...

class Mode(Enum):
    FP32 : int = 0
    FP16 : int = 1
    INT8 : int = 2

class NormType(Enum):
    NONE      : int = 0
    MeanStd   : int = 1
    AlphaBeta : int = 2

class ChannelType(Enum):
    NONE      : int = 0
    Invert    : int = 1

class Norm(object):
    mean   : List[float]
    std    : List[float]
    alpha  : float
    beta   : float
    type   : NormType
    channel_type : ChannelType

    # out = (src * alpha - mean) / std
    @staticmethod
    def mean_std(mean : List[float], std : List[float], alpha : float = 1.0, channel_type : ChannelType = ChannelType.NONE): ...

    # out = src * alpha + beta
    @staticmethod
    def alpha_beta(alpha : float, beta : float, channel_type : ChannelType = ChannelType.NONE): ...

    @staticmethod
    def none(): ...

def set_device(device_id : int): ...
def get_device()->int : ...

class ModelSourceType(Enum):
    Caffe    = 0
    OnnX     = 1
    OnnXData = 2

class ModelSource(object):
    type       : ModelSourceType
    onnxmodel  : str
    descript   : str
    onnx_data  : bytes

    @staticmethod
    def from_onnx(file : str): ...

    @staticmethod
    def from_onnx_data(data : bytes): ...

class CompileOutputType(Enum):
    File    = 0
    Memory  = 1

class CompileOutput(object):
    type    : CompileOutputType
    data    : bytes
    file    : str

    @staticmethod
    def to_file(file): ...

    @staticmethod
    def to_memory(): ...
    
def compileTRT(
    max_batch_size               : int,
    source                       : ModelSource,
    saveto                       : CompileOutput,
    mode                         : Mode        = Mode.FP32,
    inputs_dims                  : np.ndarray  = np.array([], dtype=int),
    device_id                    : int         = 0,
    int8_norm                    : Norm        = Norm.none(),
    int8_preprocess_const_value  : int = 114,
    int8_image_directory         : str = ".",
    int8_entropy_calibrator_file : str = ""
)->bool: ...

class FallState(Enum):
    Fall      = 0
    Stand     = 1
    UnCertain = 2

class SharedFutureFallState(object):
    def get(self)->Tuple[FallState, float]:...

class SharedFutureAlphaPosePoints(object):
    def get(self)->np.ndarray: ...

class SharedFutureArcfaceFeature(object):
    def get(self)->np.ndarray: ...

class FaceBox(object):
    left       : float
    top        : float
    right      : float
    bottom     : float
    confidence : float
    landmark   : np.ndarray

class ObjectBox(object):
    left        : float
    top         : float
    right       : float
    bottom      : float
    confidence  : float
    class_label : int
    width       : float
    height      : float
    cx          : float
    cy          : float

class YoloType(Enum):
    V5         : int  =  0
    X          : int  =  1

class SharedFutureFaceBoxArray(object):
    def get(self)->List[FaceBox]: ...

class SharedFutureObjectBoxArray(object):
    def get(self)->List[ObjectBox]: ...

class Fall(object):
    valid : bool
    def __init__(self, engine : str, device_id : int = 0): ...
    def commit(self, keys : np.ndarray, box : List[int])->SharedFutureFallState: ...

class AlphaPose(object):
    valid : bool
    def __init__(self, engine : str, device_id : int = 0): ...
    def commit(self, image : np.ndarray, box : List[int])->SharedFutureAlphaPosePoints: ...

class Arcface(object):
    valid : bool
    def __init__(self, engine : str, device_id : int = 0): ...
    def commit(self, image : np.ndarray, landmark : np.ndarray)->SharedFutureArcfaceFeature: ...
    def face_alignment(self, image : np.ndarray, landmark : np.ndarray)->np.ndarray: ...

class Retinaface(object):
    valid : bool
    def __init__(self, engine : str, device_id : int = 0, confidence_threshold : float = 0.7, nms_threshold : float = 0.5): ...
    def commit(self, image : np.ndarray)->SharedFutureFaceBoxArray: ...
    def crop_face_and_landmark(self, image : np.ndarray, facebox : FaceBox, scale_box : float = 1.5)->np.ndarray: ...

class Scrfd(object):
    valid : bool
    def __init__(self, engine : str, device_id : int = 0, confidence_threshold : float = 0.7, nms_threshold : float = 0.5): ...
    def commit(self, image : np.ndarray)->SharedFutureFaceBoxArray: ...
    def crop_face_and_landmark(self, image : np.ndarray, facebox : FaceBox, scale_box : float = 1.5)->np.ndarray: ...

class Yolo(object):
    valid : bool
    def __init__(
        self, 
        engine : str, 
        type : YoloType = YoloType.V5, 
        device_id : int = 0, 
        confidence_threshold : float = 0.4,
        nms_threshold : float = 0.5
    ): ...
    def commit(self, image : np.ndarray)->SharedFutureObjectBoxArray: ...
    def crop_face_and_landmark(self, image : np.ndarray, facebox : FaceBox, scale_box : float = 1.5)->np.ndarray: ...

def random_color(idd : int)->Tuple[int, int, int]: ...
def set_log_level(level : LogLevel): ...

from .libtrtpyc import *

def onnx_hub(name):
    # arcface_iresnet50 ：人脸识别Arcface
    # mb_retinaface     ：人脸检测Retinaface
    # scrfd_2.5g_bnkps  ：人脸检测SCRFD小模型2.5G Flops
    # fall_bp           ：摔倒分类模型
    # sppe              ：人体关键点检测AlphaPose
    # yolov5m           ：yolov5 m模型，目标检测coco80类
    # yolox_m           ：yolox m模型，目标检测coco80类

    root = os.path.join(os.environ["HOME"], ".trtpy")
    if not os.path.exists(root):
        os.mkdir(root)

    local_file = os.path.join(root, f"{name}.onnx")
    if not os.path.exists(local_file):
        url        = f"http://zifuture.com:1556/fs/25.shared/{name}.onnx"
        remote     = requests.get(url)
        print(f"OnnxHub: download from {url}, to {local_file}")

        with open(local_file, "wb") as f:
            f.write(remote.content)
    return local_file


def reference_numpy_tensor(t, tensor):

    if tensor is None:
        return None

    if tensor.size == 0 or tensor.dtype != np.float32:
        raise TypeError("tensor Must float32 numpy.ndarray")

    tensor = np.ascontiguousarray(tensor)
    t.reference_data(tensor.shape, tensor.ctypes.data, tensor.size * 4, 0, 0)


def reference_torch_tensor(t, tensor):

    lazy_import()
    if tensor is None:
        return None

    if tensor.numel() == 0 or tensor.dtype != torch.float32:
        raise TypeError("Must float32 torch tensor")

    tensor = tensor.contiguous()
    if tensor.is_cuda:
        t.reference_data(tensor.shape, 0, 0, tensor.data_ptr(), tensor.numel() * 4)
    else:
        t.reference_data(tensor.shape, tensor.data_ptr(), tensor.numel() * 4, 0, 0)

def reference_tensor(t, tensor):

    if isinstance(tensor, np.ndarray):
        return reference_numpy_tensor(t, tensor)
    else:
        return reference_torch_tensor(t, tensor)


def infer_torch__call__(self : Infer, *args):

    lazy_import()
    templ  = args[0]
    stream = torch.cuda.current_stream().cuda_stream

    for index, x in enumerate(args):
        self.input(index).stream = stream
        reference_tensor(self.input(index), x)

    batch   = templ.size(0)
    device  = templ.device
    outputs = []
    for index in range(self.num_output):
        out_shape = self.output(index).shape
        out_shape[0] = batch
        out_tensor = torch.empty(out_shape, dtype=torch.float32, device=device)
        self.output(index).stream = stream
        reference_tensor(self.output(index), out_tensor)
        outputs.append(out_tensor)

    self.forward(False)

    if not templ.is_cuda:
        for index in range(self.num_output):
            self.output(index).to_cpu()

    if len(outputs) == 1:
        return outputs[0]

    return tuple(outputs)


def infer_numpy__call__(self : Infer, *args):

    templ = args[0]
    batch = templ.shape[0]
    assert batch <= self.max_batch_size, "Batch must be less max_batch_size"

    for index, x in enumerate(args):
        reference_tensor(self.input(index), x)

    outputs = []
    for index in range(self.num_output):
        out_shape = self.output(index).shape
        out_shape[0] = batch
        out_tensor = np.empty(out_shape, dtype=np.float32)
        reference_tensor(self.output(index), out_tensor)
        outputs.append(out_tensor)

    self.forward(False, True)

    for index in range(self.num_output):
        self.output(index).to_cpu()

    if len(outputs) == 1:
        return outputs[0]

    return tuple(outputs)


def infer__call__(self : Infer, *args):
    
    templ = args[0]
    if isinstance(templ, np.ndarray):
        return infer_numpy__call__(self, *args)
    else:
        return infer_torch__call__(self, *args)


Infer.__call__ = infer__call__



def normalize_numpy(norm : Norm, image):

    if norm.channel_type == ChannelType.Invert:
        image = image[..., ::-1]

    if image != np.float32:
        image = image.astype(np.float32)

    if norm.type == NormType.MeanStd:
        mean = np.array(norm.mean, dtype=np.float32)
        std = np.array(norm.std, dtype=np.float32)
        alpha = norm.alpha
        out = (image * alpha - mean) / std
    elif norm.type == NormType.AlphaBeta:
        out = image * norm.alpha + norm.beta
    else:
        out = image
    return np.expand_dims(out.transpose(2, 0, 1), 0)


def normalize_torch(norm : Norm, image):

    lazy_import()
    if norm.channel_type == ChannelType.Invert:
        image = image[..., [2, 1, 0]]

    if image != torch.float32:
        image = image.float()

    if norm.type == NormType.MeanStd:
        mean = torch.tensor(norm.mean, dtype=torch.float32, device=image.device)
        std = torch.tensor(norm.std, dtype=torch.float32, device=image.device)
        alpha = norm.alpha
        out = (image * alpha - mean) / std
    elif norm.type == NormType.AlphaBeta:
        out = image * norm.alpha + norm.beta
    else:
        out = image
    return out.permute(2, 0, 1).unsqueeze(0)


def normalize(norm : Norm, image):
    if isinstance(image, np.ndarray):
        return normalize_numpy(norm, image)
    else:
        return normalize_torch(norm, image)


class MemoryData(object):

    def __init__(self):
        self.data = None

    def write(self, data):
        if self.data is None:
            self.data = data
        else:
            self.data += data

    def flush(self):
        pass


def compile_onnx_to_file(
    max_batch_size               : int,
    file                         : str,
    saveto                       : str,
    mode                         : Mode        = Mode.FP32,
    inputs_dims                  : np.ndarray  = np.array([], dtype=int),
    device_id                    : int         = 0,
    int8_norm                    : Norm        = Norm.none(),
    int8_preprocess_const_value  : int = 114,
    int8_image_directory         : str = ".",
    int8_entropy_calibrator_file : str = ""
)->bool:
    return compileTRT(
        max_batch_size               = max_batch_size,
        source                       = ModelSource.from_onnx(file),
        output                       = CompileOutput.to_file(saveto),
        mode                         = mode,
        inputs_dims                  = inputs_dims,
        device_id                    = device_id,
        int8_norm                    = int8_norm,
        int8_preprocess_const_value  = int8_preprocess_const_value,
        int8_image_directory         = int8_image_directory,
        int8_entropy_calibrator_file = int8_entropy_calibrator_file,
    )

def compile_onnxdata_to_memory(
    max_batch_size               : int,
    data                         : bytes,
    mode                         : Mode        = Mode.FP32,
    inputs_dims                  : np.ndarray  = np.array([], dtype=int),
    device_id                    : int         = 0,
    int8_norm                    : Norm        = Norm.none(),
    int8_preprocess_const_value  : int = 114,
    int8_image_directory         : str = ".",
    int8_entropy_calibrator_file : str = ""
)->bytes:
    mem     = CompileOutput.to_memory()
    success = compileTRT(
        max_batch_size               = max_batch_size,
        source                       = ModelSource.from_onnx_data(data),
        output                       = mem,
        mode                         = mode,
        inputs_dims                  = inputs_dims,
        device_id                    = device_id,
        int8_norm                    = int8_norm,
        int8_preprocess_const_value  = int8_preprocess_const_value,
        int8_image_directory         = int8_image_directory,
        int8_entropy_calibrator_file = int8_entropy_calibrator_file,
    )

    if not success:
        return None

    return mem.data


def convert_torch_to_trt(torch_model, input, max_batch_size=None)->Infer:

    lazy_import()
    if isinstance(input, torch.Tensor):
        input = (input,)

    assert isinstance(input, tuple) or isinstance(input, list), "Input must tuple or list"
    input = tuple(input)

    if max_batch_size is None:
        max_batch_size = input[0].size(0)

    onnx_data  = MemoryData()
    torch.onnx.export(torch_model, input, onnx_data, opset_version=11, enable_onnx_checker=False)
    model_data = compile_onnxdata_to_memory(max_batch_size, onnx_data.data)
    trt_model = load_infer_data(model_data)
    trt_model.stream = torch.cuda.current_stream().cuda_stream
    return trt_model

def upbound(value, align=32):
    return (value + align - 1) // align * align

RETINFACE_NORM = Norm.mean_std([104, 117, 123], [1, 1, 1], 1.0, ChannelType.NONE)
YOLOV5_NORM    = Norm.alpha_beta(1 / 255.0, 0.0, ChannelType.Invert)
YOLOX_NORM     = Norm.mean_std([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 1.0 / 255.0, ChannelType.Invert)
ALPHAPOSE_NORM = Norm.mean_std([0.406, 0.457, 0.480], [1, 1, 1], 1.0 / 255.0, ChannelType.Invert)
ARCFACE_NORM   = Norm.mean_std([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 1.0 / 255.0, ChannelType.NONE)
