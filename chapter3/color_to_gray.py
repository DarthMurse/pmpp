import torch, os, math
import torchvision as tv
import torchvision.transforms.functional as tvf
from torchvision import io
import matplotlib.pyplot as plt
import time

img = io.read_image("zzy.jpg")
small_img = tvf.resize(img, 300)

def timeCounter(f, **kwargs):
    start = time.time()
    result = f(**kwargs)
    end = time.time()
    return end - start, result

# Python Implementation
def colorToGrayPython(x):
    C, H, W = x.shape
    result = torch.zeros([H, W], dtype=x.dtype)
    for i in range(H):
        for j in range(W):
            result[i, j] = 0.21 * x[0, i, j] + 0.71 * x[1, i, j] + 0.07 * x[2, i, j]
    return result 

py_time, py_img = timeCounter(colorToGrayPython, x=small_img)
io.write_png(py_img.unsqueeze(0), "py_img.png")
print(f"python time: {py_time}s")

# CPP Implementation
from torch.utils.cpp_extension import load 
cpp_module = load("cpp_module", sources=["color_to_gray.cpp", "color_to_gray.cu"], verbose=True)

def cppWrapper(x):
    return cpp_module.colorToGrayCpp(x)

cpp_time, cpp_img = timeCounter(cppWrapper, x=small_img)
io.write_png(cpp_img.unsqueeze(0), "cpp_img.png")
print(f"CPP time: {cpp_time}s")

## CUDA Implementation
def cudaWrapper(x):
    x = x.cuda()
    out = cpp_module.colorToGrayCuda(x)
    return out.cpu()

cuda_time, cuda_img = timeCounter(cudaWrapper, x=small_img)
io.write_png(cuda_img.unsqueeze(0), "cuda_img.png")
print(f"Cuda time: {cuda_time}s")