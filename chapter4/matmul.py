import torch
import time
from torch.utils.cpp_extension import load

# Torch Matmul
def torchMatMul(x: torch.Tensor, y: torch.Tensor):
    x = x.cuda()
    y = y.cuda()
    out = torch.matmul(x, y)
    return out.cpu()

x = torch.rand([4, 4])
y = torch.rand([4, 4])
print("------------------- torch matmul -------------------------")
start = time.time()
out = torchMatMul(x, y)
end = time.time()
print(out)
print(f"time consumed: {end-start}s")

cuda_module = load(name="cuda_matmul", sources=["matmul.cpp", "matmul.cu"], verbose=True)
# Naive cuda matmul
def naiveCudaMatMul(x: torch.Tensor, y: torch.Tensor):
    x = x.cuda()
    y = y.cuda()
    out = cuda_module.naiveCudaMatMul(x, y)
    return out.cpu()

print("------------------- naive cuda matmul -------------------------")
start = time.time()
out = naiveCudaMatMul(x, y)
end = time.time()
print(out)
print(f"time consumed: {end-start}s")