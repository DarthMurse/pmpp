#include <torch/extension.h>

torch::Tensor naiveCudaMatMul(torch::Tensor x, torch::Tensor y);
torch::Tensor tileCudaMatMul(torch::Tensor x, torch::Tensor y);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("naiveCudaMatMul", &naiveCudaMatMul, "naive cuda matmul");
  m.def("tileCudaMatMul", &tileCudaMatMul, "tiled cuda matmul");
}