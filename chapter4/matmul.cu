#include <torch/extension.h>
#include <cuda.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void naive_cuda_matmul_kernel(float *result, float *x, float *y, int m, int n, int k)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < m && col < n)
    {
        result[row * n + col] = 0;
        for (int i = 0; i < k; i++)
            result[row * n + col] += x[row * n + i] * y[i * n + col];
    }
}

torch::Tensor naiveCudaMatMul(torch::Tensor x, torch::Tensor y)
{
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    int row = x.size(0), col = y.size(1), k = x.size(1);
    auto result = torch::zeros({row, col}).to(x.type());
    result = result.to(x.device());
    dim3 block(16, 16, 1);
    dim3 grid(ceil(row / 16.0), ceil(row / 16.0), 1);
    naive_cuda_matmul_kernel<<<grid, block>>>(result.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>(), row, col, k);
    return result;
}

__global__ void tile_cuda_matmul_kernel()
{

}

torch::Tensor tileCudaMatMul(torch::Tensor x, torch::Tensor y)
{
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    int row = x.size(0), col = y.size(1), k = x.size(1);
    auto result = torch::zeros({row, col}).to(x.type());
    result = result.to(x.device());
    

    return result;
}