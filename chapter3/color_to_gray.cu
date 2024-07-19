#include <torch/extension.h>
#include <cuda.h>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void colorToGrayKernel(unsigned char *input, unsigned char *output, int H, int W)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < H && col < W)
    {
        int pos = row * W + col;
        output[pos] = 0.21 * input[pos] + 0.71 * input[H*W+pos] + 0.07 * input[2*H*W+pos];
    }
}

torch::Tensor colorToGrayCuda(torch::Tensor input)
{
    CHECK_INPUT(input);
    int C = input.size(0), H = input.size(1), W = input.size(2);
    auto result = torch::zeros({H, W}).to(input.type());
    result = result.to(input.device());
    dim3 block_dim(16, 16, 1), grid_dim(ceil(H/16.0), ceil(W/16.0), 1);
    colorToGrayKernel<<<grid_dim, block_dim>>>(input.data_ptr<unsigned char>(), result.data_ptr<unsigned char>(), H, W);
    return result;
}