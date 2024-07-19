#include <torch/extension.h>
#include <stdio.h>

torch::Tensor colorToGrayCpp(torch::Tensor input)
{
    // printf("input size: %ld, %ld, %ld\n", input.size(0), input.size(1), input.size(2));
    int C = input.size(0), H = input.size(1), W = input.size(2);
    auto result = torch::zeros({H, W}).to(input.type());
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            result.index({i, j}) = 0.21 * input.index({0, i, j}) + 0.71 * input.index({1, i, j}) + 0.07 * input.index({2, i, j});
    return result;
}

torch::Tensor colorToGrayCuda(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("colorToGrayCpp", &colorToGrayCpp, "colorToGray Cpp");
  m.def("colorToGrayCuda", &colorToGrayCuda, "colorToGray Cuda");
}