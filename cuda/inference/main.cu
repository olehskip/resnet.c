#include "ops.cuh"
#include "helpers.cuh"

void conv2dTest()
{
    const uint64_t B = 2;
    const uint64_t kernel_size = 2;
    const uint64_t in_channels = 1;
    const uint64_t out_channels = 2;
    const uint64_t W = 7, H = 7, stride = 1, padding = 0;

    const uint64_t weight_numel = out_channels * in_channels * kernel_size * kernel_size;
    float *weight_cuda = (float *)safeCudaMalloc(weight_numel * sizeof(float));
    float *weight = (float *)malloc(weight_numel * sizeof(float));
    for (uint64_t i = 0; i < weight_numel; ++i) {
        weight[i] = i;
    }
    std::cout << "weight:\n";
    for (uint64_t channel = 0; channel < in_channels; ++channel) {
        for (uint64_t i = 0; i < kernel_size; ++i) {
            for (uint64_t j = 0; j < kernel_size; ++j) {
                std::cout << weight[channel * kernel_size * kernel_size + i * kernel_size + j]
                          << " ";
            }
            std::cout << "\n";
        }
    }
    cudaMemcpy(weight_cuda, weight, weight_numel * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "weight.numel() = " << weight_numel << "\n";

    const uint64_t inp_numel = B * in_channels * W * H;
    float *inp_cuda = (float *)safeCudaMalloc(inp_numel * sizeof(float));
    float *inp = (float *)malloc(inp_numel * sizeof(float));
    for (uint64_t i = 0; i < inp_numel; ++i) {
        inp[i] = i;
    }
    std::cout << "inp:\n";
    for (uint64_t b = 0; b < B; ++b) {
        std::cout << "batch = " << b << ":\n";
        for (uint64_t channel = 0; channel < in_channels; ++channel) {
            for (uint64_t i = 0; i < H; ++i) {
                for (uint64_t j = 0; j < W; ++j) {
                    std::cout << inp[b * in_channels * W * H + channel * W * H + i * W + j] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }
    cudaMemcpy(inp_cuda, inp, inp_numel * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "inp.numel() = " << inp_numel << "\n";

    const uint64_t h_out = convOutputSize(H, kernel_size, stride, padding),
                   w_out = convOutputSize(W, kernel_size, stride, padding);
    std::cout << "h_out = " << h_out << " w_out = " << w_out << "\n";
    const uint64_t out_numel = B * out_channels * h_out * w_out;
    float *out_cuda = (float *)safeCudaMalloc(out_numel * sizeof(float));
    float *out = (float *)malloc(out_numel * sizeof(float));
    std::cout << "out.numel() = " << out_numel << "\n";

    const auto block_size = dim3(16, 16, 4);
    const auto blocks = dim3(CEIL(B * h_out, block_size.x), CEIL(w_out, block_size.y),
                             CEIL(out_channels, block_size.z));
    conv2dForwardKernel<<<blocks, block_size>>>(inp_cuda, weight_cuda, out_cuda, kernel_size,
                                                stride, padding, h_out, w_out, B, in_channels,
                                                out_channels, H, W);
    // maxPool2dKernel<<<blocks, block_size>>>(inp_cuda, out_cuda, kernel_size, stride, padding,
    //                                          h_out, w_out, B, in_channels, H, W);
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
    std::cout << "Finished kernel\n";

    cudaMemcpy(out, out_cuda, out_numel * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);

    std::cout << "out:\n";
    for (uint64_t b = 0; b < B; ++b) {
        std::cout << "batch = " << b << ":\n";
        for (uint64_t channel = 0; channel < out_channels; ++channel) {
            std::cout << "[\n";
            for (uint64_t i = 0; i < h_out; ++i) {
                std::cout << "[";
                for (uint64_t j = 0; j < w_out; ++j) {
                    std::cout << out[b * out_channels * h_out * w_out + channel * h_out * w_out +
                                     i * w_out + j]
                              << " ";
                }
                std::cout << "]\n";
            }
            std::cout << "]\n";
        }
        std::cout << "\n";
    }
    printf("done\n");
}

void linearTest()
{
    const uint64_t B = 3;
    const uint64_t C = 16, N = 8;

    const uint64_t weight_numel = C * N;
    float *weight_cuda = (float *)safeCudaMalloc(weight_numel * sizeof(float));
    float *weight = (float *)malloc(weight_numel * sizeof(float));
    for (uint64_t i = 0; i < weight_numel; ++i) {
        weight[i] = i;
    }
    std::cout << "weight:\n";
    for (uint64_t i = 0; i < C; ++i) {
        for (uint64_t j = 0; j < N; ++j) {
            std::cout << weight[i * N + j] << " ";
        }
        std::cout << "\n";
    }
    cudaMemcpy(weight_cuda, weight, weight_numel * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "weight.numel() = " << weight_numel << "\n";

    float *bias_cuda = (float *)safeCudaMalloc(N * sizeof(float));
    float *bias = (float *)malloc(N * sizeof(float));
    for (uint64_t i = 0; i < N; ++i) {
        bias[i] = i;
    }
    std::cout << "bias:\n";
    for (uint64_t i = 0; i < N; ++i) {
        std::cout << bias[i] << " ";
    }
    std::cout << "\n";
    cudaMemcpy(bias_cuda, bias, N * sizeof(float), cudaMemcpyHostToDevice);

    const uint64_t inp_numel = B * C;
    float *inp_cuda = (float *)safeCudaMalloc(inp_numel * sizeof(float));
    float *inp = (float *)malloc(inp_numel * sizeof(float));
    for (uint64_t i = 0; i < inp_numel; ++i) {
        inp[i] = i;
    }
    std::cout << "inp:\n";
    for (uint64_t b = 0; b < B; ++b) {
        std::cout << "batch = " << b << ":\n";
        for (uint64_t i = 0; i < C; ++i) {
            std::cout << inp[b * C + i] << " ";
        }
        std::cout << "\n";
    }
    cudaMemcpy(inp_cuda, inp, inp_numel * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "inp.numel() = " << inp_numel << "\n";

    const uint64_t out_numel = B * N;
    float *out_cuda = (float *)safeCudaMalloc(out_numel * sizeof(float));
    float *out = (float *)malloc(out_numel * sizeof(float));
    std::cout << "out.numel() = " << out_numel << "\n";

    const auto block_size = dim3(16, 16);
    const auto blocks = dim3(CEIL(B, block_size.x), CEIL(N, block_size.y));
    linearForwardKernel<<<blocks, block_size>>>(inp_cuda, weight_cuda, bias_cuda, out_cuda, N, B,
                                                C);
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
    std::cout << "Finished kernel\n";

    cudaMemcpy(out, out_cuda, out_numel * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);

    std::cout << "out:\n";
    for (uint64_t b = 0; b < B; ++b) {
        std::cout << "batch = " << b << ":\n";
        for (uint64_t j = 0; j < N; ++j) {
            std::cout << out[b * N + j] << " ";
        }
        std::cout << "\n";
    }
    printf("done\n");
}

int main()
{
    std::cout << "Started\n";
    // linearTest();
    conv2dTest();

    return 0;
}
