#include <cassert>
#include <cstdint>
#include <cuda/std/limits>
#include <iostream>

#define CEIL(a, b) ((a + b - 1) / b)

// source: https://stackoverflow.com/a/14038590
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__host__ __device__ inline uint64_t convOutputSize(uint64_t x, uint64_t kernel_size,
                                                   uint64_t stride, uint64_t padding)
{
    return (2 * padding + x - kernel_size) / stride + 1;
}

__global__ void conv2dForwardKernel(float *inp, float *weight, float *out, uint64_t kernel_size,
                                    uint64_t stride, uint64_t padding, uint64_t h_out,
                                    uint64_t w_out, uint64_t B, uint64_t in_channels,
                                    uint64_t out_channels, uint64_t H, uint64_t W)
{
    const uint64_t thread_x = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64_t x = thread_x % h_out;
    const uint64_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const uint64_t out_channel = threadIdx.z + blockIdx.z * blockDim.z;
    const uint64_t b = thread_x / h_out;
    if (x >= h_out || y >= w_out || out_channel >= out_channels || b >= B) {
        return;
    }

    const uint64_t x_out = x * stride;
    const uint64_t y_out = y * stride;
    float sum = 0;
    for (uint64_t in_channel = 0; in_channel < in_channels; ++in_channel) {
        for (uint64_t i = 0; i < kernel_size; ++i) {
            const uint64_t x_pad = x_out + i;
            if (x_pad < padding || x_pad - padding >= H) {
                continue;
            }
            for (uint64_t j = 0; j < kernel_size; ++j) {
                const uint64_t y_pad = y_out + j;
                if (y_pad < padding || y_pad - padding >= W) {
                    continue;
                }
                sum += inp[b * in_channels * H * W + in_channel * H * W + (x_pad - padding) * W +
                           y_pad - padding] *
                       weight[out_channel * in_channels * kernel_size * kernel_size +
                              in_channel * kernel_size * kernel_size + i * kernel_size + j];
            }
        }
    }
    out[b * out_channels * w_out * h_out + out_channel * h_out * w_out + x * w_out + y] = sum;
}

__global__ void maxPool2dKernel(float *inp, float *out, uint64_t kernel_size, uint64_t stride,
                                uint64_t padding, uint64_t h_out, uint64_t w_out, uint64_t B,
                                uint64_t channels, uint64_t H, uint64_t W)
{
    const uint64_t thread_x = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64_t x = thread_x % h_out;
    const uint64_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const uint64_t channel = threadIdx.z + blockIdx.z * blockDim.z;
    const uint64_t b = thread_x / h_out;
    if (x >= h_out || y >= w_out || channel >= channels || b >= B) {
        return;
    }

    const uint64_t x_out = x * stride;
    const uint64_t y_out = y * stride;
    for (uint64_t channel = 0; channel < channels; ++channel) {
        float mx = cuda::std::numeric_limits<float>::min();
        for (uint64_t i = 0; i < kernel_size; ++i) {
            const uint64_t x_pad = x_out + i;
            if (x_pad < padding || x_pad - padding >= H) {
                continue;
            }
            for (uint64_t j = 0; j < kernel_size; ++j) {
                const uint64_t y_pad = y_out + j;
                if (y_pad < padding || y_pad - padding >= W) {
                    continue;
                }
                mx = fmax(mx, inp[b * channels * H * W + channel * H * W + (x_pad - padding) * W +
                                  y_pad - padding]);
            }
        }
        out[b * channels * w_out * h_out + channel * h_out * w_out + x * w_out + y] = mx;
    }
}

void *safeCudaMalloc(uint64_t size)
{
    void *dest;
    gpuErrchk(cudaMalloc(&dest, size));
    return dest;
}

int main()
{
    std::cout << "Started\n";

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
    return 0;
}