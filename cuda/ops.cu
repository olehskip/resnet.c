#include "ops.cuh"

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

// (B, C) x (C, N) = (B, N)
// N - number of neurons, C - number of features
__global__ void linearForwardKernel(float *inp, float *weight, float *bias, float *out, uint64_t N,
                                    uint64_t B, uint64_t C)
{
    const uint64_t b = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64_t n = threadIdx.y + blockIdx.y * blockDim.y;
    if (b >= B || n >= N) {
        return;
    }
    float curr = 0;
    for (uint64_t i = 0; i < C; ++i) {
        curr += inp[b * C + i] * weight[i * N + n]; // inp[b][i] * weight[i][n]
    }
    out[b * N + n] = curr + bias[n]; // out[b][n]
}
