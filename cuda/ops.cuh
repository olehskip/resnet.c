#ifndef CUDA_OPS_CUH
#define CUDA_OPS_CUH

#include <cassert>
#include <cuda/std/limits>

#include "helpers.cuh"

__host__ __device__ inline uint64_t convOutputSize(uint64_t x, uint64_t kernel_size,
                                                   uint64_t stride, uint64_t padding)
{
    return (2 * padding + x - kernel_size) / stride + 1;
}

__global__ void conv2dForwardKernel(float *inp, float *weight, float *out, uint64_t kernel_size,
                                    uint64_t stride, uint64_t padding, uint64_t h_out,
                                    uint64_t w_out, uint64_t B, uint64_t in_channels,
                                    uint64_t out_channels, uint64_t H, uint64_t W);
__global__ void maxPool2dKernel(float *inp, float *out, uint64_t kernel_size, uint64_t stride,
                                uint64_t padding, uint64_t h_out, uint64_t w_out, uint64_t B,
                                uint64_t channels, uint64_t H, uint64_t W);
// (B, C) x (C, N) = (B, N)
// N - number of neurons, C - number of features
// adds bias if it's not NULL
__global__ void linearForwardKernel(float *inp, float *weight, float *bias, float *out, uint64_t N,
                                    uint64_t B, uint64_t C);
__global__ void reluForwardKernel(float *inp, float *out, uint64_t N);

__global__ void batchNorm2dForwardKernel(float *inp, float *out, float *weight, float *bias,
                                         float *mean, float *var, uint64_t B, uint64_t C,
                                         uint64_t N);

#endif // CUDA_OPS_CUH
