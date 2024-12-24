#include "ops.cuh"

static inline __device__ uint64_t idx4d(uint64_t a, uint64_t b, uint64_t B, uint64_t c, uint64_t C,
                                        uint64_t d, uint64_t D)
{
    return ((a * B + b) * C * D) + c * D + d;
}

// static inline __device__ uint64_t idx3d(uint64_t a, uint64_t b, uint64_t B, uint64_t c, uint64_t C)
// {
//     return ((a * B + b) * C) + c;
// }

__global__ void conv2dForwardKernel(float *inp, float *out, float *weight, uint64_t kernel_size,
                                    uint64_t stride, uint64_t padding, uint64_t out_h,
                                    uint64_t out_w, uint64_t B, uint64_t in_channels,
                                    uint64_t out_channels, uint64_t H, uint64_t W)
{
    const uint64_t oh = blockIdx.x / out_w;
    const uint64_t ow = blockIdx.x % out_w;
    const uint64_t b = blockIdx.y;
    const uint64_t oc = blockIdx.z;
    if (b >= B || oc >= out_channels || oh >= out_h || ow >= out_w) {
        return;
    }

    const int64_t ih_start = oh * stride - padding;
    const int64_t iw_start = ow * stride - padding;
    float sum = 0;
    for (uint64_t ic = 0; ic < in_channels; ++ic) {
        for (uint64_t kh = 0; kh < kernel_size; ++kh) {
            for (uint64_t kw = 0; kw < kernel_size; ++kw) {
                const int64_t ih = ih_start + kh;
                const int64_t iw = iw_start + kw;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                    continue;
                }
                const uint64_t weight_idx = idx4d(oc, ic, in_channels, kh, kernel_size, kw,
                                                  kernel_size); // W[oc][ic][kh][kw]
                const uint64_t in_idx =
                    idx4d(b, ic, in_channels, ih, H, iw, W); // in[b][ic][ih][iw]
                sum += inp[in_idx] * weight[weight_idx];
            }
        }
    }
    const uint64_t out_idx = idx4d(b, oc, out_channels, oh, out_h, ow, out_w); // out[b][oc][oh][ow]
    out[out_idx] = sum;
}

__global__ void maxPool2dKernel(float *inp, float *out, uint64_t kernel_size, uint64_t stride,
                                uint64_t padding, uint64_t out_h, uint64_t out_w, uint64_t B,
                                uint64_t channels, uint64_t H, uint64_t W)
{
    const uint64_t oh = blockIdx.x / out_w;
    const uint64_t ow = blockIdx.x % out_w;
    const uint64_t b = blockIdx.y;
    const uint64_t c = blockIdx.z;
    if (b >= B || c >= channels || oh >= out_h || ow >= out_w) {
        return;
    }

    const int64_t ih_start = oh * stride - padding;
    const int64_t iw_start = ow * stride - padding;
    float mx = -cuda::std::numeric_limits<float>::infinity();
    for (uint64_t kh = 0; kh < kernel_size; ++kh) {
        for (uint64_t kw = 0; kw < kernel_size; ++kw) {
            const int64_t ih = ih_start + kh;
            const int64_t iw = iw_start + kw;
            if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                continue;
            }
            const uint64_t in_idx = idx4d(b, c, channels, ih, H, iw, W); // in[b][c][ih][iw]
            mx = fmax(mx, inp[in_idx]);
        }
    }
    const uint64_t out_idx = idx4d(b, c, channels, oh, out_h, ow, out_w); // out[b][c][oh][ow]
    out[out_idx] = mx;
}

__global__ void avgPool2dKernel(float *inp, float *out, uint64_t kernel_size, uint64_t stride,
                                uint64_t padding, uint64_t out_h, uint64_t out_w, uint64_t B,
                                uint64_t channels, uint64_t H, uint64_t W)
{
    const uint64_t oh = blockIdx.x / out_w;
    const uint64_t ow = blockIdx.x % out_w;
    const uint64_t b = blockIdx.y;
    const uint64_t c = blockIdx.z;
    if (b >= B || c >= channels || oh >= out_h || ow >= out_w) {
        return;
    }

    const int64_t ih_start = oh * stride - padding;
    const int64_t iw_start = ow * stride - padding;
    float sum = 0;
    for (uint64_t kh = 0; kh < kernel_size; ++kh) {
        for (uint64_t kw = 0; kw < kernel_size; ++kw) {
            const int64_t ih = ih_start + kh;
            const int64_t iw = iw_start + kw;
            if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                continue;
            }
            const uint64_t in_idx = idx4d(b, c, channels, ih, H, iw, W); // in[b][c][ih][iw]
            sum += inp[in_idx];
        }
    }
    const uint64_t out_idx = idx4d(b, c, channels, oh, out_h, ow, out_w); // out[b][c][oh][ow]
    out[out_idx] = sum / kernel_size / kernel_size;
}

__global__ void linearForwardKernel(float *inp, float *out, float *weight, float *bias, uint64_t B,
                                    uint64_t in_features, uint64_t out_features)
{
    const uint64_t b = blockIdx.y;
    const uint64_t out_feature = blockIdx.x;
    if (b >= B || out_feature >= out_features) {
        return;
    }
    float curr = 0;
    for (uint64_t in_feature = 0; in_feature < in_features; ++in_feature) {
        curr += inp[b * in_features + in_feature] *
                weight[out_feature * in_features +
                       in_feature]; // inp[b][in_feature] * weight[out_feature][in_feature]
    }
    if (bias) {
        curr += bias[out_feature];
    }
    out[b * out_features + out_feature] = curr; // out[b][out_feature]
}

__global__ void reluForwardKernel(float *inp, float *out, uint64_t N)
{
    const uint64_t n = blockIdx.x;
    if (n >= N) {
        return;
    }
    out[n] = fmax(inp[n], 0.f);
}

__global__ void batchNorm2dForwardKernel(float *inp, float *out, float *weight, float *bias,
                                         float *mean, float *var, uint64_t B, uint64_t C,
                                         uint64_t N)
{
    const uint64_t b = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64_t c = threadIdx.y + blockIdx.y * blockDim.y;
    const uint64_t n = threadIdx.z + blockIdx.z * blockDim.z;
    if (b >= B || c >= C || n >= N) {
        return;
    }
    out[b * C * N + c * N + n] =
        (inp[b * C * N + c * N + n] - mean[c]) / sqrt(var[c] + 1e-5) * weight[c] + bias[c];
}

__global__ void addForwardKernel(float *inp1, float *inp2, float *out, uint64_t N)
{
    const uint64_t n = blockIdx.x;
    if (n >= N) {
        return;
    }
    out[n] = inp1[n] + inp2[n];
}