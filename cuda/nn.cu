#include "nn.cuh"

void Conv2d::forward(FloatTensor &x, FloatTensor &out)
{
    const uint64_t B = x.shape().at(0);
    const uint64_t H = x.shape().at(2);
    const uint64_t W = x.shape().at(3);
    const auto [h, w, h_out, w_out] = out.shape().as_tuple<4>();
    const auto block_size = dim3(1, 1, 1);
    const auto blocks = dim3(w_out * h_out, B, out_channels);
    conv2dForwardKernel<<<blocks, block_size>>>(x.data(), out.data(), weight.data(), kernel_size,
                                                stride, padding, h_out, w_out, B, in_channels,
                                                out_channels, H, W);
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
}

void BatchNorm2d::forward(FloatTensor &x, FloatTensor &out)
{
    const auto [B, C, conv_w_out, conv_h_out] = x.shape().as_tuple<4>();
    const auto bn_block_size = dim3(8, 8, 16);
    const auto bn_blocks = dim3(CEIL(B, bn_block_size.x), CEIL(channels_num, bn_block_size.y),
                                CEIL(conv_w_out * conv_h_out, bn_block_size.z));
    batchNorm2dForwardKernel<<<bn_blocks, bn_block_size>>>(x.data(), out.data(), weight.data(),
                                                           bias.data(), mean.data(), var.data(), B,
                                                           channels_num, conv_w_out * conv_h_out);
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
}

void Pool2d::avgforward(FloatTensor &x, FloatTensor &out)
{
    const auto [B, C, H, W] = x.shape().as_tuple<4>();
    const auto out_h = out.shape().at(2), out_w = out.shape().at(3);
    const auto block_size = dim3(1, 1, 1);
    const auto blocks = dim3(H * W, B, C);
    avgPool2dKernel<<<blocks, block_size>>>(x.data(), out.data(), kernel_size, stride, padding,
                                            out_h, out_w, B, C, H, W);
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
}

void Pool2d::maxforward(FloatTensor &x, FloatTensor &out)
{
    const auto [B, C, H, W] = x.shape().as_tuple<4>();
    const auto out_h = out.shape().at(2), out_w = out.shape().at(3);
    const auto block_size = dim3(1, 1, 1);
    const auto blocks = dim3(H * W, B, C);
    maxPool2dKernel<<<blocks, block_size>>>(x.data(), out.data(), kernel_size, stride, padding,
                                            out_h, out_w, B, C, H, W);
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
}

void Linear::forward(FloatTensor &x, FloatTensor &out)
{
    const auto B = x.shape().at(0);
    const auto block_size = dim3(1, 1);
    const auto blocks = dim3(out_features, x.shape().at(0));
    linearForwardKernel<<<blocks, block_size>>>(x.data(), out.data(), weight.data(), bias.data(), B,
                                                in_features, out_features);
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
}

void reluForward(FloatTensor &x, FloatTensor &out)
{
    assert(x.shape() == out.shape());
    const auto n = x.numel();
    const auto block_size = dim3(1);
    const auto blocks = dim3(n);
    reluForwardKernel<<<blocks, block_size>>>(x.data(), out.data(), n);
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
}

void addForward(FloatTensor &a, FloatTensor &b, FloatTensor &out)
{
    assert(a.shape() == b.shape());
    assert(a.shape() == out.shape());
    const auto n = a.numel();
    const auto block_size = dim3(1);
    const auto blocks = dim3(n);
    addForwardKernel<<<blocks, block_size>>>(a.data(), b.data(), out.data(), n);
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
}
