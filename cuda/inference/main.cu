#include <fstream>
#include <iomanip>

#include "helpers.cuh"
#include "ops.cuh"

uint64_t loadArray(std::string file_name, float **out)
{
    std::ifstream file(file_name, std::ios::binary);
    assert(file.is_open());

    file.seekg(0, std::ios::end);
    const std::streampos file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    const uint64_t n = file_size / sizeof(float);
    assert(n > 0);

    *out = (float *)malloc(n * sizeof(float));

    file.read(reinterpret_cast<char *>(*out), file_size);
    assert(!file.fail());
    file.close();
    return n;
}

void saveArray(std::string file_name, float *inp, uint64_t size)
{
    std::ofstream file(file_name, std::ios::binary);
    assert(file.is_open());

    file.write(reinterpret_cast<char *>(inp), size);
    assert(!file.fail());
    file.close();
}

void *copyArrayToCuda(float *inp, uint64_t size)
{
    float *out_cuda = (float *)safeCudaMalloc(size);
    cudaMemcpy(out_cuda, inp, size, cudaMemcpyHostToDevice);
    return out_cuda;
}

int main()
{
    std::cout << "Started\n";
    // linearTest();
    // conv2dTest();
    // reluTest();

    float *conv1_weights;
    const uint64_t conv1_weights_numel = loadArray("weights_bin/conv1.weight", &conv1_weights);
    std::cout << conv1_weights_numel << "\n";
    float *conv1_weights_cuda =
        (float *)copyArrayToCuda(conv1_weights, conv1_weights_numel * sizeof(float));

    const uint64_t in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3;
    const uint64_t B = 2, C = 3, W = 224, H = 224; 
    const uint64_t inp_numel = B * in_channels * W * H;
    float *inp = (float *)malloc(inp_numel * sizeof(float));
    for (uint64_t i = 0; i < inp_numel; ++i) {
        inp[i] = i;
    }
    float *inp_cuda = (float *)copyArrayToCuda(inp, inp_numel * sizeof(float));
    const uint64_t h_out = convOutputSize(H, kernel_size, stride, padding),
                   w_out = convOutputSize(W, kernel_size, stride, padding);
    const uint64_t out_numel = B * out_channels * h_out * w_out;
    float *out_cuda = (float *)safeCudaMalloc(out_numel * sizeof(float));
    float *out = (float *)malloc(out_numel * sizeof(float));
    const auto block_size = dim3(64, 64, 64);
    const auto blocks = dim3(CEIL(B * h_out, block_size.x), CEIL(w_out, block_size.y),
                             CEIL(out_channels, block_size.z));
    std::cout << blocks.x << " " << blocks.y << " " << blocks.z << "\n";
    conv2dForwardKernel<<<block_size, blocks>>>(inp_cuda, conv1_weights_cuda, out_cuda, kernel_size, stride, padding, h_out,
                        w_out, B, in_channels, out_channels, H, W);
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
    std::cout << "Finished kernel\n";

    cudaMemcpy(out, out_cuda, out_numel * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);

    saveArray("cuda_out.bin", out, out_numel * sizeof(float));

    return 0;
}
