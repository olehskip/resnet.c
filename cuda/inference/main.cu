#include <fstream>
#include <iomanip>

#include "helpers.cuh"
#include "ops.cuh"

enum class Device
{
    CPU,
    GPU
};

template <class T>
struct Array
{
    Array(uint64_t numel, Device device = Device::CPU)
        : numel(numel), size(numel * sizeof(T)), device(device)
    {
        if (numel != 0) {
            switch (device) {
                case Device::CPU: {
                    data = (T *)malloc(size);
                    break;
                }
                case Device::GPU: {
                    data = (T *)safeCudaMalloc(size);
                    break;
                }
                default:
                    assert("not implemented");
            }
        } else {
            data = NULL;
        }
    }
    ~Array()
    {
        if (numel != 0) {
            switch (device) {
                case Device::CPU: {
                    free(data);
                    break;
                }
                case Device::GPU: {
                    // TODO
                    break;
                }
                default:
                    assert("not implemented");
            }
        } else {
            data = NULL;
        }
    }
    const uint64_t numel, size;
    T *data;
    const Device device = Device::CPU;

    Array<T> copyTo(Device new_device)
    {
        Array<T> ret(numel, new_device);
        if (device == Device::CPU && new_device == Device::GPU) {
            cudaMemcpy(ret.data, data, size, cudaMemcpyHostToDevice);
        } else if (device == Device::GPU && new_device == Device::CPU) {
            cudaMemcpy(ret.data, data, size, cudaMemcpyDeviceToHost);
        } else {
            assert("not implemented");
        }
        return ret;
    }
};
using FloatArray = Array<float>;

template <class T>
Array<T> loadArray(std::string file_name)
{
    std::ifstream file(file_name, std::ios::binary);
    assert(file.is_open());

    file.seekg(0, std::ios::end);
    const std::streampos file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    const uint64_t n = file_size / sizeof(T);
    assert(n > 0);

    Array<T> out(n, Device::CPU);
    file.read(reinterpret_cast<char *>(out.data), file_size);
    assert(!file.fail());
    file.close();
    return out;
}

template <class T>
Array<T> loadArrayToCuda(std::string file_name)
{
    return loadArray<T>(file_name).copyTo(Device::GPU);
}

template <class T>
void saveArray(std::string file_name, Array<T> &array)
{
    assert(array.device == Device::CPU);
    std::ofstream file(file_name, std::ios::binary);
    assert(file.is_open());

    file.write(reinterpret_cast<char *>(array.data), array.size);
    assert(!file.fail());
    file.close();
}

struct Conv2d
{
    Conv2d(FloatArray weight, FloatArray bias, uint64_t in_channels, uint64_t out_channels,
           uint64_t kernel_size, uint64_t stride, uint64_t padding)
        : weight(weight), bias(bias), in_channels(in_channels), out_channels(out_channels),
          kernel_size(kernel_size), stride(stride), padding(padding)
    {
    }
    FloatArray weight;
    FloatArray bias; // optional
    const uint64_t in_channels, out_channels, kernel_size, stride, padding;

    uint64_t out_side_size(uint64_t side_size)
    {
        return convOutputSize(side_size, kernel_size, stride, padding);
    }

    uint64_t out_numel(uint64_t b, uint64_t h, uint64_t w)
    {
        return b * out_channels * out_side_size(h) * out_side_size(w);
    }
};

struct BatchNorm2d
{
    BatchNorm2d(FloatArray weight, FloatArray bias, FloatArray mean, FloatArray var,
                uint64_t channels_num)
        : weight(weight), bias(bias), mean(mean), var(var), channels_num(channels_num)
    {
        assert(weight.numel == channels_num);
        assert(bias.numel == channels_num);
        assert(mean.numel == channels_num);
        assert(var.numel == channels_num);
    }
    FloatArray weight;
    FloatArray bias;
    FloatArray mean;
    FloatArray var;
    const uint64_t channels_num;
};

struct ResnetModel
{
    Conv2d conv1;
    BatchNorm2d bn1;
};

ResnetModel createResnet152()
{
    ResnetModel ret{.conv1 = Conv2d(loadArrayToCuda<float>("weights_bin/conv1.weight"),
                                    FloatArray(0), 3, 64, 7, 2, 3),
                    .bn1 = BatchNorm2d(loadArrayToCuda<float>("weights_bin/bn1.weight"),
                                       loadArrayToCuda<float>("weights_bin/bn1.bias"),
                                       loadArrayToCuda<float>("weights_bin/bn1.running_mean"),
                                       loadArrayToCuda<float>("weights_bin/bn1.running_var"), 64)};

    return ret;
}

void resnet_152_forward(ResnetModel &model, FloatArray x, uint64_t B, uint64_t C, uint64_t W,
                        uint64_t H)
{
    assert(B * C * W * H == x.numel);
    assert(x.device == Device::GPU);

    const uint64_t conv1_out_numel = model.conv1.out_numel(B, H, W);
    const auto w_out = model.conv1.out_side_size(W), h_out = model.conv1.out_side_size(H);
    FloatArray conv1_out = FloatArray(conv1_out_numel, Device::GPU);
    const auto conv1_block_size = dim3(8, 8, 16);
    const auto conv1_blocks =
        dim3(CEIL(B * w_out, conv1_block_size.x), CEIL(h_out, conv1_block_size.y),
             CEIL(model.conv1.out_channels, conv1_block_size.z));
    conv2dForwardKernel<<<conv1_blocks, conv1_block_size>>>(
        x.data, model.conv1.weight.data, conv1_out.data, model.conv1.kernel_size,
        model.conv1.stride, model.conv1.padding, h_out, w_out, B, model.conv1.in_channels,
        model.conv1.out_channels, H, W);
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
    std::cout << "conv1 kernel done\n";

    FloatArray bn1_out = FloatArray(conv1_out_numel, Device::GPU);
    const auto bn1_block_size = dim3(8, 8, 16);
    const auto bn1_blocks =
        dim3(CEIL(B, bn1_block_size.x), CEIL(model.bn1.channels_num, bn1_block_size.y),
             CEIL(w_out * h_out, bn1_block_size.z));
    batchNorm2dForwardKernel<<<bn1_blocks, bn1_block_size>>>(
        conv1_out.data, bn1_out.data, model.bn1.weight.data, model.bn1.bias.data,
        model.bn1.mean.data, model.bn1.var.data, B, model.bn1.channels_num, w_out * h_out);
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
    std::cout << "bn1 kernel done\n";
    
    FloatArray relu_out = FloatArray(conv1_out_numel, Device::GPU);
    const auto relu_block_size = dim3(1024);
    const auto relu_blocks =
        dim3(CEIL(B * model.bn1.channels_num * w_out * h_out, bn1_block_size.x));
    reluForwardKernel<<<relu_blocks, relu_block_size>>>(
        bn1_out.data, relu_out.data, B * model.bn1.channels_num * w_out * h_out);
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
    std::cout << "relu kernel done\n";

    FloatArray out = relu_out.copyTo(Device::CPU);
    saveArray("cuda_out.bin", out);
}

int main()
{
    std::cout << "Started\n";
    // linearTest();
    // conv2dTest();
    // reluTest();

    ResnetModel resnet_model = createResnet152();

    const uint64_t B = 2, C = 3, W = 224, H = 224;
    const uint64_t inp_numel = B * resnet_model.conv1.in_channels * W * H;
    FloatArray inp(inp_numel);
    for (uint64_t i = 0; i < inp_numel; ++i) {
        inp.data[i] = i;
    }
    FloatArray inp_cuda = inp.copyTo(Device::GPU);
    resnet_152_forward(resnet_model, inp_cuda, B, C, W, H);

    return 0;
}
