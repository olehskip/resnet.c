#include <fstream>
#include <iomanip>
#include <numeric>
#include <optional>
#include <vector>

#include "helpers.cuh"
#include "ops.cuh"

enum class Device
{
    CPU,
    GPU
};

class Shape : public std::vector<uint64_t>
{
public:
    using std::vector<uint64_t>::vector;
    uint64_t numel() const
    {
        assert(!empty());
        return std::accumulate(begin(), end(), 1, [](auto a, auto b) { return a * b; });
    }
    template <std::size_t N>
    auto as_tuple() const
    {
        static_assert(N > 0, "Tuple size must be positive");

        if (size() != N) {
            std::abort();
        }

        return [this]<std::size_t... I>(std::index_sequence<I...>) {
            return std::make_tuple((*this)[I]...);
        }(std::make_index_sequence<N>{});
    }

    friend std::ostream &operator<<(std::ostream &os, const Shape &shape)
    {
        os << "(";
        for (size_t i = 0; i < shape.size(); ++i) {
            os << shape[i];
            if (i < shape.size() - 1) {
                os << ", ";
            }
        }
        os << ")";
        return os;
    }
};

template <class T>
struct Tensor
{
    Tensor() : device(Device::CPU)
    {
        shape = Shape({0});
        data = NULL;
    }

    Tensor(Device device) : device(device)
    {
        shape = Shape({0});
        data = NULL;
    }

    Tensor(Shape shape, Device device = Device::CPU) : shape(std::move(shape)), device(device)
    {
        assert(this->shape.size() != 0);
        if (numel() != 0) {
            switch (device) {
                case Device::CPU: {
                    data = (T *)malloc(size());
                    break;
                }
                case Device::GPU: {
                    data = (T *)safeCudaMalloc(size());
                    break;
                }
                default:
                    assert("not implemented");
            }
        } else {
            data = NULL;
        }
    }

    Tensor(Tensor<T> &&another) : shape(another.shape), device(another.device)
    {
        assert(device == another.device);
        data = another.data;
        shape = std::move(another.shape);
        assert(shape.size() != 0);
        another.data = NULL;
        another.shape = Shape({0});
    }

    ~Tensor()
    {
        clear();
    }

    static Tensor load(std::string file_name)
    {
        std::ifstream file(file_name, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Can't open " << file_name << std::endl;
            std::abort();
        }

        file.seekg(0, std::ios::end);
        const std::streampos file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        const uint64_t n = file_size / sizeof(T);
        assert(n > 0);

        const auto abc = Shape({n});
        Tensor<T> out(Shape({n}), Device::CPU);
        file.read(reinterpret_cast<char *>(out.data), file_size);
        assert(!file.fail());
        file.close();
        return out;
    }

    Tensor<T> cuda()
    {
        return copyTo(Device::GPU);
    }

    static Tensor<T> loadToCuda(std::string file_name)
    {
        return Tensor<T>::load(file_name).cuda();
    }

    void save(std::string file_name)
    {
        assert(device == Device::CPU);
        std::ofstream file(file_name, std::ios::binary);
        assert(file.is_open());

        file.write(reinterpret_cast<char *>(data), size());
        assert(!file.fail());
        file.close();
    }

    void reshape(Shape new_shape)
    {
        assert(new_shape.size() != 0);
        assert(shape.numel() == new_shape.numel());
        shape = new_shape;
    }

    uint64_t numel() const
    {
        return shape.numel();
    }

    uint64_t size()
    {
        return numel() * sizeof(T);
    }

    Shape shape;
    T *data;
    const Device device = Device::CPU;

    Tensor<T> copyTo(Device new_device)
    {
        Tensor<T> ret(shape, new_device);
        if (device == Device::CPU && new_device == Device::GPU) {
            cudaMemcpy(ret.data, data, size(), cudaMemcpyHostToDevice);
        } else if (device == Device::GPU && new_device == Device::CPU) {
            cudaMemcpy(ret.data, data, size(), cudaMemcpyDeviceToHost);
        } else {
            assert("not implemented");
        }
        return ret;
    }

    void operator=(const Tensor<T> &) = delete;
    void operator=(Tensor<T> &&another)
    {
        clear();
        assert(device == another.device);
        data = another.data;
        shape = std::move(another.shape);
        assert(shape.size() != 0);
        another.data = NULL;
        another.shape = Shape({0});
    }

    void clear()
    {
        if (data && numel() != 0) {
            switch (device) {
                case Device::CPU: {
                    free(data);
                    break;
                }
                case Device::GPU: {
                    cudaFree(data);
                    break;
                }
                default:
                    assert("not implemented");
            }
        }
        data = NULL;
        shape = Shape({0});
    }

    explicit operator bool() const
    {
        return bool(data);
    }
};
using FloatTensor = Tensor<float>;

struct Conv2d
{
    Conv2d(FloatTensor weight, uint64_t in_channels, uint64_t out_channels, uint64_t kernel_size,
           uint64_t stride = 1, uint64_t padding = 0)
        : weight(std::move(weight)), in_channels(in_channels), out_channels(out_channels),
          kernel_size(kernel_size), stride(stride), padding(padding)
    {
    }

    static Conv2d loadWeightToCuda(std::string name, uint64_t in_channels, uint64_t out_channels,
                                   uint64_t kernel_size, uint64_t stride = 1, uint64_t padding = 0)
    {
        auto weight = FloatTensor::loadToCuda("weights_bin/" + name + ".weight");
        weight.reshape(Shape({out_channels, in_channels, kernel_size, kernel_size}));
        return Conv2d(std::move(weight), in_channels, out_channels, kernel_size, stride, padding);
    }

    FloatTensor weight;
    const uint64_t in_channels, out_channels, kernel_size, stride, padding;

    Shape getOutShape(Shape x_shape)
    {
        assert(x_shape.size() == 4);
        assert(x_shape[1] == in_channels);
        return Shape({x_shape[0], out_channels,
                      convOutputSize(x_shape[2], kernel_size, stride, padding),
                      convOutputSize(x_shape[3], kernel_size, stride, padding)});
    }
};

struct BatchNorm2d
{
    BatchNorm2d(FloatTensor &&weight, FloatTensor &&bias, FloatTensor &&mean, FloatTensor &&var,
                uint64_t channels_num)
        : weight(std::move(weight)), bias(std::move(bias)), mean(std::move(mean)),
          var(std::move(var)), channels_num(channels_num)
    {
        assert(this->weight.shape == Shape({channels_num}));
        assert(this->bias.shape == Shape({channels_num}));
        assert(this->mean.shape == Shape({channels_num}));
        assert(this->var.shape == Shape({channels_num}));
    }

    static BatchNorm2d loadWeightToCuda(std::string name, uint64_t channels_num)
    {
        return BatchNorm2d(FloatTensor::loadToCuda("weights_bin/" + name + ".weight"),
                           FloatTensor::loadToCuda("weights_bin/" + name + ".bias"),
                           FloatTensor::loadToCuda("weights_bin/" + name + ".running_mean"),
                           FloatTensor::loadToCuda("weights_bin/" + name + ".running_var"),
                           channels_num);
    }

    FloatTensor weight;
    FloatTensor bias;
    FloatTensor mean;
    FloatTensor var;
    const uint64_t channels_num;
};

struct MaxPool2d
{
    MaxPool2d(uint64_t channels, uint64_t kernel_size, uint64_t stride = 1, uint64_t padding = 0)
        : channels(channels), kernel_size(kernel_size), stride(stride), padding(padding)
    {
    }
    const uint64_t channels, kernel_size, stride, padding;
    uint64_t outSideSize(uint64_t side_size)
    {
        return convOutputSize(side_size, kernel_size, stride, padding);
    }

    Shape getOutShape(Shape x_shape)
    {
        assert(x_shape.size() == 4);
        assert(x_shape[1] == channels);
        return Shape({x_shape[0], channels,
                      convOutputSize(x_shape[2], kernel_size, stride, padding),
                      convOutputSize(x_shape[3], kernel_size, stride, padding)});
    }
};

struct ResnetModel
{
    Conv2d conv1;
    BatchNorm2d bn1;
    FloatTensor act1_out;

    MaxPool2d maxpool;
    FloatTensor maxpool_out;
};

ResnetModel createResnet152()
{
    ResnetModel ret{.conv1 = Conv2d::loadWeightToCuda("conv1", 3, 64, 7, 2, 3),
                    .bn1 = BatchNorm2d::loadWeightToCuda("bn1", 64),
                    .act1_out = FloatTensor(Device::GPU),
                    .maxpool = MaxPool2d(64, 3, 2, 1),
                    .maxpool_out = FloatTensor(Device::GPU)};
    return ret;
}

void resnet152Forward(ResnetModel &model, FloatTensor &x)
{
    const auto [B, C, H, W] = x.shape.as_tuple<4>();
    assert(x.device == Device::GPU);
    std::cout << "x.shape = " << x.shape << std::endl;

    const Shape conv1_out_shape = model.conv1.getOutShape(x.shape);
    const auto [h, w, conv1_h_out, conv1_w_out] = conv1_out_shape.as_tuple<4>();
    if (!model.act1_out) {
        model.act1_out = FloatTensor(conv1_out_shape, Device::GPU);
    }
    std::cout << "model.act1_out.shape = " << model.act1_out.shape << std::endl;

    const auto conv1_block_size = dim3(8, 8, 16);
    const auto conv1_blocks =
        dim3(CEIL(B * conv1_h_out, conv1_block_size.x), CEIL(conv1_w_out, conv1_block_size.y),
             CEIL(model.conv1.out_channels, conv1_block_size.z));
    conv2dForwardKernel<<<conv1_blocks, conv1_block_size>>>(
        x.data, model.act1_out.data, model.conv1.weight.data, model.conv1.kernel_size,
        model.conv1.stride, model.conv1.padding, conv1_h_out, conv1_w_out, B,
        model.conv1.in_channels, model.conv1.out_channels, H, W);
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
    std::cout << "conv1 kernel done\n";

    const auto bn1_block_size = dim3(8, 8, 16);
    const auto bn1_blocks =
        dim3(CEIL(B, bn1_block_size.x), CEIL(model.bn1.channels_num, bn1_block_size.y),
             CEIL(conv1_w_out * conv1_h_out, bn1_block_size.z));
    batchNorm2dForwardKernel<<<bn1_blocks, bn1_block_size>>>(
        model.act1_out.data, model.act1_out.data, model.bn1.weight.data, model.bn1.bias.data,
        model.bn1.mean.data, model.bn1.var.data, B, model.bn1.channels_num,
        conv1_w_out * conv1_h_out);
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
    std::cout << "bn1 kernel done\n";

    const auto relu_block_size = dim3(1024);
    const auto relu_blocks =
        dim3(CEIL(B * model.bn1.channels_num * conv1_w_out * conv1_h_out, bn1_block_size.x));
    reluForwardKernel<<<relu_blocks, relu_block_size>>>(model.act1_out.data, model.act1_out.data,
                                                        B * model.bn1.channels_num * conv1_w_out *
                                                            conv1_h_out);
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
    std::cout << "relu kernel done\n";

    const Shape maxpool_out_shape = model.maxpool.getOutShape(conv1_out_shape);
    if (!model.maxpool_out) {
        model.maxpool_out = FloatTensor(maxpool_out_shape, Device::GPU);
    }
    std::cout << "model.maxpool_out.shape = " << model.maxpool_out.shape << std::endl;

    const auto maxpool_h_out = maxpool_out_shape.at(2), maxpool_w_out = maxpool_out_shape.at(3);
    const auto maxpool_block_size = dim3(8, 8, 16);
    const auto maxpool_blocks =
        dim3(CEIL(B * conv1_w_out, maxpool_block_size.x), CEIL(conv1_h_out, maxpool_block_size.y),
             CEIL(model.bn1.channels_num, maxpool_block_size.z));
    maxPool2dKernel<<<maxpool_blocks, maxpool_block_size>>>(
        model.act1_out.data, model.maxpool_out.data, model.maxpool.kernel_size,
        model.maxpool.stride, model.maxpool.padding, maxpool_h_out, maxpool_w_out, B,
        model.bn1.channels_num, conv1_w_out, conv1_h_out);
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
    std::cout << "maxpool1 kernel done\n";

    FloatTensor out = model.maxpool_out.copyTo(Device::CPU);
    out.save("cuda_out.bin");
    std::cout << "Saved output" << std::endl;
}

int main()
{
    const uint64_t B = 2, /* C = 3, */ H = 224, W = 224;
    std::cout << "Started\n";
    // linearTest();
    // conv2dTest();
    // reluTest();

    ResnetModel resnet_model = createResnet152();
    std::cout << "created model\n";

    FloatTensor inp(Shape({B, resnet_model.conv1.in_channels, W, H}), Device::CPU);
    for (uint64_t i = 0; i < inp.numel(); ++i) {
        inp.data[i] = i;
    }
    FloatTensor inp_cuda = inp.cuda();
    resnet152Forward(resnet_model, inp_cuda);

    return 0;
}
