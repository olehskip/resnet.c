#ifndef CUDA_TENSOR_CUH
#define CUDA_TENSOR_CUH

#include <numeric>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <cassert>
#include <cuda_runtime.h>

#include "helpers.cuh"

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

    Tensor(T *data, Shape shape, Device device = Device::CPU)
        : data(data), shape(std::move(shape)), device(device)
    {
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

    static Tensor<T> arange_cpu(Shape shape)
    {
        Tensor<T> tensor(shape, Device::CPU);
        const uint64_t end = shape.numel();
        for(uint64_t i = 0; i <= end; ++i) {
            tensor.data[i] = i;
        }

        return tensor;
    }

    static Tensor<T> ones_cpu(Shape shape)
    {
        Tensor<T> tensor(shape, Device::CPU);
        const uint64_t end = shape.numel();
        for(uint64_t i = 0; i <= end; ++i) {
            tensor.data[i] = 1;
        }

        return tensor;
    }

    static Tensor<T> load(std::string file_name)
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

    Tensor<T> reshape(Shape new_shape)
    {
        assert(new_shape.size() != 0);
        assert(shape.numel() == new_shape.numel());
        return Tensor<T>(data, new_shape, device);
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

    Tensor<T> toDevice(Device new_device)
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

    Tensor<T> cuda()
    {
        return toDevice(Device::GPU);
    }

    Tensor<T> cpu()
    {
        return toDevice(Device::CPU);
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
#endif // CUDA_TENSOR_CUH