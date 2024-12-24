#ifndef CUDA_TENSOR_CUH
#define CUDA_TENSOR_CUH

#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <vector>

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

    friend std::ostream &operator<<(std::ostream &os, const Shape &shape_)
    {
        os << "(";
        for (size_t i = 0; i < shape_.size(); ++i) {
            os << shape_[i];
            if (i < shape_.size() - 1) {
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
    Tensor(Device device) : device(device)
    {
        shape_ = Shape({0});
    }

    Tensor(Shape shape_, Device device = Device::CPU) : shape_(std::move(shape_)), device(device)
    {
        assert(this->shape_.size() != 0);
        if (numel() != 0) {
            switch (device) {
                case Device::CPU: {
                    data_smart = std::shared_ptr<T>((T *)malloc(size()), [](T *ptr) {
                        if (ptr) {
                            free(ptr);
                        }
                    });
                    break;
                }
                case Device::GPU: {
                    data_smart = std::shared_ptr<T>((T *)safeCudaMalloc(size()), [](T *ptr) {
                        if (ptr) {
                            cudaFree(ptr);
                            cudaDeviceSynchronize();
                        }
                    });
                    cudaDeviceSynchronize();
                    break;
                }
                default:
                    assert("not implemented");
            }
        }
    }

    Tensor(Tensor<T> &&another) : shape_(another.shape_), device(another.device)
    {
        assert(device == another.device);
        data_smart = another.data_smart;
        shape_ = std::move(another.shape_);
        assert(shape_.size() != 0);
    }

    static Tensor<T> arange_cpu(Shape shape_)
    {
        Tensor<T> tensor(shape_, Device::CPU);
        const uint64_t end = shape_.numel();
        for (uint64_t i = 0; i < end; ++i) {
            tensor.data_smart[i] = i;
        }

        return tensor;
    }

    static Tensor<T> ones_cpu(Shape shape_)
    {
        Tensor<T> tensor(shape_, Device::CPU);
        const uint64_t end = shape_.numel();
        for (uint64_t i = 0; i < end; ++i) {
            tensor.data_smart[i] = 1;
        }

        return tensor;
    }

    static Tensor<T> loadToCpu(std::string file_name)
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
        file.read(reinterpret_cast<char *>(out.data_smart.get()), file_size);
        assert(!file.fail());
        file.close();
        return out;
    }

    static Tensor<T> loadToCuda(std::string file_name)
    {
        return Tensor<T>::loadToCpu(file_name).cuda();
    }

    void save(std::string file_name)
    {
        assert(device == Device::CPU);
        std::ofstream file(file_name, std::ios::binary);
        assert(file.is_open());

        file.write(reinterpret_cast<char *>(data_smart.get()), size());
        assert(!file.fail());
        file.close();
    }

    Tensor<T> view(Shape new_shape)
    {
        assert(new_shape.size() != 0);
        assert(shape_.numel() == new_shape.numel());
        return Tensor<T>(data_smart, new_shape, device);
    }

    uint64_t numel() const
    {
        return shape_.numel();
    }

    uint64_t size()
    {
        return numel() * sizeof(T);
    }

    const Device device = Device::CPU;

    Tensor<T> toDevice(Device new_device)
    {
        cudaDeviceSynchronize();
        Tensor<T> ret(shape_, new_device);
        if (device == Device::CPU && new_device == Device::GPU) {
            cudaMemcpy(ret.data_smart.get(), data_smart.get(), size(), cudaMemcpyHostToDevice);
        } else if (device == Device::GPU && new_device == Device::CPU) {
            cudaMemcpy(ret.data_smart.get(), data_smart.get(), size(), cudaMemcpyDeviceToHost);
        } else {
            throw std::runtime_error("Unsupported device transfer combination");
        }

        cudaDeviceSynchronize();
        gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
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
        assert(device == another.device);
        data_smart = another.data_smart;
        shape_ = std::move(another.shape_);
        assert(shape_.size() != 0);
        another.data_smart = NULL;
        another.shape_ = Shape({0});
    }

    explicit operator bool() const
    {
        return bool(data_smart);
    }

    const Shape &shape() const
    {
        return shape_;
    }

    T *data() const
    {
        return data_smart.get();
    }

private:
    Tensor(std::shared_ptr<T> data_smart, Shape shape_, Device device = Device::CPU)
        : data_smart(data_smart), shape_(std::move(shape_)), device(device)
    {
    }

    Shape shape_;
    std::shared_ptr<T> data_smart;
};

using FloatTensor = Tensor<float>;
#endif // CUDA_TENSOR_CUH
