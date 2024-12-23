#ifndef CUDA_NN_CUH
#define CUDA_NN_CUH

#include "tensor.cuh"
#include "helpers.cuh"
#include "ops.cuh"

class Conv2d
{
public:
    Conv2d(FloatTensor weight, uint64_t in_channels, uint64_t out_channels, uint64_t kernel_size,
           uint64_t stride = 1, uint64_t padding = 0)
        : weight(std::move(weight)), in_channels(in_channels), out_channels(out_channels),
          kernel_size(kernel_size), stride(stride), padding(padding)
    {
    }

    static Conv2d loadWeightToCuda(std::string name, uint64_t in_channels, uint64_t out_channels,
                                   uint64_t kernel_size, uint64_t stride = 1, uint64_t padding = 0)
    {
        auto weight = FloatTensor::loadToCuda("weights_bin/" + name + ".weight")
                          .reshape(Shape({out_channels, in_channels, kernel_size, kernel_size}));
        // std::cout << "load " << "weights_bin/" + name + ".weight" << "\n";
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

    void forward(FloatTensor &x, FloatTensor &out);
};

class BatchNorm2d
{
public:
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
    
    void forward(FloatTensor &x, FloatTensor &out);
};

class Pool2d
{
public:
    Pool2d(uint64_t channels, uint64_t kernel_size, uint64_t stride = 1, uint64_t padding = 0)
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

    void maxforward(FloatTensor &x, FloatTensor &out);
    void avgforward(FloatTensor &x, FloatTensor &out);
};

class Linear
{
public:
    Linear(FloatTensor weight, FloatTensor bias, uint64_t in_features, uint64_t out_features)
        : weight(std::move(weight)), bias(std::move(bias)), in_features(in_features),
          out_features(out_features)
    {
        assert(this->weight.shape == Shape({out_features, in_features}));
        assert(this->bias.shape == Shape({out_features}));
    }

    static Linear loadWeightToCuda(std::string name, uint64_t in_features, uint64_t out_features)
    {
        auto weight = FloatTensor::loadToCuda("weights_bin/" + name + ".weight")
                          .reshape(Shape({out_features, in_features}));
        // std::cout << "load " << "weights_bin/" + name + ".weight" << "\n";
        auto bias =
            FloatTensor::loadToCuda("weights_bin/" + name + ".bias").reshape(Shape({out_features}));
        // std::cout << "load " << "weights_bin/" + name + ".bias" << "\n";
        return Linear(std::move(weight), std::move(bias), in_features, out_features);
    }

    FloatTensor weight, bias;
    const uint64_t in_features, out_features;

    Shape getOutShape(Shape x_shape)
    {
        assert(x_shape.size() == 2);
        assert(x_shape[1] == in_features);
        return Shape({x_shape.at(0), out_features});
    }
    
    void forward(FloatTensor &x, FloatTensor &out);
};

#endif // CUDA_NN_CUH