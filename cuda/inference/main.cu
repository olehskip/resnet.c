#include <optional>

#include "ops.cuh"
#include "tensor.cuh"
#include "nn.cuh"

struct Downsample
{
    Downsample(Conv2d &&conv, BatchNorm2d &&bn)
        : conv(std::move(conv)), bn(std::move(bn)), act(FloatTensor(Device::GPU))
    {
    }
    Conv2d conv;
    BatchNorm2d bn;
    FloatTensor act;
};

struct ResnetBlock
{
    ResnetBlock(Conv2d &&conv1, BatchNorm2d &&bn1, Conv2d &&conv2, BatchNorm2d &&bn2,
                Conv2d &&conv3, BatchNorm2d &&bn3, std::optional<Downsample> downsample,
                uint64_t in_channels, uint64_t inter_channels, uint64_t out_channels,
                uint64_t stride)
        : in_channels(in_channels), inter_channels(inter_channels), out_channels(out_channels),
          stride(stride), conv1(std::move(conv1)), bn1(std::move(bn1)), conv2(std::move(conv2)),
          bn2(std::move(bn2)), conv3(std::move(conv3)), bn3(std::move(bn3)),
          downsample(std::move(downsample)), act1_out(FloatTensor(Device::GPU)),
          act2_out(FloatTensor(Device::GPU)), act3_out(FloatTensor(Device::GPU))
    {
    }
    const uint64_t in_channels, inter_channels, out_channels, stride; // remove?

    Conv2d conv1;
    BatchNorm2d bn1;
    FloatTensor act1_out;

    Conv2d conv2;
    BatchNorm2d bn2;
    FloatTensor act2_out;

    Conv2d conv3;
    BatchNorm2d bn3;
    FloatTensor act3_out;

    std::optional<Downsample> downsample;
};

struct Layer
{
    std::vector<ResnetBlock> blocks;
};

Layer createLayer(uint64_t layer_id, uint64_t in_channels, uint64_t inter_channels,
                  uint64_t out_channels, uint64_t n_blocks, uint64_t stride = 1)
{
    auto loadResnetBlock = [layer_id](uint64_t block_id, uint64_t in_channels,
                                      uint64_t inter_channels, uint64_t out_channels,
                                      uint64_t stride) -> ResnetBlock {
        const std::string common_name =
            "layer" + std::to_string(layer_id) + "." + std::to_string(block_id) + ".";
        Conv2d conv1 =
            Conv2d::loadWeightToCuda(common_name + "conv1", in_channels, inter_channels, 1);
        BatchNorm2d bn1 = BatchNorm2d::loadWeightToCuda(common_name + "bn1", inter_channels);
        Conv2d conv2 = Conv2d::loadWeightToCuda(common_name + "conv2", inter_channels,
                                                inter_channels, 3, stride, 1);
        BatchNorm2d bn2 = BatchNorm2d::loadWeightToCuda(common_name + "bn2", inter_channels);
        Conv2d conv3 =
            Conv2d::loadWeightToCuda(common_name + "conv3", inter_channels, out_channels, 1);
        BatchNorm2d bn3 = BatchNorm2d::loadWeightToCuda(common_name + "bn3", out_channels);
        std::optional<Downsample> downsample = {};
        if (block_id == 0 && (stride != 1 || in_channels != out_channels)) {
            downsample.emplace(Downsample(
                Conv2d::loadWeightToCuda(common_name + "downsample.0", in_channels, out_channels, 1,
                                         stride),
                BatchNorm2d::loadWeightToCuda(common_name + "downsample.1", out_channels)));
        }
        return ResnetBlock(std::move(conv1), std::move(bn1), std::move(conv2), std::move(bn2),
                           std::move(conv3), std::move(bn3), std::move(downsample), in_channels,
                           inter_channels, out_channels, stride);
    };
    std::vector<ResnetBlock> blocks;
    blocks.emplace_back(loadResnetBlock(0, in_channels, inter_channels, out_channels, stride));
    for (uint64_t i = 1; i < n_blocks; ++i) {
        blocks.emplace_back(loadResnetBlock(i, out_channels, inter_channels, out_channels, 1));
    }
    return Layer{
        .blocks = std::move(blocks),
    };
}

struct ResnetModel
{
    Conv2d conv1;
    BatchNorm2d bn1;
    FloatTensor act1_out;

    Pool2d maxpool;
    FloatTensor maxpool_out;

    Layer layer1, layer2, layer3, layer4;

    Pool2d avgpool;
    FloatTensor avgpool_out;

    Linear fc;
    FloatTensor fc_out;
};

ResnetModel createResnet152()
{
    ResnetModel ret{
        .conv1 = Conv2d::loadWeightToCuda("conv1", 3, 64, 7, 2, 3),
        .bn1 = BatchNorm2d::loadWeightToCuda("bn1", 64),
        .act1_out = FloatTensor(Device::GPU),
        .maxpool = Pool2d(64, 3, 2, 1),
        .maxpool_out = FloatTensor(Device::GPU),
        .layer1 = createLayer(1, 64, 64, 256, 3),
        .layer2 = createLayer(2, 256, 128, 512, 8, 2),
        .layer3 = createLayer(3, 512, 256, 1024, 36, 2),
        .layer4 = createLayer(4, 1024, 512, 2048, 3, 2),
        .avgpool = Pool2d(2048, 7),
        .avgpool_out = FloatTensor(Device::GPU),
        .fc = Linear::loadWeightToCuda("fc", 2048, 1000),
        .fc_out = FloatTensor(Device::GPU)
    };
    return ret;
}

void layerForward(Layer &layer, FloatTensor &x)
{
    const uint64_t B = x.shape.at(0);
    FloatTensor *y = &x;
    auto convForward = [](Conv2d &conv, FloatTensor &x, FloatTensor &act_out) {
        const auto [B, C, H, W] = x.shape.as_tuple<4>();
        const Shape conv_out_shape = conv.getOutShape(x.shape);
        const auto [h, w, conv_h_out, conv_w_out] = conv_out_shape.as_tuple<4>();
        if (!act_out) {
            act_out = FloatTensor(conv_out_shape, Device::GPU);
        }

        const auto conv_block_size = dim3(8, 8, 16);
        const auto conv_blocks =
            dim3(CEIL(B * conv_h_out, conv_block_size.x), CEIL(conv_w_out, conv_block_size.y),
                 CEIL(conv.out_channels, conv_block_size.z));
        conv2dForwardKernel<<<conv_blocks, conv_block_size>>>(
            x.data, act_out.data, conv.weight.data, conv.kernel_size, conv.stride, conv.padding,
            conv_h_out, conv_w_out, B, conv.in_channels, conv.out_channels, H, W);
        cudaDeviceSynchronize();
        gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
    };
    auto bnForward = [](BatchNorm2d &bn, FloatTensor &x) {
        const auto [B, C, conv_w_out, conv_h_out] = x.shape.as_tuple<4>();
        const auto bn_block_size = dim3(8, 8, 16);
        const auto bn_blocks =
            dim3(CEIL(B, bn_block_size.x), CEIL(bn.channels_num, bn_block_size.y),
                 CEIL(conv_w_out * conv_h_out, bn_block_size.z));
        batchNorm2dForwardKernel<<<bn_blocks, bn_block_size>>>(
            x.data, x.data, bn.weight.data, bn.bias.data, bn.mean.data, bn.var.data, B,
            bn.channels_num, conv_w_out * conv_h_out);
        cudaDeviceSynchronize();
        gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
    };
    auto reluForward = [](FloatTensor &x) {
        const auto relu_block_size = dim3(1024);
        const auto relu_blocks = dim3(CEIL(x.numel(), relu_block_size.x));
        reluForwardKernel<<<relu_blocks, relu_block_size>>>(x.data, x.data, x.numel());
        cudaDeviceSynchronize();
        gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
    };
    for (auto &block : layer.blocks) {
        std::cout << "\nnew block" << std::endl;
        if (block.downsample) {
            convForward(block.downsample->conv, *y, block.downsample->act);
            bnForward(block.downsample->bn, block.downsample->act);
            std::cout << "downsample" << std::endl;
        }

        convForward(block.conv1, *y, block.act1_out);
        bnForward(block.bn1, block.act1_out);
        reluForward(block.act1_out);

        convForward(block.conv2, block.act1_out, block.act2_out);
        bnForward(block.bn2, block.act2_out);
        reluForward(block.act2_out);

        convForward(block.conv3, block.act2_out, block.act3_out);
        bnForward(block.bn3, block.act3_out);
        const auto add_block_size = dim3(1024);
        const auto add_blocks = dim3(CEIL(block.act3_out.numel(), add_block_size.x));
        // assert(block.act1_out.shape == block.downsample->act.shape);
        addForwardKernel<<<add_blocks, add_block_size>>>(
            block.act3_out.data, block.downsample ? block.downsample->act.data : y->data,
            block.act3_out.data, block.act3_out.numel());
        reluForward(block.act3_out);
        y = &block.act3_out;
        
        break;
    }
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
    FloatTensor out = x.cpu();
    out.save("cuda_out.bin");
    std::cout << "Saved output with shape = " << out.shape << std::endl;

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
    const auto relu_blocks = dim3(CEIL(model.act1_out.numel(), relu_block_size.x));
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
    std::cout << "maxpool kernel done\n";

    layerForward(model.layer1, model.maxpool_out);
    std::cout << "layer1 finished\n";
    

/* 
    layerForward(model.layer2, model.layer1.blocks.back().act3_out);
    std::cout << "layer2 finished\n";
    layerForward(model.layer3, model.layer2.blocks.back().act3_out);
    std::cout << "layer3 finished\n";
    layerForward(model.layer4, model.layer3.blocks.back().act3_out);
    std::cout << "layer4 finished\n";

   
    FloatTensor last_cpu = model.layer4.blocks.back().act3_out.toDevice(Device::CPU);
    FloatTensor &last_layer_out = model.layer4.blocks.back().act3_out;
    const Shape avgpool_out_shape = model.avgpool.getOutShape(last_layer_out.shape);
    if (!model.avgpool_out) {
        model.avgpool_out = FloatTensor(avgpool_out_shape, Device::GPU);
    }
    std::cout << "model.avgpool_out.shape = " << model.avgpool_out.shape << std::endl;
    const auto avgpool_h_out = avgpool_out_shape.at(2), avgpool_w_out = avgpool_out_shape.at(3);
    const auto avgpool_block_size = dim3(8, 8, 16);
    const auto avgpool_blocks =
        dim3(CEIL(last_layer_out.shape.at(0) * last_layer_out.shape.at(1), avgpool_block_size.x),
             CEIL(last_layer_out.shape.at(2), avgpool_block_size.y),
             CEIL(last_layer_out.shape.at(3), avgpool_block_size.z));
    avgPool2dKernel<<<avgpool_blocks, avgpool_block_size>>>(
        last_layer_out.data, model.avgpool_out.data, model.avgpool.kernel_size,
        model.avgpool.stride, model.avgpool.padding, avgpool_h_out, avgpool_w_out, B,
        last_layer_out.shape.at(1), last_layer_out.shape.at(2), last_layer_out.shape.at(3));
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
    std::cout << "avgpool kernel done\n";


    FloatTensor avgpool_out_flatten = model.avgpool_out.reshape(
        Shape({model.avgpool_out.shape.at(0), model.avgpool_out.shape.at(1) *
                                                  model.avgpool_out.shape.at(2) *
                                                  model.avgpool_out.shape.at(3)}));
    if (!model.fc_out) {
        model.fc_out = FloatTensor(Shape({avgpool_out_flatten.shape.at(0), model.fc.out_features}),
                                   Device::GPU);
    }
    const auto linear_block_size = dim3(16, 16);
    const auto linear_blocks = dim3(CEIL(avgpool_out_flatten.shape.at(0), linear_block_size.x),
                                    CEIL(avgpool_out_flatten.shape.at(1), linear_block_size.y));
    linearForwardKernel<<<linear_blocks, linear_block_size>>>(
        avgpool_out_flatten.data, model.fc_out.data, model.fc.weight.data, model.fc.bias.data,
        model.fc.in_features, avgpool_out_flatten.shape.at(0), model.fc.out_features);
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__);
    std::cout << "Finished kernel\n";
    */

    // FloatTensor out = model.avgpool_out.cpu();
    // out.save("cuda_out.bin");
    // std::cout << "Saved output with shape = " << out.shape << std::endl;
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
