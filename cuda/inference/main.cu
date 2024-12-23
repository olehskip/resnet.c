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
            block.downsample->conv.forward(*y, block.downsample->act);
            block.downsample->bn.forward(block.downsample->act, block.downsample->act);
            std::cout << "downsample" << std::endl;
        }

        block.conv1.forward(*y, block.act1_out);
        block.bn1.forward(block.act1_out, block.act1_out);
        reluForward(block.act1_out);

        block.conv1.forward(block.act1_out, block.act2_out);
        block.bn2.forward(block.act1_out, block.act2_out);
        reluForward(block.act2_out);

        block.conv3.forward(block.act2_out, block.act3_out);
        block.bn3.forward(block.act3_out, block.act3_out);
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
    model.conv1.forward(x, model.act1_out);
    std::cout << "conv1 kernel done\n";
    
    model.bn1.forward(model.act1_out, model.act1_out);
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
    model.maxpool.maxforward(model.act1_out, model.maxpool_out);
    std::cout << "maxpool kernel done\n";

    layerForward(model.layer1, model.maxpool_out);
    std::cout << "layer1 finished\n";
     
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
    model.avgpool.avgforward(last_layer_out, model.avgpool_out);
    std::cout << "avgpool kernel done\n";


    FloatTensor avgpool_out_flatten = model.avgpool_out.reshape(
        Shape({model.avgpool_out.shape.at(0), model.avgpool_out.shape.at(1) *
                                                  model.avgpool_out.shape.at(2) *
                                                  model.avgpool_out.shape.at(3)}));
    if (!model.fc_out) {
        model.fc_out = FloatTensor(Shape({avgpool_out_flatten.shape.at(0), model.fc.out_features}),
                                   Device::GPU);
    }
    model.fc.forward(avgpool_out_flatten, model.fc_out);
    std::cout << "Finished\n";

    FloatTensor out = model.fc_out.cpu();
    out.save("cuda_out.bin");
    std::cout << "Saved output with shape = " << out.shape << std::endl;
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
