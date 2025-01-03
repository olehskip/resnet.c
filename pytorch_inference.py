import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import struct
from pathlib import Path

def check_out(y, name="cuda_out.bin"):
    my_y = load_tensor(name).view(y.shape)
    print(f"y Equal with {name}: {torch.allclose(y, my_y, atol=1e-2, rtol=0.1)}")
    breakpoint()

DEVICE = "cuda"

def load_tensor(file_path):
    file_path = Path(file_path)
    file_size = file_path.stat().st_size
    num_floats = file_size // 4
    
    with open(file_path, "rb") as f:
        data = []
        for _ in range(num_floats):
            float_data = struct.unpack('f', f.read(4))[0]
            data.append(float_data)
    
    tensor = torch.tensor(data).to(DEVICE)
    
    return tensor
class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        inter_channels,
        out_channels,
        stride=1,
        downsample: nn.Module = nn.Identity(),
    ):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(inter_channels)

        self.conv2 = nn.Conv2d(
            in_channels=inter_channels,
            out_channels=inter_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(inter_channels)

        self.conv3 = nn.Conv2d(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        shortcut = self.downsample(x)

        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y)

        y = self.conv3(y)
        y = self.bn3(y)
        y += shortcut
        y = F.relu(y)

        return y


def make_layer(in_channels, inter_channels, out_channels, n_blocks, stride=1):
    downsample = (
        nn.Sequential(
            *[
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            ]
        )
        if stride != 1 or in_channels != out_channels
        else nn.Identity()
    )
    return nn.Sequential(
        *(
            [ResnetBlock(in_channels, inter_channels, out_channels, stride, downsample)]
            + [
                ResnetBlock(out_channels, inter_channels, out_channels)
                for _ in range(n_blocks - 1)
            ]
        )
    )


class Resnet152(nn.Module):
    def __init__(self, n_classes=64):
        super(Resnet152, self).__init__()

        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = make_layer(64, 64, 256, n_blocks=3)
        self.layer2 = make_layer(256, 128, 512, n_blocks=8, stride=2)
        self.layer3 = make_layer(512, 256, 1024, n_blocks=36, stride=2)
        self.layer4 = make_layer(1024, 512, 2048, n_blocks=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, n_classes)

    def forward(self, x):
        y = self.conv1(x)
        assert y.shape[-2:] == (112,) * 2
        y = self.bn1(y)
        y = F.relu(y)

        y = self.maxpool(y)
        assert y.shape[-2:] == (56,) * 2
        
        y = self.layer1(y)
        assert y.shape[-2:] == (56,) * 2

        y = self.layer2(y)
        assert y.shape[-2:] == (28,) * 2

        y = self.layer3(y)
        assert y.shape[-2:] == (14,) * 2

        y = self.layer4(y)
        assert y.shape[-2:] == (7,) * 2

        y = self.avgpool(y)
        assert y.shape[-2:] == (1,) * 2

        y = self.fc(y.flatten(-3))
        assert y.shape[-1:] == (self.n_classes,)

        return y

C = 3
H, W = 224, 224
x = load_tensor("test_bins/ILSVRC2012_val_00004749.bin").view(1, C, H, W)
resnet = Resnet152(1000).to(DEVICE)
resnet2 = torchvision.models.resnet152(weights="IMAGENET1K_V1").to(DEVICE)
resnet.load_state_dict(resnet2.state_dict())
resnet.eval()
y = resnet(x)
print(y.argmax(1))