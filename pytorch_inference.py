import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

DEVICE = "cuda"


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


a = torch.rand((16, 3, 224, 224)).to(DEVICE)
resnet = Resnet152(1000).to(DEVICE)
resnet2 = torchvision.models.resnet152(weights="IMAGENET1K_V1").to(DEVICE)
resnet.load_state_dict(resnet2.state_dict())
c = resnet2(a)
b = resnet(a)
print(torch.equal(b, c))
