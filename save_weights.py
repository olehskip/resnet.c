import torch
import torchvision
import struct
from pathlib import Path
from tqdm import tqdm

resnet = torchvision.models.resnet152(weights="IMAGENET1K_V1")
dir_name = "./weights_bin"
for name, tensor in tqdm(resnet.state_dict().items()):
    with open(Path(dir_name) / name, "wb") as f:
        for x in tensor.data.flatten():
            f.write(struct.pack('f', x.item()))