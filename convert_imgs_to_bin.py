from PIL import Image
from tqdm import tqdm
import os
from pathlib import Path
import torchvision
import struct

input_dir = "test_imgs"
out_dir = "test_bins"

resnet = torchvision.models.resnet152(weights="IMAGENET1K_V1").cuda()
preprocess = torchvision.models.ResNet152_Weights.IMAGENET1K_V1.transforms()
with os.scandir(input_dir) as it:
    for entry in tqdm(it):
        if entry.name.endswith(".jpeg") and entry.is_file():
            print(entry.name, entry.path)
            with open(entry.path, "rb") as f:
                img = Image.open(f)
                print(img)
                tensor = preprocess(img).unsqueeze(0)
            with open((Path(out_dir) / entry.name).with_suffix(".bin"), "wb") as f:
                for x in tensor.data.flatten():
                    f.write(struct.pack('f', x.item()))