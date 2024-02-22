from openslide.deepzoom import DeepZoomGenerator
from torch.utils.data import Dataset
from tqdm import tqdm

from pathlib import Path
import os

OPENSLIDE_PATH = r"D:\DataManage\openslide-win64-20231011\bin"
if hasattr(os, "add_dll_directory"):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
import xml.etree.ElementTree as ET

import numpy as np
import openslide
import torch
from openslide.deepzoom import DeepZoomGenerator
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
import torch
from torchvision.models.resnet import Bottleneck, ResNet


class TilesDataset(Dataset):
    def __init__(self, slide: openslide.OpenSlide, tiles_coords: np.ndarray) -> None:
        self.slide = slide
        self.tiles_coords = tiles_coords
        self.transform = Compose(
            [
                # ToTensor(),
                Normalize(
                    mean=(0.70322989, 0.53606487, 0.66096631),
                    std=(0.21716536, 0.26081574, 0.20723464),
                ),
            ]
        )  # Specific normalization for BT
        self.dz = DeepZoomGenerator(slide, tile_size=224, overlap=0)
        file_extension = Path(self.slide._filename).suffix
        if file_extension == ".svs":
            self.magnification = int(self.slide.properties["openslide.objective-power"])
        elif file_extension == ".qptiff":
            r = (
                ET.fromstring(slide.properties["openslide.comment"])
                .find("ScanProfile")
                .find("root")
                .find("ScanResolution")
            )
            self.magnification = float(r.find("Magnification").text)
        elif file_extension == ".ndpi":
            self.magnification = int(self.slide.properties["openslide.objective-power"])
        else:
            raise ValueError(f"File extension {file_extension} not supported")
        # We want the second highest level so as to have 112 microns tiles / 0.5 microns per pixel
        if self.magnification == 20:
            self.level = self.dz.level_count - 1
        elif self.magnification == 40:
            self.level = self.dz.level_count - 2
            self.magnification = 20
        else:
            raise ValueError(f"Objective power {self.magnification}x not supported")
        self.z = self.level

        assert np.all(
            self.tiles_coords[:, 0] == self.z
        ), "The resolution of the tiles is not the same as the resolution of the slide."

    def __getitem__(self, item: int):
        tile_coords = self.tiles_coords[item, 2:4].astype(int)
        try:
            im = self.dz.get_tile(level=self.level, address=tile_coords)
        except ValueError:
            print(f"ValueError: impossible to open tile {tile_coords} from {self.slide}")
            raise ValueError
        im = ToTensor()(im)
        if im.shape != torch.Size([3, 224, 224]):
            print(f"Image shape is {im.shape} for tile {tile_coords}. Padding...")
            # PAD the image in white to reach 224x224
            im = torch.nn.functional.pad(im, (0, 224 - im.shape[2], 0, 224 - im.shape[1]), value=1)

        im = self.transform(im)
        return im

    def __len__(self) -> int:
        return len(self.tiles_coords)


class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.mean(dim=[2, 3])  # globalpool

        return x


def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def resnet50_special(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(torch.hub.load_state_dict_from_url(pretrained_url, progress=progress))
        print(verbose)
    return model


def get_features(
    slide: openslide.OpenSlide,
    model: torch.nn.Module,
    tiles_coords: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 0,
    prefetch_factor: int = 8,
) -> np.ndarray:
    dataset = TilesDataset(slide=slide, tiles_coords=tiles_coords)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,  # num_workers=0 is necessary when using windows
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )

    features = []
    with torch.inference_mode():
        for images in tqdm(dataloader, total=len(dataloader)):
            features_b = model(images.to(device))
            features_b = features_b.cpu().numpy()
            features.append(features_b)
    features = np.concatenate(features)
    features = np.concatenate([tiles_coords[:, [0, 2, 3]], features], axis=1)
    # features is of shape (n_tiles, 2051) where the first columns is the resolution, the 2nd and 3rd are the coordinates of the tile, and the 4th to 2051 are the features

    return features


def extract_features(
    slide_path: Path,
    device: torch.device,
    batch_size: int = 32,
    tiles_coords_path: Path = None,
    tiles_coords: np.ndarray = None,
    num_workers: int = 0,
    prefetch_factor: int = None,
):
    assert (
        tiles_coords_path is not None or tiles_coords is not None
    ), "Either tiles_coords_path or tiles_coords must be provided"

    slide = openslide.OpenSlide(str(slide_path))
    if tiles_coords is None:
        tiles_coords = np.load(tiles_coords_path)
    model = resnet50_special(pretrained=True, progress=True, key="BT").to(device)
    model.eval()
    features = get_features(
        slide=slide,
        model=model,
        tiles_coords=tiles_coords,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    return features
