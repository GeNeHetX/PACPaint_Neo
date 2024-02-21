import numpy as np
from tqdm import tqdm
from openslide.deepzoom import DeepZoomGenerator
from torch.utils.data import Dataset
from skimage.morphology import disk, binary_closing
from scipy.ndimage import binary_fill_holes
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


class TilesWhiteDataset(Dataset):
    def __init__(
        self,
        slide: openslide.OpenSlide,
        tile_size: int = 224,
    ) -> None:
        self.slide = slide
        file_extension = Path(self.slide._filename).suffix
        if file_extension == ".svs":
            self.z = int(self.slide.properties["openslide.objective-power"])
        elif file_extension == ".qptiff":
            r = (
                ET.fromstring(slide.properties["openslide.comment"])
                .find("ScanProfile")
                .find("root")
                .find("ScanResolution")
            )
            self.z = int(r.find("Magnification").text)
        elif file_extension == ".ndpi":
            self.z = int(self.slide.properties["openslide.objective-power"])
        else:
            raise ValueError(f"File extension {file_extension} not supported")
        self.dz = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0)
        # We want the second highest level so as to have 112 microns tiles / 0.5 microns per pixel
        if self.z == 20:
            self.level = self.dz.level_count - 1
        elif self.z == 40:
            self.level = self.dz.level_count - 2
        else:
            raise ValueError(f"Objective power {self.z}x not supported")
        self.h, self.w = self.dz.level_dimensions[self.level]
        self.h_tile, self.w_tile = self.dz.level_tiles[self.level]
        # Get rid of the last row and column because they can't fit a full tile usually
        self.h_tile -= 1
        self.w_tile -= 1

    def idx_to_ij(self, item: int):
        return np.unravel_index(item, (self.h_tile, self.w_tile))

    def __len__(self) -> int:
        return self.h_tile * self.w_tile


def filter_whites(path_svs, folder_path):
    tile_size = 224
    slide = openslide.OpenSlide(path_svs)
    slide_dt = TilesWhiteDataset(slide, tile_size=tile_size)

    # First segment the slide
    img_pil = slide.get_thumbnail((1000, 1000))
    img = np.array(img_pil)
    tresh = 220
    mask = img.mean(axis=2) < tresh
    closed = binary_closing(mask, disk(3))
    filled = binary_fill_holes(closed, structure=np.ones((15, 15)))
    final_mask = binary_closing(filled, disk(3))

    # Get various dimensions
    h_thumb, w_thumb = img.shape[:2]
    w_slide, h_slide = slide.dimensions
    z = slide_dt.z
    if z == 20:
        w_slide, h_slide = w_slide, h_slide
    elif z == 40:
        w_slide, h_slide = w_slide // 2, h_slide // 2
        z = z // 2
    w_ratio = w_thumb / w_slide
    h_ratio = h_thumb / h_slide
    w_ratio, h_ratio

    all_tiles_coord = [[z, k, i, j] for k, i, j in zip(range(len(slide_dt)), *slide_dt.idx_to_ij(range(len(slide_dt))))]
    all_tiles_coord = np.array(all_tiles_coord)
    # Get the coordinates of the tiles in the thumbnail
    coord_thumb = np.zeros((all_tiles_coord.shape[0], 4, 2), dtype=np.int32)
    for k in tqdm(range(all_tiles_coord.shape[0])):
        # Convert tile adress to pixel coordinates in the full resolution image
        i, j = all_tiles_coord[k, 2] * tile_size, all_tiles_coord[k, 3] * tile_size
        corners = np.array([[i, j], [i + tile_size, j], [i, j + tile_size], [i + tile_size, j + tile_size]]).astype(
            np.float32
        )
        # Map the coordinates of the pixels i,j to the coordinates on the thumbnail
        corners[:, 0] *= h_ratio
        corners[:, 1] *= w_ratio
        coord_thumb[k] = np.round(corners).astype(np.int32)

    coord_thumb[:, :, 0] = np.clip(coord_thumb[:, :, 0], 0, w_thumb - 1)
    coord_thumb[:, :, 1] = np.clip(coord_thumb[:, :, 1], 0, h_thumb - 1)

    # We keep the tile if at least one of its corner is inside the mask
    valid_tiles_idx = final_mask[coord_thumb[:, :, 1], coord_thumb[:, :, 0]].sum(axis=1) > 0
    tiles_coord = all_tiles_coord[valid_tiles_idx]

    slide_name = Path(path_svs).stem
    export_path = folder_path / f"{slide_name}" / "tiles_coord.npy"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(export_path, tiles_coord)

    print("Finished process for slide", Path(path_svs).stem)
    print("Exported tiles coordinates to", export_path)

    return tiles_coord
