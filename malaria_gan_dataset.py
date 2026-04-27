import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

try:
    RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE = Image.BILINEAR


def load_rgb(path: Path):
    path = Path(path)
    if path.suffix.lower() not in VALID_EXTENSIONS:
        return None
    try:
        with Image.open(path) as img:
            return img.convert("RGB").copy()
    except Exception:
        return None


class MalariaGanDataset(Dataset):
    def __init__(self, image_paths, target_size, horizontal_flip=True, labels=None):
        self.image_paths = [Path(p) for p in image_paths]
        self.target_size = (target_size, target_size)
        self.horizontal_flip = horizontal_flip
        self.labels = None if labels is None else [int(label) for label in labels]
        if self.labels is not None and len(self.labels) != len(self.image_paths):
            raise ValueError("labels must have the same length as image_paths")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = load_rgb(path)
        if image is None:
            raise RuntimeError(f"Invalid image during training: {path}")

        image = image.resize(self.target_size, RESAMPLE)
        if self.horizontal_flip and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        if self.labels is None:
            return tensor
        return tensor, torch.tensor(self.labels[index], dtype=torch.long)
