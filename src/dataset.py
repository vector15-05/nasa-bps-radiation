import tifffile
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import config

LABEL_MAP = {"X-ray": 0, "Fe": 1}
INV_LABEL_MAP = {0: "X-ray", 1: "Fe"}

def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class NASACellDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row[config.COL_FILENAME]

        # Read 16-bit grayscale TIFF and min-max normalize to [0, 1]
        img = tifffile.imread(img_path).astype("float32")
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Expand (H, W) -> (3, H, W)
        tensor_img = torch.tensor(img).unsqueeze(0).repeat(3, 1, 1)

        if self.transform:
            tensor_img = self.transform(tensor_img)

        label = LABEL_MAP[row[config.COL_PARTICLE]]
        return tensor_img, torch.tensor(label, dtype=torch.long)