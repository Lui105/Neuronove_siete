from PIL import Image
from torch.utils.data import Dataset


class Animal10Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["path"]
        target = int(row["target_id"])

        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)

        return img, target
