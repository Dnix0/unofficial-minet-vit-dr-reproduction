import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class APTOSDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        required_cols = {"image_path", "label"}
        missing_cols = required_cols - set(self.df.columns)

        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label