import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AddSyntheticNoise:
    def __init__(
        self,
        gaussian_p=0.5,
        poisson_p=0.5,
        gaussian_std_range=(0.01, 0.06)
    ):
        self.gaussian_p = gaussian_p
        self.poisson_p = poisson_p
        self.gaussian_std_range = gaussian_std_range

    def add_gaussian(self, x):
        std = torch.empty(1).uniform_(
            self.gaussian_std_range[0],
            self.gaussian_std_range[1]
        ).item()

        noise = torch.randn_like(x) * std
        return torch.clamp(x + noise, 0.0, 1.0)

    def add_poisson(self, x):
        vals = 256.0
        noisy = torch.poisson(x * vals) / vals
        return torch.clamp(noisy, 0.0, 1.0)

    def __call__(self, x):
        noisy = x.clone()

        if torch.rand(1).item() < self.gaussian_p:
            noisy = self.add_gaussian(noisy)

        if torch.rand(1).item() < self.poisson_p:
            noisy = self.add_poisson(noisy)

        return noisy


class DenoiseDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.to_tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.noise = AddSyntheticNoise()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        clean = self.to_tensor(image)

        noisy = self.noise(clean)

        return noisy, clean
