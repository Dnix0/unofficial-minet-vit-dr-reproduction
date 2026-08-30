import torch
from torchvision import transforms


class AddGaussianNoise:
    def __init__(self, mean=0.0, std_range=(0.01, 0.04), p=0.2):
        self.mean = mean
        self.std_range = std_range
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() > self.p:
            return tensor

        std = torch.empty(1).uniform_(
            self.std_range[0],
            self.std_range[1]
        ).item()

        noise = torch.randn_like(tensor) * std + self.mean
        tensor = tensor + noise
        return torch.clamp(tensor, 0.0, 1.0)


class AddPoissonNoise:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() > self.p:
            return tensor

        vals = 256.0
        noisy = torch.poisson(tensor * vals) / vals
        return torch.clamp(noisy, 0.0, 1.0)


def get_train_transform_denoised():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            scale=(0.85, 1.15)
        ),
        transforms.ColorJitter(
            brightness=(0.7, 1.4),
            contrast=(0.8, 1.2)
        ),
        transforms.ToTensor(),
        AddGaussianNoise(std_range=(0.01, 0.04), p=0.2),
        AddPoissonNoise(p=0.2)
    ])


def get_eval_transform_denoised():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
