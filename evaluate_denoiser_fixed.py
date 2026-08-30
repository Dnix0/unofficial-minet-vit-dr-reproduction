import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pandas as pd

from models.denoising_unet import ResidualDenoisingUNet


TEST_CSV = os.environ.get("APTOS_TEST_CSV", "data/aptos2019_processed/test.csv")
CHECKPOINT_PATH = r"outputs\best_residual_denoising_unet.pth"

SAVE_DIR = r"outputs\denoiser_eval_fixed"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16


class FixedNoiseDenoiseDataset(Dataset):
    def __init__(self, csv_path, gaussian_std=0.04, poisson=True):
        self.df = pd.read_csv(csv_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.gaussian_std = gaussian_std
        self.poisson = poisson

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        clean = self.transform(image)

        # deterministic-ish per sample
        torch.manual_seed(idx)

        noisy = clean.clone()

        noise = torch.randn_like(noisy) * self.gaussian_std
        noisy = torch.clamp(noisy + noise, 0.0, 1.0)

        if self.poisson:
            vals = 256.0
            noisy = torch.poisson(noisy * vals) / vals
            noisy = torch.clamp(noisy, 0.0, 1.0)

        return noisy, clean, row["image_id"]


def psnr_batch(pred, target, eps=1e-8):
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + eps))
    return psnr


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    noisy_psnr_all = []
    denoised_psnr_all = []
    identity_psnr_all = []

    for noisy, clean, image_ids in loader:
        noisy = noisy.to(DEVICE)
        clean = clean.to(DEVICE)

        denoised = model(noisy)

        # clean image를 denoiser에 넣었을 때 얼마나 원본을 보존하는지
        clean_pass = model(clean)

        noisy_psnr = psnr_batch(noisy, clean)
        denoised_psnr = psnr_batch(denoised, clean)
        identity_psnr = psnr_batch(clean_pass, clean)

        noisy_psnr_all.extend(noisy_psnr.cpu().numpy().tolist())
        denoised_psnr_all.extend(denoised_psnr.cpu().numpy().tolist())
        identity_psnr_all.extend(identity_psnr.cpu().numpy().tolist())

    results = {
        "noisy_psnr": float(np.mean(noisy_psnr_all)),
        "denoised_psnr": float(np.mean(denoised_psnr_all)),
        "improvement": float(np.mean(denoised_psnr_all) - np.mean(noisy_psnr_all)),
        "identity_psnr": float(np.mean(identity_psnr_all))
    }

    return results


@torch.no_grad()
def save_examples(model, loader, save_path, num_images=6):
    model.eval()

    noisy, clean, image_ids = next(iter(loader))

    noisy = noisy[:num_images].to(DEVICE)
    clean = clean[:num_images].to(DEVICE)

    denoised = model(noisy)
    clean_pass = model(clean)

    noisy = noisy.cpu()
    clean = clean.cpu()
    denoised = denoised.cpu()
    clean_pass = clean_pass.cpu()

    fig, axes = plt.subplots(4, num_images, figsize=(num_images * 2.2, 8))

    row_titles = ["Clean", "Noisy", "Denoised", "Clean→Denoiser"]

    for r, imgs in enumerate([clean, noisy, denoised, clean_pass]):
        for c in range(num_images):
            img = imgs[c].permute(1, 2, 0).numpy()
            img = np.clip(img, 0.0, 1.0)

            axes[r, c].imshow(img)
            axes[r, c].axis("off")

            if c == 0:
                axes[r, c].set_ylabel(row_titles[r], fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

    print("Saved:", save_path)


def main():
    print("Device:", DEVICE)

    dataset = FixedNoiseDenoiseDataset(
        TEST_CSV,
        gaussian_std=0.04,
        poisson=True
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    model = ResidualDenoisingUNet(
        in_channels=3,
        base_channels=32
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("Loaded checkpoint:", CHECKPOINT_PATH)
    print("Best epoch:", checkpoint.get("epoch"))
    print("Best val loss:", checkpoint.get("best_val_loss"))

    results = evaluate(model, loader)

    print("\nFixed Noise Evaluation")
    print(f"Noisy PSNR: {results['noisy_psnr']:.4f}")
    print(f"Denoised PSNR: {results['denoised_psnr']:.4f}")
    print(f"Improvement: {results['improvement']:.4f}")
    print(f"Identity PSNR clean→denoiser: {results['identity_psnr']:.4f}")

    save_examples(
        model,
        loader,
        os.path.join(SAVE_DIR, "fixed_noise_examples.png"),
        num_images=6
    )


if __name__ == "__main__":
    main()
