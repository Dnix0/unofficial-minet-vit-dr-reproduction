import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.denoise_dataset import DenoiseDataset
from models.denoising_unet import ResidualDenoisingUNet


TRAIN_CSV = os.environ.get("APTOS_TRAIN_CSV", "data/aptos2019_processed/train.csv")
VAL_CSV = os.environ.get("APTOS_VAL_CSV", "data/aptos2019_processed/val.csv")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-5
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr(pred, target, eps=1e-8):
    mse = torch.mean((pred - target) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse + eps))


def train_one_epoch(model, loader, criterion_l1, criterion_mse, optimizer, device):
    model.train()

    total_loss = 0.0
    total_psnr_noisy = 0.0
    total_psnr_denoised = 0.0

    for noisy, clean in tqdm(loader, desc="Train", leave=False):
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()

        denoised = model(noisy)

        loss_l1 = criterion_l1(denoised, clean)
        loss_mse = criterion_mse(denoised, clean)
        loss = loss_l1 + 0.2 * loss_mse

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * noisy.size(0)
        total_psnr_noisy += psnr(noisy.detach(), clean.detach()).item() * noisy.size(0)
        total_psnr_denoised += psnr(denoised.detach(), clean.detach()).item() * noisy.size(0)

    n = len(loader.dataset)

    return {
        "loss": total_loss / n,
        "psnr_noisy": total_psnr_noisy / n,
        "psnr_denoised": total_psnr_denoised / n
    }


@torch.no_grad()
def evaluate(model, loader, criterion_l1, criterion_mse, device):
    model.eval()

    total_loss = 0.0
    total_psnr_noisy = 0.0
    total_psnr_denoised = 0.0

    for noisy, clean in tqdm(loader, desc="Val", leave=False):
        noisy = noisy.to(device)
        clean = clean.to(device)

        denoised = model(noisy)

        loss_l1 = criterion_l1(denoised, clean)
        loss_mse = criterion_mse(denoised, clean)
        loss = loss_l1 + 0.2 * loss_mse

        total_loss += loss.item() * noisy.size(0)
        total_psnr_noisy += psnr(noisy, clean).item() * noisy.size(0)
        total_psnr_denoised += psnr(denoised, clean).item() * noisy.size(0)

    n = len(loader.dataset)

    return {
        "loss": total_loss / n,
        "psnr_noisy": total_psnr_noisy / n,
        "psnr_denoised": total_psnr_denoised / n
    }


def main():
    set_seed(SEED)

    print("Device:", DEVICE)

    train_dataset = DenoiseDataset(TRAIN_CSV)
    val_dataset = DenoiseDataset(VAL_CSV)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    model = ResidualDenoisingUNet(
        in_channels=3,
        base_channels=32
    ).to(DEVICE)

    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    best_path = os.path.join(OUTPUT_DIR, "best_residual_denoising_unet.pth")

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{EPOCHS}]")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion_l1,
            criterion_mse,
            optimizer,
            DEVICE
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion_l1,
            criterion_mse,
            DEVICE
        )

        print(
            f"Train Loss: {train_metrics['loss']:.5f} | "
            f"Noisy PSNR: {train_metrics['psnr_noisy']:.2f} | "
            f"Denoised PSNR: {train_metrics['psnr_denoised']:.2f}"
        )

        print(
            f"Val Loss: {val_metrics['loss']:.5f} | "
            f"Noisy PSNR: {val_metrics['psnr_noisy']:.2f} | "
            f"Denoised PSNR: {val_metrics['psnr_denoised']:.2f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_metrics": val_metrics
                },
                best_path
            )

            print(f"Best denoiser saved: {best_path}")

    print("\nDenoiser training finished.")
    print("Best Val Loss:", best_val_loss)


if __name__ == "__main__":
    main()
