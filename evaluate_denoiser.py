import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from datasets.denoise_dataset import DenoiseDataset
from models.denoising_unet import ResidualDenoisingUNet


VAL_CSV = os.environ.get("APTOS_VAL_CSV", "data/aptos2019_processed/val.csv")
TEST_CSV = os.environ.get("APTOS_TEST_CSV", "data/aptos2019_processed/test.csv")

CHECKPOINT_PATH = r"outputs\best_residual_denoising_unet.pth"

SAVE_DIR = r"outputs\denoiser_eval"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def psnr_batch(pred, target, eps=1e-8):
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + eps))
    return psnr


@torch.no_grad()
def evaluate(model, loader, device, split_name):
    model.eval()

    all_noisy_psnr = []
    all_denoised_psnr = []

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        denoised = model(noisy)

        noisy_psnr = psnr_batch(noisy, clean)
        denoised_psnr = psnr_batch(denoised, clean)

        all_noisy_psnr.extend(noisy_psnr.detach().cpu().numpy().tolist())
        all_denoised_psnr.extend(denoised_psnr.detach().cpu().numpy().tolist())

    noisy_mean = float(np.mean(all_noisy_psnr))
    denoised_mean = float(np.mean(all_denoised_psnr))
    improvement = denoised_mean - noisy_mean

    print(f"\n[{split_name}]")
    print(f"Noisy PSNR: {noisy_mean:.4f}")
    print(f"Denoised PSNR: {denoised_mean:.4f}")
    print(f"Improvement: {improvement:.4f}")

    return noisy_mean, denoised_mean, improvement


@torch.no_grad()
def save_visual_examples(model, loader, device, save_path, num_images=6):
    model.eval()

    noisy, clean = next(iter(loader))

    noisy = noisy[:num_images].to(device)
    clean = clean[:num_images].to(device)

    denoised = model(noisy)

    # CPU로 이동
    clean = clean.detach().cpu()
    noisy = noisy.detach().cpu()
    denoised = denoised.detach().cpu()

    # [clean rows, noisy rows, denoised rows]
    combined = torch.cat([clean, noisy, denoised], dim=0)

    grid = make_grid(
        combined,
        nrow=num_images,
        padding=2
    )

    np_img = grid.permute(1, 2, 0).numpy()
    np_img = np.clip(np_img, 0.0, 1.0)

    plt.figure(figsize=(num_images * 2.2, 7))
    plt.imshow(np_img)
    plt.axis("off")
    plt.title("Top: Clean | Middle: Noisy | Bottom: Denoised")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"Saved visual examples: {save_path}")


def main():
    print("Device:", DEVICE)

    val_dataset = DenoiseDataset(VAL_CSV)
    test_dataset = DenoiseDataset(TEST_CSV)

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
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
    print("Val metrics:", checkpoint.get("val_metrics"))

    evaluate(model, val_loader, DEVICE, "Val")
    evaluate(model, test_loader, DEVICE, "Test")

    save_visual_examples(
        model,
        test_loader,
        DEVICE,
        save_path=os.path.join(SAVE_DIR, "denoiser_examples.png"),
        num_images=6
    )


if __name__ == "__main__":
    main()
