import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.aptos_dataset import APTOSDataset
from datasets.transforms_denoised import get_train_transform_denoised, get_eval_transform_denoised
from models.denoised_hybrid_cnn_vit import DenoisedHybridCNNViT
from utils.metrics import compute_metrics


TRAIN_CSV = os.environ.get("APTOS_TRAIN_CSV", "data/aptos2019_processed/train.csv")
VAL_CSV = os.environ.get("APTOS_VAL_CSV", "data/aptos2019_processed/val.csv")
TEST_CSV = os.environ.get("APTOS_TEST_CSV", "data/aptos2019_processed/test.csv")

DENOISER_CHECKPOINT = r"outputs\best_residual_denoising_unet.pth"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CLASSES = 5
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42

ALPHA = 0.10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_class_weights(train_csv, num_classes=5):
    df = pd.read_csv(train_csv)
    counts = df["label"].value_counts().sort_index()

    total = len(df)
    weights = []

    for cls in range(num_classes):
        cls_count = counts.get(cls, 0)
        weight = total / (num_classes * cls_count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    if hasattr(model, "denoiser") and model.freeze_denoiser:
        model.denoiser.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)

    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)

    return avg_loss, metrics


def main():
    set_seed(SEED)

    print("Device:", DEVICE)
    print("Denoiser checkpoint:", DENOISER_CHECKPOINT)
    print("Denoiser alpha:", ALPHA)

    train_dataset = APTOSDataset(
        csv_path=TRAIN_CSV,
        transform=get_train_transform_denoised()
    )

    val_dataset = APTOSDataset(
        csv_path=VAL_CSV,
        transform=get_eval_transform_denoised()
    )

    test_dataset = APTOSDataset(
        csv_path=TEST_CSV,
        transform=get_eval_transform_denoised()
    )

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    model = DenoisedHybridCNNViT(
        num_classes=NUM_CLASSES,
        denoiser_checkpoint=DENOISER_CHECKPOINT,
        freeze_denoiser=True,
        alpha=ALPHA,
        cnn_pretrained=True
    ).to(DEVICE)

    class_weights = get_class_weights(TRAIN_CSV, NUM_CLASSES).to(DEVICE)
    print("Class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_val_qwk = -1.0
    best_model_path = os.path.join(
        OUTPUT_DIR,
        "best_denoised_hybrid_cnn_vit_alpha01.pth"
    )

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{EPOCHS}]")

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )

        val_loss, val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE,
            desc="Val"
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Macro F1: {train_metrics['macro_f1']:.4f} | "
            f"QWK: {train_metrics['qwk']:.4f}"
        )

        print(f"Val Loss: {val_loss:.4f}")
        print(
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Macro F1: {val_metrics['macro_f1']:.4f} | "
            f"QWK: {val_metrics['qwk']:.4f}"
        )

        if val_metrics["qwk"] > best_val_qwk:
            best_val_qwk = val_metrics["qwk"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_qwk": best_val_qwk,
                    "val_metrics": val_metrics,
                    "alpha": ALPHA,
                    "denoiser_checkpoint": DENOISER_CHECKPOINT
                },
                best_model_path
            )

            print(f"Best model saved: {best_model_path}")

    print("\nTraining finished.")
    print("Best Val QWK:", best_val_qwk)

    print("\nLoading best model for test evaluation...")

    checkpoint = torch.load(
        best_model_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE,
        desc="Test"
    )

    print("\nTest Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_metrics['accuracy']:.4f}")
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test Weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(f"Test QWK: {test_metrics['qwk']:.4f}")
    print("Confusion Matrix:")
    print(test_metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
