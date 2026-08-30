import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

from datasets.aptos_dataset import APTOSDataset
from datasets.transforms import get_eval_transform
from models.hybrid_cnn_vit import build_hybrid_cnn_vit
from utils.metrics import compute_metrics


TRAIN_CSV = os.environ.get("APTOS_TRAIN_CSV", "data/aptos2019_processed/train.csv")
TEST_CSV = os.environ.get("APTOS_TEST_CSV", "data/aptos2019_processed/test.csv")
CHECKPOINT_PATH = r"outputs\best_hybrid_cnn_vit_ep20.pth"

NUM_CLASSES = 5
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    batch_count = 0
    image_count = 0

    for images, labels in loader:
        batch_count += 1
        image_count += images.size(0)

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    print(f"[DEBUG] Test batches: {batch_count}, images: {image_count}")

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)

    return avg_loss, metrics


def main():
    print("Device:", DEVICE)

    test_dataset = APTOSDataset(
        csv_path=TEST_CSV,
        transform=get_eval_transform()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print("Test dataset:", len(test_dataset))
    print("Test batches:", len(test_loader))

    model = build_hybrid_cnn_vit(
        num_classes=NUM_CLASSES,
        cnn_pretrained=False
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("Loaded checkpoint:", CHECKPOINT_PATH)
    print("Best epoch:", checkpoint.get("epoch"))
    print("Best Val QWK:", checkpoint.get("best_val_qwk"))
    print("Val metrics:", checkpoint.get("val_metrics"))

    class_weights = get_class_weights(TRAIN_CSV, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE
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
