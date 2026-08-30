import os
import json
import torch
import torch.nn as nn
import pandas as pd

from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from datasets.aptos_dataset import APTOSDataset
from datasets.transforms import get_eval_transform
from models.hybrid_cnn_vit import build_hybrid_cnn_vit
from utils.metrics import compute_metrics


# =========================
# Config
# =========================

TRAIN_CSV = os.environ.get("APTOS_TRAIN_CSV", "data/aptos2019_processed/train.csv")
TEST_CSV = os.environ.get("APTOS_TEST_CSV", "data/aptos2019_processed/test.csv")

CHECKPOINT_PATH = r"outputs\best_hybrid_cnn_vit_vst_retinex.pth"

SAVE_DIR = r"outputs\hybrid_cnn_vit_vst_retinex"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_CLASSES = 5
BATCH_SIZE = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "PDR"
}


# =========================
# Class weights
# =========================

def get_class_weights(train_csv, num_classes=5):
    df = pd.read_csv(train_csv)
    counts = df["label"].value_counts().sort_index()

    total = len(df)
    weights = []

    for cls in range(num_classes):
        cls_count = counts.get(cls, 0)
        weight = total / (num_classes * cls_count)
        weights.append(weight)

    weights = torch.tensor(weights, dtype=torch.float32)
    return weights


# =========================
# Dataset with metadata
# =========================

class APTOSDatasetWithMeta(APTOSDataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        label = int(row["label"])
        image_id = row["image_id"]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, image_id, image_path


# =========================
# Evaluation
# =========================

@torch.no_grad()
def evaluate_with_predictions(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0

    all_image_ids = []
    all_image_paths = []
    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels, image_ids, image_paths in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        total_loss += loss.item() * images.size(0)

        all_image_ids.extend(list(image_ids))
        all_image_paths.extend(list(image_paths))
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)

    return avg_loss, metrics, all_image_ids, all_image_paths, all_labels, all_preds, all_probs


# =========================
# Save functions
# =========================

def save_metrics(save_dir, test_loss, test_metrics, checkpoint):
    metrics_to_save = {
        "model": "Hybrid CNN-ViT + Strong VST/Retinex",
        "checkpoint_path": CHECKPOINT_PATH,
        "best_epoch": int(checkpoint.get("epoch")),
        "best_val_qwk": float(checkpoint.get("best_val_qwk")),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_weighted_f1": float(test_metrics["weighted_f1"]),
        "test_qwk": float(test_metrics["qwk"])
    }

    save_path = os.path.join(save_dir, "metrics.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=4, ensure_ascii=False)

    print(f"Saved metrics: {save_path}")


def save_confusion_matrix(save_dir, cm):
    cm_df = pd.DataFrame(
        cm,
        index=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        columns=[CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    )

    save_path = os.path.join(save_dir, "confusion_matrix.csv")
    cm_df.to_csv(save_path, encoding="utf-8-sig")

    print(f"Saved confusion matrix: {save_path}")


def save_predictions(save_dir, image_ids, image_paths, labels, preds, probs):
    records = []

    for image_id, image_path, label, pred, prob in zip(
        image_ids, image_paths, labels, preds, probs
    ):
        record = {
            "image_id": image_id,
            "image_path": image_path,
            "true_label": int(label),
            "true_class": CLASS_NAMES[int(label)],
            "pred_label": int(pred),
            "pred_class": CLASS_NAMES[int(pred)],
            "correct": int(label == pred)
        }

        for cls_idx in range(NUM_CLASSES):
            record[f"prob_{cls_idx}_{CLASS_NAMES[cls_idx]}"] = float(prob[cls_idx])

        records.append(record)

    pred_df = pd.DataFrame(records)

    save_path = os.path.join(save_dir, "test_predictions.csv")
    pred_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"Saved predictions: {save_path}")


def save_classification_report(save_dir, labels, preds):
    target_names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]

    report_dict = classification_report(
        labels,
        preds,
        target_names=target_names,
        digits=4,
        output_dict=True
    )

    save_path_json = os.path.join(save_dir, "classification_report.json")

    with open(save_path_json, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4, ensure_ascii=False)

    report_text = classification_report(
        labels,
        preds,
        target_names=target_names,
        digits=4
    )

    save_path_txt = os.path.join(save_dir, "classification_report.txt")

    with open(save_path_txt, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Saved classification report json: {save_path_json}")
    print(f"Saved classification report txt: {save_path_txt}")


# =========================
# Main
# =========================

def main():
    print("Device:", DEVICE)

    test_dataset = APTOSDatasetWithMeta(
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

    model = build_hybrid_cnn_vit(
        num_classes=NUM_CLASSES,
        cnn_pretrained=False
    )
    model = model.to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("Loaded checkpoint:", CHECKPOINT_PATH)
    print("Best epoch:", checkpoint.get("epoch"))
    print("Best Val QWK:", checkpoint.get("best_val_qwk"))

    class_weights = get_class_weights(TRAIN_CSV, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    (
        test_loss,
        test_metrics,
        image_ids,
        image_paths,
        labels,
        preds,
        probs
    ) = evaluate_with_predictions(
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

    print("\nConfusion Matrix:")
    print(test_metrics["confusion_matrix"])

    print("\nClassification Report:")
    print(
        classification_report(
            labels,
            preds,
            target_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
            digits=4
        )
    )

    save_metrics(SAVE_DIR, test_loss, test_metrics, checkpoint)
    save_confusion_matrix(SAVE_DIR, test_metrics["confusion_matrix"])
    save_predictions(SAVE_DIR, image_ids, image_paths, labels, preds, probs)
    save_classification_report(SAVE_DIR, labels, preds)

    print("\nAll Hybrid CNN-ViT + Strong VST/Retinex results saved successfully.")


if __name__ == "__main__":
    main()
