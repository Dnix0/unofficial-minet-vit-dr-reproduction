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
from datasets.transforms_denoised import get_eval_transform_denoised
from models.hybrid_cnn_vit import build_hybrid_cnn_vit
from models.denoised_hybrid_cnn_vit import DenoisedHybridCNNViT
from utils.metrics import compute_metrics


TRAIN_CSV = os.environ.get("APTOS_TRAIN_CSV", "data/aptos2019_processed/train.csv")
TEST_CSV = os.environ.get("APTOS_TEST_CSV", "data/aptos2019_processed/test.csv")
DENOISER_CHECKPOINT = r"outputs\best_residual_denoising_unet.pth"

NUM_CLASSES = 5
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "PDR"
}

EXPERIMENTS = [
    {
        "name": "Hybrid CNN-ViT ep20",
        "type": "hybrid",
        "checkpoint": r"outputs\best_hybrid_cnn_vit_ep20.pth",
        "save_dir": r"outputs\final_hybrid_cnn_vit_ep20",
        "alpha": None
    },
    {
        "name": "Denoised Hybrid alpha=0.25 ep20",
        "type": "denoised",
        "checkpoint": r"outputs\best_denoised_hybrid_cnn_vit_alpha025_ep20.pth",
        "save_dir": r"outputs\final_denoised_hybrid_alpha025_ep20",
        "alpha": 0.25
    },
    {
        "name": "Denoised Hybrid alpha=0.5 ep20",
        "type": "denoised",
        "checkpoint": r"outputs\best_denoised_hybrid_cnn_vit_alpha05_ep20.pth",
        "save_dir": r"outputs\final_denoised_hybrid_alpha05_ep20",
        "alpha": 0.50
    }
]


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


@torch.no_grad()
def evaluate_with_predictions(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_image_ids, all_image_paths, all_labels, all_preds, all_probs = [], [], [], [], []
    batch_count = 0
    image_count = 0

    for images, labels, image_ids, image_paths in loader:
        batch_count += 1
        image_count += images.size(0)
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

    print(f"[DEBUG] Test batches: {batch_count}, images: {image_count}")
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics, all_image_ids, all_image_paths, all_labels, all_preds, all_probs


def build_model(exp):
    if exp["type"] == "hybrid":
        model = build_hybrid_cnn_vit(num_classes=NUM_CLASSES, cnn_pretrained=False)
    elif exp["type"] == "denoised":
        model = DenoisedHybridCNNViT(
            num_classes=NUM_CLASSES,
            denoiser_checkpoint=DENOISER_CHECKPOINT,
            freeze_denoiser=True,
            alpha=exp["alpha"],
            cnn_pretrained=False
        )
    else:
        raise ValueError(f"Unknown experiment type: {exp['type']}")
    return model


def save_metrics(save_dir, exp, test_loss, test_metrics, checkpoint):
    metrics_to_save = {
        "model": exp["name"],
        "type": exp["type"],
        "checkpoint_path": exp["checkpoint"],
        "alpha": exp["alpha"],
        "best_epoch": int(checkpoint.get("epoch")),
        "best_val_qwk": float(checkpoint.get("best_val_qwk")),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_weighted_f1": float(test_metrics["weighted_f1"]),
        "test_qwk": float(test_metrics["qwk"]),
        "total_epochs": checkpoint.get("total_epochs"),
        "batch_size": checkpoint.get("batch_size"),
        "lr": checkpoint.get("lr"),
        "weight_decay": checkpoint.get("weight_decay")
    }
    path = os.path.join(save_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=4, ensure_ascii=False)
    print("Saved metrics:", path)


def save_confusion_matrix(save_dir, cm):
    df = pd.DataFrame(
        cm,
        index=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        columns=[CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    )
    path = os.path.join(save_dir, "confusion_matrix.csv")
    df.to_csv(path, encoding="utf-8-sig")
    print("Saved confusion matrix:", path)


def save_predictions(save_dir, image_ids, image_paths, labels, preds, probs):
    records = []
    for image_id, image_path, label, pred, prob in zip(image_ids, image_paths, labels, preds, probs):
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
    df = pd.DataFrame(records)
    path = os.path.join(save_dir, "test_predictions.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("Saved predictions:", path)


def save_classification_report(save_dir, labels, preds):
    target_names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    report_dict = classification_report(labels, preds, target_names=target_names, digits=4, output_dict=True)
    path_json = os.path.join(save_dir, "classification_report.json")
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4, ensure_ascii=False)
    report_text = classification_report(labels, preds, target_names=target_names, digits=4)
    path_txt = os.path.join(save_dir, "classification_report.txt")
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write(report_text)
    print("Saved classification report json:", path_json)
    print("Saved classification report txt:", path_txt)


def run_one_experiment(exp, criterion):
    print("\n" + "=" * 90)
    print("Experiment:", exp["name"])
    print("Checkpoint:", exp["checkpoint"])
    print("Save dir:", exp["save_dir"])
    print("=" * 90)

    if not os.path.exists(exp["checkpoint"]):
        print("[SKIP] Missing checkpoint:", exp["checkpoint"])
        return None

    os.makedirs(exp["save_dir"], exist_ok=True)

    if exp["type"] == "hybrid":
        transform = get_eval_transform()
    else:
        transform = get_eval_transform_denoised()

    test_dataset = APTOSDatasetWithMeta(TEST_CSV, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print("Test dataset:", len(test_dataset), "| batches:", len(test_loader))

    model = build_model(exp).to(DEVICE)
    checkpoint = torch.load(exp["checkpoint"], map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Best epoch:", checkpoint.get("epoch"))
    print("Best Val QWK:", checkpoint.get("best_val_qwk"))

    test_loss, test_metrics, image_ids, image_paths, labels, preds, probs = evaluate_with_predictions(
        model, test_loader, criterion, DEVICE
    )

    print("\nTest Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_metrics['accuracy']:.4f}")
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test Weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(f"Test QWK: {test_metrics['qwk']:.4f}")
    print("Confusion Matrix:")
    print(test_metrics["confusion_matrix"])

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)], digits=4))

    save_metrics(exp["save_dir"], exp, test_loss, test_metrics, checkpoint)
    save_confusion_matrix(exp["save_dir"], test_metrics["confusion_matrix"])
    save_predictions(exp["save_dir"], image_ids, image_paths, labels, preds, probs)
    save_classification_report(exp["save_dir"], labels, preds)

    return {
        "model": exp["name"],
        "best_epoch": checkpoint.get("epoch"),
        "best_val_qwk": checkpoint.get("best_val_qwk"),
        "test_loss": test_loss,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_qwk": test_metrics["qwk"],
        "alpha": exp["alpha"]
    }


def main():
    print("Device:", DEVICE)
    class_weights = get_class_weights(TRAIN_CSV, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    rows = []
    for exp in EXPERIMENTS:
        row = run_one_experiment(exp, criterion)
        if row is not None:
            rows.append(row)

    if rows:
        comparison_dir = r"outputs\final_ep20_comparison"
        os.makedirs(comparison_dir, exist_ok=True)
        df = pd.DataFrame(rows).sort_values(
            by=["test_qwk", "test_accuracy", "test_macro_f1"],
            ascending=False
        )
        csv_path = os.path.join(comparison_dir, "final_ep20_comparison.csv")
        json_path = os.path.join(comparison_dir, "final_ep20_comparison.json")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        df.to_json(json_path, orient="records", indent=4, force_ascii=False)

        print("\nFinal EP20 Comparison")
        print(df)
        print("Saved:", csv_path)
        print("Saved:", json_path)


if __name__ == "__main__":
    main()
