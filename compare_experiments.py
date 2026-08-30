import os
import json
import pandas as pd


EXPERIMENTS = {
    "EfficientNet-B0": r"outputs\baseline_efficientnet_b0\metrics.json",
    "Hybrid CNN-ViT": r"outputs\hybrid_cnn_vit\metrics.json",
    "Hybrid + Strong VST/Retinex": r"outputs\hybrid_cnn_vit_vst_retinex\metrics.json",
    "Hybrid + Mild VST/Retinex": r"outputs\hybrid_cnn_vit_vst_retinex_mild\metrics.json",
    "Denoised Hybrid alpha=0.1": r"outputs\denoised_hybrid_alpha01\metrics.json",
    "Denoised Hybrid alpha=0.25": r"outputs\denoised_hybrid_alpha025\metrics.json",
    "Denoised Hybrid alpha=0.5": r"outputs\denoised_hybrid_alpha05\metrics.json"
}

SAVE_DIR = r"outputs\comparison"
os.makedirs(SAVE_DIR, exist_ok=True)


def load_metrics(name, path):
    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    row = {
        "model": name,
        "best_epoch": metrics.get("best_epoch"),
        "best_val_qwk": metrics.get("best_val_qwk"),
        "test_loss": metrics.get("test_loss"),
        "test_accuracy": metrics.get("test_accuracy"),
        "test_macro_f1": metrics.get("test_macro_f1"),
        "test_weighted_f1": metrics.get("test_weighted_f1"),
        "test_qwk": metrics.get("test_qwk")
    }

    if "alpha" in metrics:
        row["alpha"] = metrics.get("alpha")
    else:
        row["alpha"] = None

    return row


def dataframe_to_markdown(df):
    columns = list(df.columns)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator] + rows)


def main():
    rows = []

    for name, path in EXPERIMENTS.items():
        if not os.path.exists(path):
            print(f"Missing metrics file: {path}")
            continue

        rows.append(load_metrics(name, path))

    if len(rows) == 0:
        print("No experiment metrics found.")
        return

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by=["test_qwk", "test_accuracy", "test_macro_f1"],
        ascending=False
    )

    csv_path = os.path.join(SAVE_DIR, "experiment_comparison.csv")
    json_path = os.path.join(SAVE_DIR, "experiment_comparison.json")
    md_path = os.path.join(SAVE_DIR, "experiment_comparison.md")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_json(json_path, orient="records", indent=4, force_ascii=False)

    markdown_text = "# Experiment Comparison\n\n"
    markdown_text += dataframe_to_markdown(df)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    print("\nExperiment Comparison")
    print(df)

    print("\nSaved files:")
    print(csv_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
