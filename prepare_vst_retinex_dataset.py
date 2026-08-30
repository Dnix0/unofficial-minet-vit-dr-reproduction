import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

from preprocessing.vst_retinex import apply_vst_retinex


# =========================
# Paths
# =========================

SRC_DIR = os.environ.get("APTOS_PROCESSED_DIR", "data/aptos2019_processed")
OUT_DIR = os.environ.get("APTOS_VST_RETINEX_DIR", "data/aptos2019_vst_retinex_processed")
OUT_IMG_DIR = os.path.join(OUT_DIR, "images_224_vst_retinex")

os.makedirs(OUT_IMG_DIR, exist_ok=True)


SPLIT_FILES = [
    "manifest.csv",
    "train.csv",
    "val.csv",
    "test.csv"
]


def process_unique_images(manifest_df):
    unique_df = manifest_df.drop_duplicates(subset=["image_id"]).reset_index(drop=True)

    print(f"Unique images to process: {len(unique_df)}")

    processed_paths = {}

    for _, row in tqdm(unique_df.iterrows(), total=len(unique_df), desc="VST-Retinex"):
        image_id = row["image_id"]
        src_path = row["image_path"]

        out_path = os.path.join(OUT_IMG_DIR, f"{image_id}.png")

        if os.path.exists(out_path):
            processed_paths[image_id] = out_path
            continue

        image = Image.open(src_path).convert("RGB")

        processed = apply_vst_retinex(
            image,
            vst=True,
            retinex=True,
            gamma=True,
            green_guided=True,
            retinex_radius=15,
            gamma_value=0.9
        )

        processed.save(out_path)
        processed_paths[image_id] = out_path

    return processed_paths


def rewrite_csv(split_name, processed_paths):
    src_csv = os.path.join(SRC_DIR, split_name)
    out_csv = os.path.join(OUT_DIR, split_name)

    df = pd.read_csv(src_csv)

    df["image_path"] = df["image_id"].map(processed_paths)
    df["source"] = "aptos2019_vst_retinex"

    missing = df["image_path"].isna().sum()
    if missing > 0:
        raise ValueError(f"{split_name}: missing processed paths = {missing}")

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"Saved: {out_csv} | rows: {len(df)}")


def main():
    manifest_path = os.path.join(SRC_DIR, "manifest.csv")
    manifest_df = pd.read_csv(manifest_path)

    processed_paths = process_unique_images(manifest_df)

    for split_name in SPLIT_FILES:
        rewrite_csv(split_name, processed_paths)

    print("\nDone.")
    print(f"Processed dataset saved to: {OUT_DIR}")
    print(f"Processed images saved to: {OUT_IMG_DIR}")


if __name__ == "__main__":
    main()
