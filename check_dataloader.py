import os
import torch
from torch.utils.data import DataLoader

from datasets.aptos_dataset import APTOSDataset
from datasets.transforms import get_train_transform, get_eval_transform


TRAIN_CSV = os.environ.get("APTOS_TRAIN_CSV", "data/aptos2019_processed/train.csv")
VAL_CSV = os.environ.get("APTOS_VAL_CSV", "data/aptos2019_processed/val.csv")
TEST_CSV = os.environ.get("APTOS_TEST_CSV", "data/aptos2019_processed/test.csv")


def main():
    train_dataset = APTOSDataset(
        csv_path=TRAIN_CSV,
        transform=get_train_transform()
    )

    val_dataset = APTOSDataset(
        csv_path=VAL_CSV,
        transform=get_eval_transform()
    )

    test_dataset = APTOSDataset(
        csv_path=TEST_CSV,
        transform=get_eval_transform()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print("Train dataset:", len(train_dataset))
    print("Val dataset:", len(val_dataset))
    print("Test dataset:", len(test_dataset))

    images, labels = next(iter(train_loader))

    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
    print("Labels:", labels)

    assert images.shape[1:] == torch.Size([3, 224, 224])
    assert labels.ndim == 1

    print("DataLoader check passed.")


if __name__ == "__main__":
    main()