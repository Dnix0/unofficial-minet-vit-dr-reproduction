import torch
import torch.nn as nn

from models.denoising_unet import ResidualDenoisingUNet
from models.hybrid_cnn_vit import build_hybrid_cnn_vit


class DenoisedHybridCNNViT(nn.Module):
    def __init__(
        self,
        num_classes=5,
        denoiser_checkpoint=None,
        freeze_denoiser=True,
        alpha=0.5,
        cnn_pretrained=True
    ):
        super().__init__()

        self.alpha = alpha

        self.denoiser = ResidualDenoisingUNet(
            in_channels=3,
            base_channels=32
        )

        if denoiser_checkpoint is not None:
            checkpoint = torch.load(
                denoiser_checkpoint,
                map_location="cpu",
                weights_only=False
            )
            self.denoiser.load_state_dict(checkpoint["model_state_dict"])

        if freeze_denoiser:
            for p in self.denoiser.parameters():
                p.requires_grad = False
            self.denoiser.eval()

        self.freeze_denoiser = freeze_denoiser

        self.classifier = build_hybrid_cnn_vit(
            num_classes=num_classes,
            cnn_pretrained=cnn_pretrained
        )

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, x):
        # x: [0,1] RGB tensor

        if self.freeze_denoiser:
            with torch.no_grad():
                denoised = self.denoiser(x)
        else:
            denoised = self.denoiser(x)

        blended = (1.0 - self.alpha) * x + self.alpha * denoised
        blended = torch.clamp(blended, 0.0, 1.0)

        normalized = self.normalize(blended)

        logits = self.classifier(normalized)

        return logits
