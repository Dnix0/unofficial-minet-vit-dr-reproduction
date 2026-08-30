import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inputs, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.handles.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self.handles.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations or gradients. "
                "Check target_layer."
            )

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().detach().cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, logits.detach(), class_idx


def load_rgb_image(image_path, size=(224, 224)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(size)
    return image


def pil_to_numpy01(image):
    return np.asarray(image).astype(np.float32) / 255.0


def overlay_cam_on_image(rgb01, cam, alpha=0.45, cmap_name="jet"):
    cmap = plt.get_cmap(cmap_name)
    heatmap = cmap(cam)[:, :, :3]

    overlay = (1.0 - alpha) * rgb01 + alpha * heatmap
    overlay = np.clip(overlay, 0.0, 1.0)

    return heatmap, overlay


def save_gradcam_figure(
    original_rgb01,
    cam,
    heatmap,
    overlay,
    save_path,
    title="Grad-CAM"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    axes[0].imshow(original_rgb01)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap)
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def save_comparison_figure(
    original_rgb01,
    hybrid_overlay,
    denoised_overlay,
    save_path,
    title="Grad-CAM Comparison"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))

    axes[0].imshow(original_rgb01)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(hybrid_overlay)
    axes[1].set_title("Hybrid CNN-ViT")
    axes[1].axis("off")

    axes[2].imshow(denoised_overlay)
    axes[2].set_title("Denoised Hybrid α=0.5")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
