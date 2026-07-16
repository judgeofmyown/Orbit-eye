"""
Minimal Grad-CAM for the TinyCloudNet/TinyEventNet/OrbitFusionNet depthwise-separable
CNN backbones. Hooks the last conv block in the network and produces a class-activation
heatmap overlaid on the original frame — so a human looking at a Priority/Review frame
in the ground station can see *why* the model flagged it, not just the label.

No matplotlib/opencv dependency — the heatmap colormap and blending are done with
plain numpy/PIL, which are already project dependencies.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, logits: torch.Tensor, class_idx: int) -> np.ndarray:
        """Call after a forward pass that produced `logits` (with grad enabled).
        Returns a 0..1 heatmap sized to the model's input resolution."""
        self.model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(128, 128), mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        if cam.max() > 1e-8:
            cam /= cam.max()
        return cam


def overlay_heatmap(orig_image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blends a 0..1 heatmap onto the original image with a simple blue->red ramp."""
    h, w = cam.shape
    orig = orig_image.convert("RGB").resize((w, h))
    orig_arr = np.asarray(orig, dtype=np.float32)

    heat = np.zeros((h, w, 3), dtype=np.float32)
    heat[..., 0] = np.clip(cam * 3, 0, 1) * 255       # red ramps up first (hot spots)
    heat[..., 1] = np.clip(cam * 3 - 1, 0, 1) * 255   # green kicks in mid-range
    heat[..., 2] = np.clip(1 - cam * 2, 0, 1) * 255   # blue dominates cool/background areas

    blended = np.clip(orig_arr * (1 - alpha) + heat * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)
