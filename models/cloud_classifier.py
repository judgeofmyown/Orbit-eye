"""
TinyCloudNet — a small depthwise-separable CNN for onboard cloud-cover classification.

Designed to be edge-friendly:
- Depthwise-separable convolutions (MobileNet-style) instead of full convs -> far
  fewer FLOPs/params, which matters when you're running on a Jetson Nano or similar
  on battery/solar power.
- No batch-norm-heavy design decisions that complicate INT8/TensorRT quantization
  later — kept deliberately simple.

Input:  128x128x3 RGB chip
Output: logits over CLOUD_CLASSES

This file defines architecture only. No trained weights are included — see
train_cloud_classifier.py and models/weights/README.md.
"""
import torch
import torch.nn as nn

CLOUD_CLASSES = ["clear", "overcast", "partly_cloudy"]  # alphabetical: matches ImageFolder's folder ordering


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                                    padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class TinyCloudNet(nn.Module):
    def __init__(self, num_classes: int = len(CLOUD_CLASSES), in_channels: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )  # 128 -> 64

        self.blocks = nn.Sequential(
            DepthwiseSeparableConv(16, 32, stride=2),   # 64 -> 32
            DepthwiseSeparableConv(32, 64, stride=2),   # 32 -> 16
            DepthwiseSeparableConv(64, 128, stride=2),  # 16 -> 8
            DepthwiseSeparableConv(128, 128, stride=1), # 8 -> 8
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


if __name__ == "__main__":
    # quick shape sanity check
    m = TinyCloudNet()
    dummy = torch.randn(2, 3, 128, 128)
    out = m(dummy)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"TinyCloudNet output shape: {out.shape}, params: {n_params:,}")
