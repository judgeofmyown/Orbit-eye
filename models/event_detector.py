"""
TinyEventNet — small depthwise-separable CNN for onboard disaster-event classification.

Same backbone shape as TinyCloudNet on purpose: in a later optimization pass the two
could share a trunk and branch into two heads (cloud head + event head) to halve
onboard inference cost. Kept separate here for clarity and independent training.

Input:  128x128x3 RGB (or single-channel SAR amplitude, see note below)
Output: logits over EVENT_CLASSES

NOTE on SAR vs optical: oil-spill detection typically uses Sentinel-1 SAR imagery,
which is single/dual-channel, not 3-channel RGB. Two options:
  1. Replicate the SAR channel to 3 channels so it fits this same architecture
     (simplest, what `in_channels=3` assumes by default).
  2. Train a second small model with `in_channels=1` for SAR-only inputs and route
     to it based on which sensor captured the frame.
This file defaults to option 1 for a single unified model on Day 1; swap
`in_channels` if you go with option 2.
"""
import torch
import torch.nn as nn

from cloud_classifier import DepthwiseSeparableConv  # reuse the same building block

EVENT_CLASSES = ["none", "oil_spill", "other_anomaly", "wildfire"]  # alphabetical: matches ImageFolder's folder ordering


class TinyEventNet(nn.Module):
    def __init__(self, num_classes: int = len(EVENT_CLASSES), in_channels: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )

        self.blocks = nn.Sequential(
            DepthwiseSeparableConv(16, 32, stride=2),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
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
    m = TinyEventNet()
    dummy = torch.randn(2, 3, 128, 128)
    out = m(dummy)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"TinyEventNet output shape: {out.shape}, params: {n_params:,}")
