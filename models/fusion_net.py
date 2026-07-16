"""
OrbitFusionNet — shared-trunk multi-task model that replaces running TinyCloudNet and
TinyEventNet as two separate forward passes with one shared backbone + two heads.

The expensive part of a depthwise-separable CNN is the stem + early blocks (that's
where most FLOPs live); running that trunk once and branching into a small
task-specific block per head roughly halves onboard inference compute compared to
two independent networks — this was flagged as a future optimization in the original
cloud_classifier.py / event_detector.py docstrings, implemented here.

Input:  128x128x3 RGB chip
Output: (cloud_logits, event_logits) — two heads, one forward pass.
"""
import torch
import torch.nn as nn

from cloud_classifier import DepthwiseSeparableConv, CLOUD_CLASSES
from event_detector import EVENT_CLASSES


class OrbitFusionNet(nn.Module):
    def __init__(self, num_cloud_classes: int = len(CLOUD_CLASSES),
                 num_event_classes: int = len(EVENT_CLASSES), in_channels: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )  # 128 -> 64

        # shared trunk: identical shape to TinyCloudNet/TinyEventNet's first two
        # blocks, run once per frame instead of twice
        self.trunk = nn.Sequential(
            DepthwiseSeparableConv(16, 32, stride=2),   # 64 -> 32
            DepthwiseSeparableConv(32, 64, stride=2),   # 32 -> 16
            DepthwiseSeparableConv(64, 128, stride=2),  # 16 -> 8
        )

        # small task-specific block per head so cloud/event features can still
        # specialize after branching off the shared trunk
        self.cloud_head_block = DepthwiseSeparableConv(128, 128, stride=1)
        self.event_head_block = DepthwiseSeparableConv(128, 128, stride=1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cloud_fc = nn.Linear(128, num_cloud_classes)
        self.event_fc = nn.Linear(128, num_event_classes)

    def forward(self, x):
        x = self.stem(x)
        trunk_feat = self.trunk(x)

        cloud_feat = self.pool(self.cloud_head_block(trunk_feat)).flatten(1)
        event_feat = self.pool(self.event_head_block(trunk_feat)).flatten(1)

        return self.cloud_fc(cloud_feat), self.event_fc(event_feat)

    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        cloud_logits, event_logits = self.forward(x)
        return torch.softmax(cloud_logits, dim=1), torch.softmax(event_logits, dim=1)


if __name__ == "__main__":
    m = OrbitFusionNet()
    dummy = torch.randn(2, 3, 128, 128)
    cloud_out, event_out = m(dummy)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"OrbitFusionNet cloud_out: {cloud_out.shape}, event_out: {event_out.shape}, params: {n_params:,}")

    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from cloud_classifier import TinyCloudNet
    from event_detector import TinyEventNet
    separate_params = (sum(p.numel() for p in TinyCloudNet().parameters()) +
                        sum(p.numel() for p in TinyEventNet().parameters()))
    print(f"vs. two separate networks: {separate_params:,} params "
          f"({separate_params / n_params:.2f}x more)")
