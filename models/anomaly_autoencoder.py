"""
TinyAutoencoder — a small convolutional autoencoder trained only on "boring" frames
(clear/partly_cloudy sky, no detected event). Reconstruction error at inference time
gives an out-of-distribution (OOD) score: a frame that doesn't look like anything the
model has seen before — not confidently "cloudy", not confidently any of the 4 known
event classes, but also not just routine ground/sky — will reconstruct poorly and get
flagged for human review even though neither classifier produced a specific wrong
label for it. This is the "unknown unknowns" safety net the two supervised classifiers
can't provide on their own.

Input:  128x128x3 RGB chip
Output: 128x128x3 reconstruction
"""
import torch
import torch.nn as nn

LATENT_CHANNELS = 64


class TinyAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),   # 128 -> 64
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),             # 64 -> 32
            nn.Conv2d(32, LATENT_CHANNELS, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 32 -> 16
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(LATENT_CHANNELS, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),  # 16 -> 32
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(inplace=True),                # 32 -> 64
            nn.ConvTranspose2d(16, in_channels, 4, stride=2, padding=1), nn.Sigmoid(),                # 64 -> 128
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


if __name__ == "__main__":
    m = TinyAutoencoder()
    dummy = torch.randn(2, 3, 128, 128)
    out = m(dummy)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"TinyAutoencoder output shape: {out.shape}, params: {n_params:,}")
