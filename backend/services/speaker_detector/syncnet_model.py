"""
SyncNet network architecture — moved verbatim into its own module.

Matches the official syncnet_v2 checkpoint layer-for-layer (layer indices in
the Sequentials must not change, or load_state_dict fails).

Reference: "Out of Time: Automated Lip Sync in the Wild"
           (Chung & Zisserman, 2016) — github.com/joonson/syncnet_python
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SyncNetModel(nn.Module):
    """
    SyncNet architecture matching the official syncnet_v2 checkpoint.

    Visual input:  (B, 3, 5, H, W) — 5 frames of RGB, H×W must yield 6×6 after pooling.
                   Resize face crops to 240×240 before feeding in.
    Audio input:   (B, 1, 13, 20) — 13 MFCC features × 20 audio frames (0.2 s at 100fps).

    Reference: "Out of Time: Automated Lip Sync in the Wild" (Chung & Zisserman, 2016)
               https://github.com/joonson/syncnet_python
    """

    def __init__(self):
        super().__init__()

        # Visual encoder — layer indices must match checkpoint keys exactly
        self.netcnnlip = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=0),   # 0
            nn.BatchNorm3d(96),                                                       # 1
            nn.ReLU(inplace=True),                                                    # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),                   # 3

            nn.Conv3d(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 1, 1)),  # 4
            nn.BatchNorm3d(256),                                                      # 5
            nn.ReLU(inplace=True),                                                    # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),                   # 7

            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),           # 8
            nn.BatchNorm3d(256),                                                      # 9
            nn.ReLU(inplace=True),                                                    # 10

            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),           # 11
            nn.BatchNorm3d(256),                                                      # 12
            nn.ReLU(inplace=True),                                                    # 13

            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),           # 14
            nn.BatchNorm3d(256),                                                      # 15
            nn.ReLU(inplace=True),                                                    # 16
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),                   # 17

            nn.Conv3d(256, 512, kernel_size=(1, 6, 6), padding=0),                   # 18
            nn.BatchNorm3d(512),                                                      # 19
        )

        self.netfclip = nn.Sequential(
            nn.Linear(512, 512),       # 0 — flattened from (B, 512, 1, 1, 1)
            nn.BatchNorm1d(512),       # 1
            nn.ReLU(inplace=True),     # 2
            nn.Linear(512, 1024),      # 3
        )

        # Audio encoder — layer indices must match checkpoint keys exactly
        # Input: (B, 1, 13, 20) → output: (B, 512, 1, 1)
        self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),    # 0
            nn.BatchNorm2d(64),                                                      # 1
            nn.ReLU(inplace=True),                                                   # 2
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),                        # 3

            nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 4
            nn.BatchNorm2d(192),                                                     # 5
            nn.ReLU(inplace=True),                                                   # 6
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),                        # 7

            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),                # 8
            nn.BatchNorm2d(384),                                                     # 9
            nn.ReLU(inplace=True),                                                   # 10

            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),                # 11
            nn.BatchNorm2d(256),                                                     # 12
            nn.ReLU(inplace=True),                                                   # 13

            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),                # 14
            nn.BatchNorm2d(256),                                                     # 15
            nn.ReLU(inplace=True),                                                   # 16
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),                        # 17

            nn.Conv2d(256, 512, kernel_size=(5, 4), padding=0),                     # 18
            nn.BatchNorm2d(512),                                                     # 19
        )

        self.netfcaud = nn.Sequential(
            nn.Linear(512, 512),       # 0
            nn.BatchNorm1d(512),       # 1
            nn.ReLU(inplace=True),     # 2
            nn.Linear(512, 1024),      # 3
        )

    def forward_visual(self, x: torch.Tensor) -> torch.Tensor:
        """Encode visual input.

        Args:
            x: (B, 3, T, H, W) — RGB face crops, T=5, H=W=240
        """
        x = self.netcnnlip(x)         # → (B, 512, 1, 1, 1)
        x = x.view(x.size(0), -1)     # → (B, 512)
        x = self.netfclip(x)          # → (B, 1024)
        return F.normalize(x, p=2, dim=1)

    def forward_audio(self, x: torch.Tensor) -> torch.Tensor:
        """Encode audio input.

        Args:
            x: (B, 1, 13, 20) — MFCC features, 13 coeffs × 20 frames
        """
        x = self.netcnnaud(x)         # → (B, 512, 1, 1)
        x = x.view(x.size(0), -1)     # → (B, 512)
        x = self.netfcaud(x)          # → (B, 1024)
        return F.normalize(x, p=2, dim=1)

    def forward(self, visual: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between visual and audio embeddings."""
        return F.cosine_similarity(self.forward_visual(visual), self.forward_audio(audio), dim=1)
