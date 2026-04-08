import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalizationNetwork(nn.Module):
    def __init__(self, num_fiducial, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,1,1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, num_fiducial*2)

    def forward(self, x):
        b = x.size(0)
        feat = self.conv(x).view(b, -1)
        ctrl = self.fc(feat)
        return ctrl.view(b, -1, 2)

class TPSSpatialTransformer(nn.Module):
    def __init__(self, num_fiducial=20, in_channels=1):
        super().__init__()
        self.loc = LocalizationNetwork(num_fiducial, in_channels)

    def forward(self, x):
        ctrl = self.loc(x)
        grid = F.affine_grid(torch.eye(2,3).unsqueeze(0).repeat(x.size(0),1,1).to(x.device), x.size(), align_corners=False)
        out = F.grid_sample(x, grid, align_corners=False)
        return out
