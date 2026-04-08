import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalizationNetwork(nn.Module):
    def __init__(self, num_fiducial, in_channels):
        super().__init__()
        self.num_fiducial = num_fiducial
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc1 = nn.Linear(256, 256)
        self.relu = nn.ReLU(True)
        self.fc2 = nn.Linear(256, num_fiducial * 2)
        self.fc2.weight.data.zero_()
        self.fc2.bias.data.zero_()

    def forward(self, x):
        b = x.size(0)
        feat = self.conv(x).view(b, -1)
        ctrl = self.fc2(self.relu(self.fc1(feat)))
        return ctrl.view(b, self.num_fiducial, 2)


class TPSSpatialTransformer(nn.Module):
    def __init__(self, num_fiducial=20, in_channels=1):
        super().__init__()
        self.num_fiducial = num_fiducial
        self.loc = LocalizationNetwork(num_fiducial, in_channels)
        target_ctrl = self._build_target_ctrl(num_fiducial)
        self.register_buffer("target_ctrl", target_ctrl)

    def _build_target_ctrl(self, num_fiducial):
        n = num_fiducial // 2
        ctrl_x = torch.linspace(-1.0, 1.0, n)
        top = torch.stack([ctrl_x, torch.full_like(ctrl_x, -1.0)], dim=1)
        bot = torch.stack([ctrl_x, torch.full_like(ctrl_x,  1.0)], dim=1)
        return torch.cat([top, bot], dim=0).unsqueeze(0)

    def _tps_grid(self, src_ctrl, target_ctrl, size):
        b = src_ctrl.size(0)
        h, w_img = int(size[2]), int(size[3])
        n = src_ctrl.size(1)
        tc = target_ctrl.expand(b, -1, -1)

        ys = torch.linspace(-1.0, 1.0, h, device=src_ctrl.device)
        xs = torch.linspace(-1.0, 1.0, w_img, device=src_ctrl.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=2).view(1, h * w_img, 2).expand(b, -1, -1)

        diff = grid.unsqueeze(2) - tc.unsqueeze(1)
        r = (diff ** 2).sum(dim=3)
        r = r * torch.log(r + 1e-6)

        r_ctrl = tc.unsqueeze(2) - tc.unsqueeze(1)
        r_ctrl = (r_ctrl ** 2).sum(dim=3)
        r_ctrl = r_ctrl * torch.log(r_ctrl + 1e-6)

        ones_n = torch.ones(b, n, 1, device=src_ctrl.device)
        P_ctrl = torch.cat([ones_n, tc], dim=2)

        L_top = torch.cat([r_ctrl, P_ctrl], dim=2)

        zeros_33 = torch.zeros(b, 3, 3, device=src_ctrl.device)
        L_bot = torch.cat([P_ctrl.transpose(1, 2), zeros_33], dim=2)

        L = torch.cat([L_top, L_bot], dim=1)

        zeros_32 = torch.zeros(b, 3, 2, device=src_ctrl.device)
        Y = torch.cat([src_ctrl, zeros_32], dim=1)

        try:
            w_coef = torch.linalg.solve(L, Y)
        except RuntimeError:
            return F.affine_grid(
                torch.eye(2, 3, device=src_ctrl.device).unsqueeze(0).expand(b, -1, -1),
                size,
                align_corners=False
            )

        w_ctrl = w_coef[:, :n, :]
        a = w_coef[:, n:, :]

        ones_hw = torch.ones(b, h * w_img, 1, device=src_ctrl.device)
        P_grid = torch.cat([ones_hw, grid], dim=2)

        out = torch.bmm(r, w_ctrl) + torch.bmm(P_grid, a)
        return out.view(b, h, w_img, 2)

    def forward(self, x):
        src_ctrl = torch.tanh(self.loc(x))
        grid = self._tps_grid(src_ctrl, self.target_ctrl, x.size())
        return F.grid_sample(x, grid, align_corners=False, padding_mode="border")
