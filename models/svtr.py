import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=1, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        return self.norm(self.proj(x))


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MixingBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        tokens = tokens + self.attn(self.norm1(tokens), self.norm1(tokens), self.norm1(tokens))[0]
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()


class SVTRTiny(nn.Module):
    def __init__(self, in_ch=1, out_channels=192):
        super().__init__()
        self.stage1 = nn.Sequential(
            PatchEmbed(in_ch, 64),
            MixingBlock(64),
            MixingBlock(64)
        )
        self.down1 = nn.Conv2d(64, 128, 3, 2, 1)
        self.stage2 = nn.Sequential(
            MixingBlock(128),
            MixingBlock(128)
        )
        self.down2 = nn.Conv2d(128, out_channels, 3, 2, 1)
        self.stage3 = nn.Sequential(
            MixingBlock(out_channels),
            MixingBlock(out_channels)
        )
        self.pos_enc = PositionalEncoding(out_channels)

    def forward(self, x):
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)

        B, C, H, W = x.shape
        # pool height first, then apply positional encoding along width
        x = x.mean(2)              # (B, C, W)
        tokens = x.permute(0, 2, 1)  # (B, W, C)
        tokens = self.pos_enc(tokens)
        return tokens.permute(0, 2, 1).unsqueeze(2)  # (B, C, 1, W)
