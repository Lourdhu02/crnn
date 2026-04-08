import torch
import torch.nn as nn
from .tps_stn import TPSSpatialTransformer
from .backbone import get_backbone
from .sequence import BiLSTM
from .head import CTCHead


class CRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone_name = config["model"]["backbone"]["name"]
        self.use_tps = config["model"]["tps"]["enable"]
        in_ch = config.get("img_channel", 3)

        if self.use_tps:
            self.tps = TPSSpatialTransformer(
                config["model"]["tps"]["num_fiducial"],
                in_ch
            )

        self.backbone = get_backbone(
            self.backbone_name,
            config["model"]["backbone"]["pretrained"],
            config["model"]["backbone"]["out_channels"]
        )

        self.use_rnn = self.backbone_name != "svtr_tiny"
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

        hidden = config["model"]["sequence"]["hidden_size"]
        out_ch = config["model"]["backbone"]["out_channels"]

        if self.use_rnn:
            self.sequence = BiLSTM(
                out_ch,
                hidden,
                config["model"]["sequence"]["num_layers"],
                config["model"]["sequence"]["dropout"]
            )
            self.head = CTCHead(hidden * 2, config["model"]["head"]["num_classes"])
        else:
            self.head = CTCHead(out_ch, config["model"]["head"]["num_classes"])

    def forward(self, x):
        if self.use_tps:
            x = self.tps(x)

        feat = self.backbone(x)
        feat = self.adaptive_pool(feat)
        feat = feat.squeeze(2).permute(0, 2, 1)

        if self.use_rnn:
            feat = self.sequence(feat)

        out = self.head(feat)
        return out.log_softmax(2)

    def freeze_tps(self):
        if self.use_tps:
            for p in self.tps.parameters():
                p.requires_grad = False

    def unfreeze_tps(self):
        if self.use_tps:
            for p in self.tps.parameters():
                p.requires_grad = True
