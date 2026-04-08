import torch
import torch.nn as nn
from .tps_stn import TPSSpatialTransformer
from .backbone import get_backbone
from .sequence import BiLSTM
from .head import CTCHead

class CRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_tps = config["model"]["tps"]["enable"]
        in_ch = 1
        if self.use_tps:
            self.tps = TPSSpatialTransformer(config["model"]["tps"]["num_fiducial"], in_ch)
        self.backbone = get_backbone(
            config["model"]["backbone"]["name"],
            config["model"]["backbone"]["pretrained"],
            config["model"]["backbone"]["out_channels"]
        )
        hidden = config["model"]["sequence"]["hidden_size"]
        self.sequence = BiLSTM(config["model"]["backbone"]["out_channels"], hidden, config["model"]["sequence"]["num_layers"], config["model"]["sequence"]["dropout"])
        self.head = CTCHead(hidden*2, config["model"]["head"]["num_classes"])

    def forward(self, x):
        if self.use_tps:
            x = self.tps(x)
        feat = self.backbone(x)
        b,c,h,w = feat.size()
        feat = feat.mean(2).permute(0,2,1)
        seq = self.sequence(feat)
        out = self.head(seq)
        return out.log_softmax(2)
