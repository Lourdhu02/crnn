import torch.nn as nn
from .svtr import SVTRTiny


def get_backbone(name="mobilenet_v3_small", pretrained=True, out_channels=512):
    if name == "mobilenet_v3_small":
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = mobilenet_v3_small(weights=weights)
        return nn.Sequential(
            m.features,
            nn.Conv2d(576, out_channels, 1)
        )

    if name == "mobilenet_v3_large":
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        m = mobilenet_v3_large(weights=weights)
        return nn.Sequential(
            m.features,
            nn.Conv2d(960, out_channels, 1)
        )

    if name == "resnet34":
        from torchvision.models import resnet34, ResNet34_Weights
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        m = resnet34(weights=weights)
        layers = list(m.children())[:-2]
        if out_channels != 512:
            return nn.Sequential(*layers, nn.Conv2d(512, out_channels, 1))
        return nn.Sequential(*layers)

    if name == "svtr_tiny":
        return SVTRTiny(in_ch=3, out_channels=out_channels)

    raise ValueError(f"Unsupported backbone: {name}")
