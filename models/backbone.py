import torch.nn as nn
import torchvision.models as models

def get_backbone(name="mobilenet_v3_small", pretrained=True, out_channels=512):
    if name == "mobilenet_v3_small":
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = mobilenet_v3_small(weights=weights)
        m.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        features = m.features
        return nn.Sequential(features, nn.Conv2d(576, out_channels, 1))

    if name == "resnet34":
        from torchvision.models import resnet34, ResNet34_Weights
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        m = resnet34(weights=weights)
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        layers = list(m.children())[:-2]
        return nn.Sequential(*layers, nn.Conv2d(512, out_channels, 1))

    raise ValueError(name)