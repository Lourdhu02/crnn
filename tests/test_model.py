import torch
from models.crnn import CRNN

def test_forward():
    config = {
        "model": {
            "tps": {"enable": True, "num_fiducial": 20},
            "backbone": {"name": "mobilenet_v3_small", "pretrained": False, "out_channels": 512},
            "sequence": {"hidden_size": 256, "num_layers": 2, "dropout": 0.1},
            "head": {"num_classes": 12}
        }
    }
    model = CRNN(config)
    x = torch.randn(2,1,32,128)
    y = model(x)
    assert y.shape[0] == 2
    assert y.shape[2] == 12
