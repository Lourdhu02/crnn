import torch

def save(path, model, optimizer=None, epoch=0):
    data = {"model": model.state_dict(), "epoch": epoch}
    if optimizer:
        data["optimizer"] = optimizer.state_dict()
    torch.save(data, path)

def load(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0)
