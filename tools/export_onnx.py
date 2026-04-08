import yaml
import torch

from models.crnn import CRNN

def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        with open(cfg["base"], "r") as f:
            base = yaml.safe_load(f)
        base.update(cfg)
        return base
    return cfg

def main(config_path):
    config = load_config(config_path)

    model = CRNN(config)
    ckpt = torch.load(config["train"]["save_dir"] + "/best.pth", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1,1,config["img_height"],config["img_width"])

    torch.onnx.export(model, dummy, "model.onnx", input_names=["input"], output_names=["output"], opset_version=11)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    args = p.parse_args()
    main(args.config)
