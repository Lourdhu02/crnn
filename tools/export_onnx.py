import yaml
import torch
from models.crnn import CRNN


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        with open(cfg["base"], "r") as f:
            base = yaml.safe_load(f)
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d:
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        return deep_update(base, cfg)
    return cfg


def main(config_path):
    config = load_config(config_path)

    model = CRNN(config)
    ckpt = torch.load("outputs/checkpoints/best.pth", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, 1, config["img_height"], config["img_width"])

    torch.onnx.export(
        model, dummy, "outputs/onnx/model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11
    )
    print("exported to outputs/onnx/model.onnx")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    args = p.parse_args()
    main(args.config)
