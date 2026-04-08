import yaml
import torch
from torch.utils.data import DataLoader

from data.dataset import MeterDataset, ctc_collate_fn
from data.transforms import get_transforms
from data.label_encoder import LabelEncoder
from models.crnn import CRNN
from utils.ctc_decoder import CTCDecoder
from engine.evaluator import Evaluator


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = LabelEncoder(config["charset"])
    transform = get_transforms(config["img_height"], config["img_width"], mode="test")

    ds = MeterDataset(
        config["test"]["data_dir"],
        config["test"]["label_file"],
        transform,
        encoder
    )
    loader = DataLoader(
        ds,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=config["eval"]["num_workers"],
        collate_fn=ctc_collate_fn
    )

    model = CRNN(config)
    ckpt = torch.load("outputs/checkpoints/best.pth", map_location="cpu")
    model.load_state_dict(ckpt["model"])

    decoder = CTCDecoder(encoder, config["eval"]["beam_size"])
    evaluator = Evaluator(model, decoder, loader, device)
    acc = evaluator.evaluate()
    print(f"accuracy: {acc:.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    args = p.parse_args()
    main(args.config)
