import yaml
import torch
import cv2
import os

from data.transforms import get_transforms
from data.label_encoder import LabelEncoder
from models.crnn import CRNN
from utils.ctc_decoder import CTCDecoder
from utils.postprocess import clean

def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        with open(cfg["base"], "r") as f:
            base = yaml.safe_load(f)
        base.update(cfg)
        return base
    return cfg

def load_image(path, transform):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = transform(image=img)["image"]
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float() / 255.0
    return img

def main(config_path, image_path):
    config = load_config(config_path)

    encoder = LabelEncoder(config["charset"])
    transform = get_transforms(config["img_height"], config["img_width"])

    model = CRNN(config)
    ckpt = torch.load(config["train"]["save_dir"] + "/best.pth", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    decoder = CTCDecoder(encoder, config["eval"]["beam_size"])

    if os.path.isdir(image_path):
        for f in os.listdir(image_path):
            p = os.path.join(image_path, f)
            img = load_image(p, transform)
            pred = model(img)
            pred = pred.argmax(2).squeeze(0).tolist()
            text = decoder.decode(pred)
            print(f, clean(text))
    else:
        img = load_image(image_path, transform)
        pred = model(img)
        pred = pred.argmax(2).squeeze(0).tolist()
        text = decoder.decode(pred)
        print(clean(text))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--image_path", required=True)
    args = p.parse_args()
    main(args.config, args.image_path)
