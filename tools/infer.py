import yaml
import torch
import cv2
import os

from data.transforms import get_transforms
from data.label_encoder import LabelEncoder
from models.crnn import CRNN
from utils.ctc_decoder import CTCDecoder
from utils.postprocess import postprocess, apply_confidence


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


def load_image(path, transform):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = transform(image=img)["image"]
    img = img[0].unsqueeze(0).unsqueeze(0)
    return img

def main(config_path, image_path):
    config = load_config(config_path)

    encoder = LabelEncoder(config["charset"])
    transform = get_transforms(config["img_height"], config["img_width"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = CRNN(config).to(device)

    ckpt_path = os.path.join("outputs", "checkpoints", "best.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    decoder = CTCDecoder(encoder, config["eval"]["beam_size"])

    save_dir = "outputs/predictions"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "preds.txt")

    threshold = 0.6

    with open(save_path, "w") as f:
        if os.path.isdir(image_path):
            for name in os.listdir(image_path):
                p = os.path.join(image_path, name)

                img = load_image(p, transform).to(device)

                with torch.no_grad():
                    pred = model(img)

                pred = pred.squeeze(0).cpu().numpy()

                text, conf = decoder.decode(pred, return_conf=True)
                text = postprocess(text)
                text = apply_confidence(text, conf, threshold)

                f.write(f"{name} {text} {conf:.3f}\n")
                print(name, text, round(conf, 3))

        else:
            img = load_image(image_path, transform).to(device)

            with torch.no_grad():
                pred = model(img)

            pred = pred.squeeze(0).cpu().numpy()

            text, conf = decoder.decode(pred, return_conf=True)
            text = postprocess(text)
            text = apply_confidence(text, conf, threshold)

            f.write(f"{image_path} {text} {conf:.3f}\n")
            print(text, round(conf, 3))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--image_path", required=True)
    args = parser.parse_args()

    main(args.config, args.image_path)