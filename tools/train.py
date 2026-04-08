import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from data.dataset import MeterDataset, ctc_collate_fn
from data.transforms import get_transforms
from data.label_encoder import LabelEncoder

from models.crnn import CRNN
from engine.trainer import Trainer
from engine.losses import get_ctc_loss


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


def build_scheduler(optimizer, config):
    total_epochs = config["train"]["epochs"]
    warmup_epochs = config["train"].get("warmup_epochs", 5)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)

        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)))

    return LambdaLR(optimizer, lr_lambda)


def main(config_path):
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    torch.backends.cudnn.benchmark = True

    encoder = LabelEncoder(config["charset"])
    transform = get_transforms(config["img_height"], config["img_width"])

    train_ds = MeterDataset(
        config["train"]["data_dir"],
        config["train"]["label_file"],
        transform,
        encoder
    )

    val_ds = MeterDataset(
        config["eval"]["data_dir"],
        config["eval"]["label_file"],
        transform,
        encoder
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        collate_fn=ctc_collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=config["eval"]["num_workers"],
        collate_fn=ctc_collate_fn,
        pin_memory=True
    )

    model = CRNN(config).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"]
    )

    scheduler = build_scheduler(optimizer, config)

    loss_fn = get_ctc_loss(config["blank_index"])

    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        loss_fn,
        train_loader,
        val_loader,
        config,
        device
    )

    trainer.fit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    main(args.config)