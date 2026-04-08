import os
import cv2
import torch
from torch.utils.data import Dataset


class MeterDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None, encoder=None):
        self.data_dir = data_dir
        self.transform = transform
        self.encoder = encoder
        self.samples = []

        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                img, label = parts[0], parts[1]
                self.samples.append((img, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        path = os.path.join(self.data_dir, img_name)

        # ✅ RGB LOAD
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read image: {path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ✅ AUGMENTATION (returns tensor already)
        if self.transform:
            image = self.transform(image=image)["image"]

        # ⚠️ NO unsqueeze, NO manual tensor conversion

        if self.encoder:
            label = self.encoder.encode(label)

        return image, torch.tensor(label, dtype=torch.long)


def ctc_collate_fn(batch):
    images, labels = zip(*batch)

    images = torch.stack(images, 0)   # (B, 3, H, W)
    lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels = torch.cat(labels)

    return images, labels, lengths