import torch
from data.dataset import MeterDataset, ctc_collate_fn
from data.label_encoder import LabelEncoder
from data.transforms import get_transforms

def test_dataset():
    encoder = LabelEncoder("0123456789.")
    transform = get_transforms(32,128)
    ds = MeterDataset("data/train", "data/train_labels.txt", transform, encoder)
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(label, torch.Tensor)

def test_collate():
    encoder = LabelEncoder("0123456789.")
    transform = get_transforms(32,128)
    ds = MeterDataset("data/train", "data/train_labels.txt", transform, encoder)
    batch = [ds[i] for i in range(min(2,len(ds)))]
    images, labels, lengths = ctc_collate_fn(batch)
    assert images.dim() == 4
    assert labels.dim() == 1
    assert lengths.dim() == 1
