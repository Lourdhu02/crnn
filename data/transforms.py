import albumentations as A

def get_transforms(height=32, width=128):
    return A.Compose([
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.2),
        A.GaussNoise(p=0.2),
        A.Resize(height, width)
    ])
