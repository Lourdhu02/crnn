import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(height, width):
    return A.Compose([

        A.Resize(height, width),

        A.OneOf([
            A.CLAHE(clip_limit=3.0),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            A.RandomGamma(gamma_limit=(80, 120)),
        ], p=0.6),

        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
        ], p=0.3),

        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.1)),
            A.ISONoise(color_shift=(0.01, 0.05)),
        ], p=0.3),

        A.OneOf([
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(0.03, 0.08),
                rotate=(-5, 5),
                shear=(-3, 3)
            ),
            A.Perspective(scale=(0.02, 0.05)),
        ], p=0.5),

        A.OneOf([
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.05, 0.1),
                hole_width_range=(0.05, 0.1)
            ),
            A.GridDistortion(),
        ], p=0.2),

        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])