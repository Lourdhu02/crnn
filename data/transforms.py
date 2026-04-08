import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MorphologicalTransform(A.ImageOnlyTransform):
    def __init__(self, p=0.3):
        super().__init__(p=p)

    def apply(self, img, **params):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        if np.random.rand() > 0.5:
            return cv2.erode(img, kernel, iterations=1)
        return cv2.dilate(img, kernel, iterations=1)

    def get_transform_init_args_names(self):
        return ()


def get_transforms(height, width, mode="train"):
    if mode in ("val", "test"):
        return A.Compose([
            A.Resize(height, width),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])

    return A.Compose([
        A.Resize(height, width),

        # lighting
        A.OneOf([
            A.CLAHE(clip_limit=4.0),
            A.RandomBrightnessContrast(0.4, 0.4),
            A.RandomGamma(gamma_limit=(60, 140)),
        ], p=0.7),

        # glare / shadow
        A.OneOf([
            A.RandomSunFlare(
                flare_roi=(0.0, 0.0, 1.0, 0.5),
                src_radius=50,
                num_flare_circles_range=(1, 3),
            ),
            A.RandomShadow(num_shadows_limit=(1, 2)),
        ], p=0.25),

        # blur
        A.OneOf([
            A.MotionBlur(blur_limit=3),
            A.GaussianBlur(blur_limit=3),
        ], p=0.3),

        # noise
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.05)),
            A.ImageCompression(quality_range=(60, 95)),
        ], p=0.3),

        # stroke variation
        MorphologicalTransform(p=0.25),

        # geometry
        A.OneOf([
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.03, 0.03),
                rotate=(-3, 3),
            ),
            A.Perspective(scale=(0.01, 0.04)),
        ], p=0.4),

        # occlusion
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(0.05, 0.15),
            hole_width_range=(0.02, 0.08),
            p=0.15,
        ),

        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])