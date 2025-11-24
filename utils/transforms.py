from typing import Dict

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transforms(image_size: int = 380) -> Dict[str, A.BasicTransform]:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_t = A.Compose(
        [
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.7),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    val_t = A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return {"train": train_t, "val": val_t}


