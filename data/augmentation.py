import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size=(224, 224)):
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),

        # ✅ Contrast Enhancement
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),

        # ✅ Noise Reduction
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),

        # ✅ Brightness and Contrast Variation
        A.RandomBrightnessContrast(p=0.4),

        # ✅ Color Jittering
        A.HueSaturationValue(p=0.4),

        # ✅ Random Flips
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # ✅ Random Rotation
        A.Rotate(limit=15, p=0.5),

        # ✅ Normalization
        A.Normalize(mean=(0.5,), std=(0.5,), always_apply=True),

        # ✅ Convert to Tensor
        ToTensorV2()
    ])


def get_valid_transforms(image_size=(224, 224)):
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.5,), std=(0.5,), always_apply=True),
        ToTensorV2()
    ])
