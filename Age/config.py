import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 4e-5
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 4
NUM_EPOCHS = 10
NUM_WORKERS = 10
CHECKPOINT_FILE = "b7_classifier_12.pth.tar"
PIN_MEMORY = True
SAVE_MODEL = True
LOAD_MODEL = True

# Data augmentation for images
train_transforms = A.Compose(
    [
        A.Resize(width=760, height=760),
        A.RandomCrop(height=728, width=728),
        A.ToGray(p=0.50),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.0),
        A.Blur(p=0.1),
        A.CLAHE(p=0.0),
        A.ColorJitter(p=0.3),
        A.CoarseDropout(max_holes=12, max_height=20, max_width=20, p=0.3),
        A.IAAAffine(shear=30, rotate=0, p=0.2, mode="constant"),
        A.Normalize(
            mean=[-0.0659, -0.1556, -0.3308],
            std=[1.0295, 0.9747, 0.7992],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

val_transforms = A.Compose(
    [
        A.Resize(height=728, width=728),
        A.Normalize(
            mean=[-0.0659, -0.1556, -0.3308],
            std=[1.0295, 0.9747, 0.7992],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)