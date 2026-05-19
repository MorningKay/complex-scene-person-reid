"""Image transforms for Re-ID data pipelines."""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ImageSize = tuple[int, int]


def build_train_transform(
    image_size: ImageSize = (256, 128),
    random_erasing: bool = False,
    erase_prob: float = 0.5,
    padding: int = 0,
) -> transforms.Compose:
    steps = [
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if padding > 0:
        steps.extend(
            [
                transforms.Pad(padding),
                transforms.RandomCrop(image_size),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    if random_erasing:
        steps.append(transforms.RandomErasing(p=erase_prob))

    return transforms.Compose(steps)


def build_eval_transform(image_size: ImageSize = (256, 128)) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
