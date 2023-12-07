from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

from pathlib import Path

from models_development.multimodal_velocity_regression_alt.custom_transforms import (
    Cutout,
    Shadowcasting,
)
import params.learning
from params.learning import NORMALIZE_PARAMS, LEARNING


DEFAULT_IMAGE_AUGMENTATION_TRANSFORM = transforms.Compose(
    [
        transforms.GaussianBlur(3),
        transforms.GaussianBlur(7),
        transforms.GaussianBlur(13),
        Cutout(),
        Shadowcasting(),
    ]
)

DEFAULT_AUGMENTATION_TRANSFORM = augmentation_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(
            params.learning.IMAGE_SHAPE,
            scale=(0.2, 1.0),
            ratio=(3, 3),
            antialias=True,
        ),
    ]
)

DEFAULT_MULTIMODAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(params.learning.IMAGE_SHAPE, antialias=True),
        transforms.Normalize(**NORMALIZE_PARAMS),
    ]
)


def _set_to_loader(
    set: Dataset,
    *,
    shuffle: bool = True,
    batch_size: int = LEARNING["batch_size"],
) -> DataLoader:
    """Return a DataLoader for a given dataset

    Args:
        set (Dataset): Dataset to load
        shuffle (bool, optional): Whether to shuffle the data. Defaults to
            True.
        batch_size (int, optional): Size of the batch. Defaults to
            LEARNING["batch_size"].

    Returns:
        DataLoader: DataLoader for the given dataset
    """
    return DataLoader(
        set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=12,  # Asynchronous data loading and augmentation
        pin_memory=True,  # Increase the transferring speed of the data to the GPU
    )


def get_dataloader(
    train_set: Dataset,
    val_set: Dataset,
    test_set: Dataset,
    *,
    batch_size: int = LEARNING["batch_size"],
) -> (DataLoader, DataLoader, DataLoader):
    """
    Returns train, validation and test dataloaders for the given dataset.

    Args:
    - train_set (Dataset): Training set.
    - val_set (Dataset): Validation set.
    - test_set (Dataset): Test set.
    - batch_size (int): Size of the batch. Defaults to
        LEARNING["batch_size"].

    Returns:
    - Tuple[DataLoader, DataLoader, DataLoader]: A tuple containing train,
          validation and test dataloaders.
    """
    train_size = params.learning.TRAIN_SIZE / (1 - params.learning.TEST_SIZE)
    train_indices, val_indices = train_test_split(
        range(len(train_set)), train_size=train_size
    )

    train_set = Subset(train_set, train_indices)
    val_set = Subset(val_set, val_indices)

    train_loader = _set_to_loader(train_set, batch_size=batch_size)
    val_loader = _set_to_loader(val_set, batch_size=batch_size)
    test_loader = _set_to_loader(
        test_set, shuffle=False, batch_size=batch_size
    )

    return train_loader, val_loader, test_loader
