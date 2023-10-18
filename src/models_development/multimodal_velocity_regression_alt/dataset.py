import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import numpy as np

from torchvision import transforms
from sklearn.model_selection import train_test_split

from pathlib import Path
from custom_transforms import Cutout, Shadowcasting
import params.learning
from params.learning import NORMALIZE_PARAMS, LEARNING


class TraversabilityDataset(Dataset):
    """Custom Dataset class to represent our dataset
    It includes data and information about the data

    Args:
        Dataset (class): Abstract class which represents a dataset
    """

    def __init__(
        self,
        traversal_costs_file: str,
        images_directory: str,
        image_transform: callable = None,
        multimodal_transform: callable = None,
    ) -> None:
        """Constructor of the class

        Args:
            traversal_costs_file (string): Path to the csv file which contains
                images index and their associated traversal cost
            images_directory (string): Directory with all the images
            image_transform (callable, optional): Transforms to be applied on a
                rdg image. Defaults to None.
            multimodal_transform (callable, optional): Transforms to be applied on the
                multimodal image. Defaults to None.
        """
        self.traversal_costs_frame = pd.read_csv(
            traversal_costs_file, converters={"image_id": str}
        )

        self.images_directory = images_directory
        self.image_transform = image_transform
        self.multimodal_transform = multimodal_transform

    def __len__(self) -> int:
        """Return the size of the dataset

        Returns:
            int: Number of samples
        """
        return len(self.traversal_costs_frame)

    def __getitem__(self, idx: int) -> tuple:
        """Allow to access a sample by its index

        Args:
            idx (int): Index of a sample

        Returns:
            tuple: Sample at index idx
            ([multimodal_image,
              traversal_cost,
              traversability_label,
              linear_velocity])
        """
        image_name = os.path.join(
            self.images_directory,
            self.traversal_costs_frame.loc[idx, "image_id"],
        )

        image = Image.open(image_name + ".png")
        if self.image_transform:
            image = self.image_transform(image)

        depth_image = Image.open(image_name + "d.png")
        normal_map = Image.open(image_name + "n.png")

        image = transforms.ToTensor()(image)
        depth_image = transforms.ToTensor()(depth_image)
        normal_map = transforms.ToTensor()(normal_map)

        multimodal_image = torch.cat((image, depth_image, normal_map)).float()
        if self.multimodal_transform:
            multimodal_image = self.multimodal_transform(multimodal_image)

        traversal_cost = self.traversal_costs_frame.loc[
            idx, "traversal_cost"
        ].astype(np.float32)
        linear_velocity = self.traversal_costs_frame.loc[
            idx, "linear_velocity"
        ].astype(np.float32)

        return multimodal_image, traversal_cost, linear_velocity


DEFAULT_IMAGE_AUGMENTATION_TRANSFORM = transforms.Compose(
    [
        transforms.ColorJitter(**params.learning.JITTER_PARAMS),
        # Black patch
        Cutout(0.5),
        Shadowcasting(0.5),
        # Gaussian noise
        # transforms.Lambda(lambda x: x + (0.001**0.5) * torch.randn(x.shape)),
    ]
)

DEFAULT_AUGMENTATION_TRANSFORM = transforms.Compose(
    [
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomCrop(100),
    ]
)

DEFAULT_MULTIMODAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(params.learning.IMAGE_SHAPE, antialias=True),
        transforms.Normalize(
            mean=NORMALIZE_PARAMS["mean"],
            std=NORMALIZE_PARAMS["std"],
        ),
    ]
)


def get_sets(
    dataset: Path,
    *,
    image_augmentation_transform: callable,
    augmentation_transform: callable,
    multimodal_transform: callable,
) -> (Dataset, Dataset, Dataset):
    """
    Returns trianing set, validation set and test set.

    Args:
    - dataset (Path): Path to the dataset directory.
    - image_augmentation_transform (callable): Transform to apply to the image
        and only for the training set.
    - augmentation_transform (callable): Transform to apply to the multimodal
        image and only for the training set.
    - multimodal_transform (callable): Transform to apply to the multimodal
        image on all sets.

    Returns:
    - Tuple[Dataset, Dataset, Dataset]: A tuple containing the three datasets,
        in the following order: training set, validation set, test set.
    """
    train_set = TraversabilityDataset(
        traversal_costs_file=dataset / "traversal_costs_train.csv",
        images_directory=dataset / "images_train",
        image_transform=image_augmentation_transform,
        multimodal_transform=transforms.Compose(
            [augmentation_transform, multimodal_transform]
        ),
    )

    val_set = TraversabilityDataset(
        traversal_costs_file=dataset / "traversal_costs_train.csv",
        images_directory=dataset / "images_train",
        image_transform=None,
        multimodal_transform=multimodal_transform,
    )

    test_set = TraversabilityDataset(
        traversal_costs_file=dataset / "traversal_costs_test.csv",
        images_directory=dataset / "images_test",
        image_transform=None,
        multimodal_transform=multimodal_transform,
    )

    return train_set, val_set, test_set


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
    dataset: Path,
    *,
    image_augmentation_transform: callable = DEFAULT_IMAGE_AUGMENTATION_TRANSFORM,
    augmentation_transform: callable = DEFAULT_AUGMENTATION_TRANSFORM,
    multimodal_transform: callable = DEFAULT_MULTIMODAL_TRANSFORM,
    batch_size: int = LEARNING["batch_size"],
) -> (DataLoader, DataLoader, DataLoader):
    """
    Returns train, validation and test dataloaders for the given dataset.

    Args:
    - dataset (Path): Path to the dataset.
    - image_augmentation_transform (callable): Transform to apply to the image
        and only for the training set. Defaults to
        DEFAULT_IMAGE_AUGMENTATION_TRANSFORM.
    - augmentation_transform (callable): Transform to apply to the multimodal
        image and only for the training set. Defaults to
        DEFAULT_AUGMENTATION_TRANSFORM.
    - multimodal_transform (callable): Transform to apply to the multimodal
        image on all sets. Defaults to DEFAULT_MULTIMODAL_TRANSFORM.
    - batch_size (int): Size of the batch. Defaults to
        LEARNING["batch_size"].

    Returns:
    - Tuple[DataLoader, DataLoader, DataLoader]: A tuple containing train,
          validation and test dataloaders.
    """
    train_set, val_set, test_set = get_sets(
        dataset,
        image_augmentation_transform=image_augmentation_transform,
        augmentation_transform=augmentation_transform,
        multimodal_transform=multimodal_transform,
    )

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


train_loader, val_loader, test_loader = get_dataloader(params.learning.DATASET)
