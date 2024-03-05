import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np

from torchvision import transforms

from pathlib import Path

from utils.dataset import (
    DEFAULT_IMAGE_AUGMENTATION_TRANSFORM,
    DEFAULT_AUGMENTATION_TRANSFORM,
    DEFAULT_MULTIMODAL_TRANSFORM,
)


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
            traversal_costs_file,
            converters={"image_id": str},
            dtype={
                "traversal_cost": np.float32,
                "linear_velocity": np.float32,
            },
        )

        self.images_directory = images_directory
        self.image_transform = image_transform
        self.multimodal_transform = multimodal_transform

        self.images = self._get_image_list_with_suffix(".png")
        self.depths = self._get_image_list_with_suffix("d.png")
        self.normals = self._get_image_list_with_suffix("n.png")

    def _get_image_list_with_suffix(self, suffix: str) -> list:
        return [
            self._get_image_with_suffix(suffix, idx)
            for idx in range(len(self))
        ]

    def _get_image_with_suffix(self, suffix: str, idx: int) -> torch.Tensor:
        image_name = os.path.join(
            self.images_directory,
            self.traversal_costs_frame.loc[idx, "image_id"],
        )

        img = Image.open(image_name + suffix)
        img.load()
        return img

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
        image = self.images[idx]
        if self.image_transform:
            image = self.image_transform(image)

        depth_image = self.depths[idx]
        normal_map = self.normals[idx]

        image = transforms.ToTensor()(image)
        depth_image = transforms.ToTensor()(depth_image)
        normal_map = transforms.ToTensor()(normal_map)

        multimodal_image = torch.cat((image, depth_image, normal_map)).float()
        if self.multimodal_transform:
            multimodal_image = self.multimodal_transform(multimodal_image)

        traversal_cost = self.traversal_costs_frame.loc[idx, "traversal_cost"]
        linear_velocity = self.traversal_costs_frame.loc[
            idx, "linear_velocity"
        ]

        return multimodal_image, traversal_cost, linear_velocity


def get_sets(
    dataset: Path,
    *,
    image_augmentation_transform: callable = DEFAULT_IMAGE_AUGMENTATION_TRANSFORM,
    augmentation_transform: callable = DEFAULT_AUGMENTATION_TRANSFORM,
    multimodal_transform: callable = DEFAULT_MULTIMODAL_TRANSFORM,
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
