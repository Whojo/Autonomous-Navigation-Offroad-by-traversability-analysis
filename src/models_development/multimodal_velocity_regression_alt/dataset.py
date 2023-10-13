import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import numpy as np

from torchvision import transforms
from sklearn.model_selection import train_test_split

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
        transform_image: callable = None,
        transform_depth: callable = None,
        transform_normal: callable = None,
    ) -> None:
        """Constructor of the class

        Args:
            traversal_costs_file (string): Path to the csv file which contains
                images index and their associated traversal cost
            images_directory (string): Directory with all the images
            transform_image (callable, optional): Transforms to be applied on a
                rdg image. Defaults to None.
            transform_depth (callable, optional): Transforms to be applied on a
                depth image. Defaults to None.
            transform_normal (callable, optional): Transforms to be applied on
                a normal image. Defaults to None.
        """
        self.traversal_costs_frame = pd.read_csv(
            traversal_costs_file, converters={"image_id": str}
        )

        self.images_directory = images_directory

        self.transform_image = transform_image
        self.transform_depth = transform_depth
        self.transform_normal = transform_normal

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
        if self.transform_image:
            image = self.transform_image(image)

        depth_image = Image.open(image_name + "d.png")
        if self.transform_depth:
            depth_image = self.transform_depth(depth_image)

        normal_map = Image.open(image_name + "n.png")
        if self.transform_normal:
            normal_map = self.transform_normal(normal_map)

        multimodal_image = torch.cat((image, depth_image, normal_map)).float()

        traversal_cost = self.traversal_costs_frame.loc[
            idx, "traversal_cost"
        ].astype(np.float32)
        linear_velocity = self.traversal_costs_frame.loc[
            idx, "linear_velocity"
        ]

        return multimodal_image, traversal_cost, linear_velocity


train_transform = transforms.Compose(
    [
        transforms.Resize(params.learning.IMAGE_SHAPE, antialias=True),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomCrop(100),
        transforms.ColorJitter(**params.learning.JITTER_PARAMS),
        # Black patch
        Cutout(0.5),
        Shadowcasting(0.5),
        transforms.ToTensor(),
        # Gaussian noise
        transforms.Lambda(lambda x: x + (0.001**0.5) * torch.randn(x.shape)),
        transforms.Normalize(
            mean=NORMALIZE_PARAMS["rbg"]["mean"],
            std=NORMALIZE_PARAMS["rbg"]["std"],
        ),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(params.learning.IMAGE_SHAPE, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=NORMALIZE_PARAMS["rbg"]["mean"],
            std=NORMALIZE_PARAMS["rbg"]["std"],
        ),
    ]
)

transform_depth = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(params.learning.IMAGE_SHAPE, antialias=True),
        transforms.Normalize(
            mean=NORMALIZE_PARAMS["depth"]["mean"],
            std=NORMALIZE_PARAMS["depth"]["std"],
        ),
    ]
)

transform_normal = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(params.learning.IMAGE_SHAPE, antialias=True),
        transforms.Normalize(
            mean=NORMALIZE_PARAMS["normal"]["mean"],
            std=NORMALIZE_PARAMS["normal"]["std"],
        ),
    ]
)


train_set = TraversabilityDataset(
    traversal_costs_file=params.learning.DATASET / "traversal_costs_train.csv",
    images_directory=params.learning.DATASET / "images_train",
    transform_image=train_transform,
    transform_depth=transform_depth,
    transform_normal=transform_normal,
)

val_set = TraversabilityDataset(
    traversal_costs_file=params.learning.DATASET / "traversal_costs_train.csv",
    images_directory=params.learning.DATASET / "images_train",
    transform_image=test_transform,
    transform_depth=transform_depth,
    transform_normal=transform_normal,
)

test_set = TraversabilityDataset(
    traversal_costs_file=params.learning.DATASET / "traversal_costs_test.csv",
    images_directory=params.learning.DATASET / "images_test",
    transform_image=test_transform,
    transform_depth=transform_depth,
    transform_normal=transform_normal,
)

train_size = params.learning.TRAIN_SIZE / (1 - params.learning.TEST_SIZE)
train_indices, val_indices = train_test_split(
    range(len(train_set)), train_size=train_size
)

train_set = Subset(train_set, train_indices)
val_set = Subset(val_set, val_indices)


def get_loader(set: Dataset, *, shuffle: bool = True) -> DataLoader:
    """Return a DataLoader for a given dataset

    Args:
        set (Dataset): Dataset to load
        shuffle (bool, optional): Whether to shuffle the data. Defaults to
            True.

    Returns:
        DataLoader: DataLoader for the given dataset
    """
    return DataLoader(
        set,
        batch_size=LEARNING["batch_size"],
        shuffle=shuffle,
        num_workers=12,  # Asynchronous data loading and augmentation
        pin_memory=True,  # Increase the transferring speed of the data to the GPU
    )


train_loader = get_loader(train_set)
val_loader = get_loader(val_set)
test_loader = get_loader(test_set, shuffle=False)
