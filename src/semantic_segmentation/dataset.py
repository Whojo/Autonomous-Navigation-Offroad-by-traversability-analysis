import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from pathlib import Path
from enum import Enum

from utils.dataset import DEFAULT_IMAGE_AUGMENTATION_TRANSFORM


InputType = Enum("InputType", ["RGB", "RGBD"])


def _read_PIL_image(path: Path) -> Image:
    img = Image.open(path)
    img.load()
    return img


class SegmentationDataset(Dataset):
    """
    Dataset class for semantic segmentation.
    """

    def __init__(
        self,
        dir: Path,
        train: bool = True,
        image_transform: callable = None,
        geometric_transform: callable = None,
        multimodal_transform: callable = None,
        input_type: InputType = InputType.RGB,
    ):
        """
        Initialize the Dataset object.

        Args:
            dir (str): The directory path of the dataset.
            image_transform (callable, optional): Transforms to be applied on a
                rdg image. Defaults to None.
            geometric_transform (callable, optional): Transforms to be
                applied on the multimodal image and the target, as geometric
                properties should be kept between these for the segmentation
                task. Defaults to None.
            multimodal_transform (callable, optional): Transforms to be
                applied on the multimodal image. Defaults to None.
            train (bool, optional): Whether to use the training dataset.
                Defaults to True.
            input_type (InputType, optional): The type of input data.
                Can be InputType.RGB, InputType.RGBD. Defaults to
                InputType.RGB.
        """
        self.dataset_dir = dir
        self.image_dir = dir / "images"
        self.image_transform = image_transform
        self.geometric_transform = geometric_transform
        self.multimodal_transform = multimodal_transform

        self.input_type = input_type

        idx_file = "train_idx.npy" if train else "test_idx.npy"
        self.indexes = np.load(dir / idx_file)

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.indexes)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves the item at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the multimodal image, and target.
        """
        img_idx = self.indexes[idx]

        image = _read_PIL_image(self.image_dir / f"{img_idx:05d}.png")
        if self.image_transform:
            image = self.image_transform(image)

        image = transforms.ToTensor()(image)

        if self.input_type == InputType.RGB:
            multimodal = image.float()
        elif self.input_type == InputType.RGBD:
            depth = _read_PIL_image(self.image_dir / f"{img_idx:05d}d.png")
            normal = _read_PIL_image(self.image_dir / f"{img_idx:05d}n.png")

            depth = transforms.ToTensor()(depth)
            normal = transforms.ToTensor()(normal)

            multimodal = torch.cat((image, depth, normal), dim=0).float()

        if self.multimodal_transform:
            multimodal = self.multimodal_transform(multimodal)

        target = self._read_target(img_idx)
        if self.geometric_transform:
            all_data = torch.cat((multimodal, target), dim=0)
            all_data = self.geometric_transform(all_data)

            multimodal = all_data[: multimodal.shape[0]]
            target = all_data[-1]

        return multimodal, target

    def _read_target(self, idx):
        seg = np.load(self.dataset_dir / "targets" / f"{idx:05d}.npy")
        return transforms.ToTensor()(seg)


SEG_SIZE = (180, 320)
DEFAULT_SEG_GEOMETRIC_AUGMENTATION_TRANSFORM = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(
            SEG_SIZE,
            scale=(0.2, 1.0),
            antialias=True,
        ),
    ]
)

# Must adapt the multimodal transform to the semantic segmentation task
DEFAULT_SEG_GEOMETRIC_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(SEG_SIZE, antialias=True),
        transforms.Lambda(
            lambda img: transforms.functional.crop(
                img,
                top=SEG_SIZE[0] // 2,
                left=0,
                height=SEG_SIZE[0] // 2,
                width=SEG_SIZE[1],
            )
        ),
    ]
)

DEFAULT_SEG_MULTIMODAL_TRANSFORM = transforms.Compose(
    [
        # transforms.Normalize(
        #     mean=[], # TODO
        #     std=[], # TODO
        # )
    ]
)


def get_sets(
    dataset_dir: Path,
    *,
    image_augmentation_transform: callable = DEFAULT_IMAGE_AUGMENTATION_TRANSFORM,
    geometric_augmentation_transform: callable = DEFAULT_SEG_GEOMETRIC_AUGMENTATION_TRANSFORM,
    geometric_transform: callable = DEFAULT_SEG_GEOMETRIC_TRANSFORM,
    multimodal_transform: callable = DEFAULT_SEG_MULTIMODAL_TRANSFORM,
    **kwargs,
) -> (Dataset, Dataset, Dataset):
    """
    Get the train, validation, and test sets for semantic segmentation.

    Args:
        dataset_dir (Path): The directory containing the dataset.
        image_augmentation_transform (callable, optional): Data Augmentation
            transforms to be applied on a rdg image. Defaults to
            DEFAULT_IMAGE_AUGMENTATION_TRANSFORM.
        geometric_augmentation_transform (callable, optional): Data
            Augmentation transforms to be applied on the multimodal image and
            the target, as geometric properties should be kept between these
            for the segmentation task. Defaults to
            DEFAULT_SEG_GEOMETRIC_AUGMENTATION_TRANSFORM.
        geometric_transform (callable, optional): Transforms to be
            applied on the multimodal image and the target, as geometric
            properties should be kept between these for the segmentation
            task. Defaults to DEFAULT_SEG_GEOMETRIC_TRANSFORM.
        multimodal_transform (callable, optional): Transforms to be
            applied on the multimodal image. Defaults to
            DEFAULT_SEG_MULTIMODAL_TRANSFORM.

        **kwargs: Additional keyword arguments for SegmentationDataset
            (e.g. input_type). Refer to SegmentationDataset for more
            information.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: A tuple containing the train,
            validation, and test sets.
    """
    train_set = SegmentationDataset(
        dataset_dir,
        train=True,
        image_transform=image_augmentation_transform,
        geometric_transform=transforms.Compose(
            [geometric_transform, geometric_augmentation_transform]
        ),
        multimodal_transform=multimodal_transform,
        **kwargs,
    )

    val_set = SegmentationDataset(
        dataset_dir,
        train=True,
        image_transform=None,
        geometric_transform=geometric_transform,
        multimodal_transform=multimodal_transform,
        **kwargs,
    )

    test_set = SegmentationDataset(
        dataset_dir,
        train=False,
        image_transform=None,
        geometric_transform=geometric_transform,
        multimodal_transform=multimodal_transform,
        **kwargs,
    )

    return train_set, val_set, test_set
