import torch
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
from pathlib import Path
import numpy as np

from sam_pipeline_utils import TargetType
from utils.dataset import (
    DEFAULT_IMAGE_AUGMENTATION_TRANSFORM,
    DEFAULT_AUGMENTATION_TRANSFORM,
    DEFAULT_MULTIMODAL_TRANSFORM,
)


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
        multimodal_transform: callable = None,
        target_type: TargetType = TargetType.COSTMAP,
        troncate: bool = True,
    ):
        """
        Initialize the Dataset object.

        Args:
            dir (str): The directory path of the dataset.
            image_transform (callable, optional): Transforms to be applied on a
                rdg image. Defaults to None.
            multimodal_transform (callable, optional): Transforms to be
                applied on the multimodal image. Defaults to None.
            train (bool, optional): Whether to use the training dataset.
                Defaults to True.
            target_type (TargetType, optional): The type of target data.
                Can be TargetType.FULL_SEGMENTATION, TargetType.COSTMAP,
                or TargetType.BOTH. Defaults to TargetType.COSTMAP.
            troncate (bool, optional): Whether to truncate the top half
                of the images. This is can be relevant as target
                segmentation are only computed for the bottom half of the
                images. Defaults to True.
        """
        self.dataset_dir = dir
        self.image_dir = dir / "images"
        self.image_transform = image_transform
        self.multimodal_transform = multimodal_transform

        self.target_type = target_type
        self.troncate = troncate

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
        depth = _read_PIL_image(self.image_dir / f"{img_idx:05d}d.png")
        normal = _read_PIL_image(self.image_dir / f"{img_idx:05d}n.png")
        target = self._read_target(img_idx)

        if self.troncate:
            half_height = image.size[1] // 2
            crop_box = (0, half_height, image.size[0], image.size[1])

            image = image.crop(crop_box)
            depth = depth.crop(crop_box)
            normal = normal.crop(crop_box)

            if self.target_type == TargetType.FULL_SEGMENTATION:
                target = target[half_height:, :]

            if self.target_type == TargetType.BOTH:
                seg, costmap = target
                seg = seg[half_height:, :]
                target = (seg, costmap)

        if self.image_transform:
            image = self.image_transform(image)

        image = transforms.ToTensor()(image)
        depth = transforms.ToTensor()(depth)
        normal = transforms.ToTensor()(normal)

        multimodal = torch.cat((image, depth, normal), dim=0).float()
        if self.multimodal_transform:
            multimodal = self.multimodal_transform(multimodal)

        return multimodal, target

    def _read_target(self, idx):
        if self.target_type in (TargetType.FULL_SEGMENTATION, TargetType.BOTH):
            seg = np.losad(self.dataset_dir / "targets" / f"{idx:05d}_seg.npy")
            seg = transforms.ToTensor()(seg)

        if self.target_type in (TargetType.COSTMAP, TargetType.BOTH):
            costmap = np.load(
                self.dataset_dir / "targets" / f"{idx:05d}_costmap.npy"
            )
            costmap = transforms.ToTensor()(costmap)

        if self.target_type == TargetType.BOTH:
            return (seg, costmap)
        elif self.target_type == TargetType.FULL_SEGMENTATION:
            return seg
        elif self.target_type == TargetType.COSTMAP:
            return costmap
        else:
            raise ValueError(f"Unknown target type: {self.target_type}")


def get_sets(
    dataset_dir: Path,
    *,
    image_augmentation_transform: callable = DEFAULT_IMAGE_AUGMENTATION_TRANSFORM,
    augmentation_transform: callable = DEFAULT_AUGMENTATION_TRANSFORM,
    multimodal_transform: callable = DEFAULT_MULTIMODAL_TRANSFORM,
    **kwargs,
) -> (Dataset, Dataset, Dataset):
    """
    Get the train, validation, and test sets for semantic segmentation.

    Args:
        dataset_dir (Path): The directory containing the dataset.
        image_augmentation_transform (callable, optional): The image
            augmentation transform function. Defaults to
            DEFAULT_IMAGE_AUGMENTATION_TRANSFORM.
        augmentation_transform (callable, optional): The augmentation
            transform function. Defaults to DEFAULT_AUGMENTATION_TRANSFORM.
        multimodal_transform (callable, optional): The multimodal transform
            function. Defaults to DEFAULT_MULTIMODAL_TRANSFORM.
        **kwargs: Additional keyword arguments for SegmentationDataset.
            either target_type or troncate. Refer to SegmentationDataset
            for more information.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: A tuple containing the train,
            validation, and test sets.
    """
    train_set = SegmentationDataset(
        dataset_dir,
        train=True,
        image_transform=image_augmentation_transform,
        multimodal_transform=transforms.Compose(
            [augmentation_transform, multimodal_transform]
        ),
        **kwargs,
    )

    val_set = SegmentationDataset(
        dataset_dir,
        train=True,
        image_transform=None,
        multimodal_transform=multimodal_transform,
        **kwargs,
    )

    test_set = SegmentationDataset(
        dataset_dir,
        train=False,
        image_transform=None,
        multimodal_transform=multimodal_transform,
        **kwargs,
    )

    return train_set, val_set, test_set
