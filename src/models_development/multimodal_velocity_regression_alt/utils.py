import torch
from torch.utils.data import DataLoader

# Import custom modules and packages
import params.learning
from dataset import TraversabilityDataset, DEFAULT_MULTIMODAL_TRANSFORM


def compute_mean_std(
    images_directory: str, traversal_costs_file: str
) -> tuple:
    """Compute the mean and standard deviation of the images of the dataset

    Args:
        images_directory (string): Directory with all the images
        traversal_costs_file (string): Name of the csv file which contains
        images index and their associated traversal cost

    Returns:
        tuple: Mean and standard deviation of the dataset
    """
    dataset = TraversabilityDataset(
        traversal_costs_file=params.learning.DATASET / traversal_costs_file,
        images_directory=params.learning.DATASET / images_directory,
        multimodal_transform=DEFAULT_MULTIMODAL_TRANSFORM,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
    )

    cnt = 0
    first_moment = torch.empty(7)
    second_moment = torch.empty(7)

    for images, _, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images**2, dim=[0, 2, 3])
        first_moment = (cnt * first_moment + sum_) / (cnt + nb_pixels)
        second_moment = (cnt * second_moment + sum_of_square) / (
            cnt + nb_pixels
        )
        cnt += nb_pixels

    mean = first_moment
    std = torch.sqrt(second_moment - first_moment**2)

    return mean, std
