import torch
import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from functools import reduce, lru_cache
from pathlib import Path
from enum import Enum

from utils.model import get_model_input
from utils.grid import get_grid_lists
import params.dataset
from data_preparation.create_dataset import (
    get_patch_dimension,
    RectangleDim,
)
from models_development.multimodal_velocity_regression_alt.model import (
    ResNet18Velocity_Regression_Alt,
)

TargetType = Enum("TargetType", ["FULL_SEGMENTATION", "COSTMAP", "BOTH"])


@lru_cache(maxsize=1)
def get_mask_generator(
    *,
    model_name: str = params.dataset.sam_model_name,
    checkpoint: str = params.dataset.sam_checkpoint,
    device: str = "cuda",
) -> SamAutomaticMaskGenerator:
    """
    Returns the Segment Anything Model that generates masks for a given image.

    Args:
        model_name (str): Name of the SAM model to use. Default to "vit_h"
            (params.dataset.sam_model_name).
        checkpoint (str): Path to the checkpoint file for the SAD model.
            Default to "src/semantic_segmentation/models/sam_vit_h.pth"
            (params.dataset.sam_checkpoint).
        device (str): Device to use to runn the model. Default to "cuda".

    Returns:
        SamAutomaticMaskGenerator: Generates segmentation masks for a given
            image.
    """
    sam = sam_model_registry[model_name](checkpoint=checkpoint).to(device)
    return SamAutomaticMaskGenerator(
        sam, stability_score_thresh=params.dataset.stability_score_thresh
    )


def _get_random_centroid(segmentation: np.array) -> tuple:
    y, x = np.nonzero(segmentation)
    i = np.random.randint(len(x))
    return x[i], y[i]


def _keep_patch_inside(
    patch: RectangleDim, segmentation: np.array
) -> RectangleDim:
    width, height = segmentation.shape
    crop_width = patch.max_x - patch.min_x
    crop_height = patch.max_y - patch.min_y

    min_x = max(patch.min_x, 0)
    min_y = max(patch.min_y, 0)

    max_x = patch.min_x + min(crop_width, width)
    max_y = patch.min_y + min(crop_height, height)

    return RectangleDim(min_x, max_x, min_y, max_y)


def get_random_patch(segmentation: np.array) -> RectangleDim:
    """
    Randomly choose a pixel in the segmentation mask and extract a patch
    around it.

    Args:
        segmentation (np.ndarray): The segmentation to extract the patch from.

    Returns:
        RectangleDim: The extracted patch.
    """
    point = _get_random_centroid(segmentation)

    point = np.array([point])
    patch = get_patch_dimension(point)
    return _keep_patch_inside(patch, segmentation)


def _sample_random_patches(segmentation: np.array, *, nb_patch: int) -> list:
    for _ in range(nb_patch):
        yield get_random_patch(segmentation)


def crop(img: np.array, patch: RectangleDim) -> np.array:
    """
    Crop the input image based on the given patch.

    Args:
        img (numpy.ndarray): The input image to crop.
        patch (RectanlgeDim): The patch object containing the coordinates to
            crop the image.

    Returns:
        numpy.ndarray: The cropped image.
    """
    return img[patch.min_y : patch.max_y, patch.min_x : patch.max_x]


def _get_segmentation_bottom_half(segmentation: np.array) -> np.array:
    """
    Only the bottom half of the mask should be kept
    (i.e. the part of the image projected into the costmap).
    """
    bottom_mask = np.zeros_like(segmentation, dtype=bool)
    half_height = segmentation.shape[0] // 2
    bottom_mask[half_height:, :] = 1

    return segmentation & bottom_mask


@lru_cache(maxsize=1)
def get_vision_model(weight_path: Path = params.dataset.weight_path):
    """
    Loads a ResNet18Velocity_Regression_Alt model from the specified weight
    path and returns it.

    Args:
        weight_path (str): The path to the weight file to load.
            Defaults to params.dataset.weight_path.

    Returns:
        ResNet18Velocity_Regression_Alt: The loaded model.
    """
    model = ResNet18Velocity_Regression_Alt()
    model.load_state_dict(torch.load(weight_path))
    model.eval().to("cuda")
    return model


def _get_cost_of_patch(
    img: np.array,
    depth: np.array,
    normal: np.array,
    patch: np.array,
    *,
    model=None,
) -> float:
    if model is None:
        model = get_vision_model()

    img_crop = crop(img, patch)
    depth_crop = crop(depth, patch)
    normal_crop = crop(normal, patch)

    input = get_model_input(img_crop, depth_crop, normal_crop, 1)
    output = model(*input).cpu().item()
    return output


def get_cost(
    mask: dict,
    img: np.array,
    depth: np.array,
    normal: np.array,
    *,
    nb_patch: int = params.dataset.nb_total_patch,
) -> float:
    """
    Computes the traversal cost of a given segmentation mask.

    After randomly sampling patches, it computes the cost of each patch using
    the provided image, depth, and normal data. The final cost is the mean of
    all patch costs.

    Args:
        mask (dict): A dictionary containing the segmentation mask.
        depth (np.array): The depth map of the input image.
        normal (np.array): The surface normal map of the input image.
        normal (np.ndarray): The surface normal data.
        nb_patch (int, optional): The number of patches to sample.
            Defaults to params.dataset.nb_total_patch.

    Returns:
        float: The cost of the segmentation mask.
    """

    seg = _get_segmentation_bottom_half(mask["segmentation"])
    if not np.any(seg):
        return None

    patches = _sample_random_patches(seg, nb_patch=nb_patch)
    costs = map(
        lambda patch: _get_cost_of_patch(img, depth, normal, patch),
        patches,
    )

    return np.mean(list(costs))


def _compute_intersection(
    segmentation: np.array, patch: RectangleDim
) -> float:
    segmentation_crop = crop(segmentation, patch)
    nb_mask_pixels = np.sum(segmentation_crop)
    nb_pixels = np.prod(segmentation_crop.shape)
    return nb_mask_pixels / nb_pixels


def _keep_most_centered_patches(
    segmentation: np.array, patches: list, nb_patch: int
) -> list:
    most_centered_patches = sorted(
        patches,
        key=lambda patch: _compute_intersection(segmentation, patch),
        reverse=True,
    )

    return most_centered_patches[:nb_patch]


def get_center_cost(
    mask: dict,
    img: np.array,
    depth: np.array,
    normal: np.array,
    *,
    nb_total_patch: int = params.dataset.nb_total_patch,
    nb_centered_patch: int = params.dataset.nb_centered_patch,
) -> float:
    """
    Computes the traversal cost of a given segmentation mask.

    After randomly sampling patches, patches are filtered to only keep
    nb_centered_patch most centered ones. Then, it computes the cost of each
    patch using the provided image, depth, and normal data.
    The final cost is the mean of all patch costs.

    Args:
        mask (dict): A dictionary containing the segmentation mask.
        img (np.ndarray): The image data.
        depth (np.array): The depth map of the input image.
        normal (np.array): The surface normal map of the input image.
        nb_total_patch (int, optional): The total number of patches to sample.
            Defaults to params.dataset.nb_total_patch (100).
        nb_centered_patch (int, optional): The number of most centered patches
            to keep. Defaults to params.dataset.nb_centered_patch (10).

    Returns:
        float: The cost of the segmentation mask.
    """
    seg = _get_segmentation_bottom_half(mask["segmentation"])
    if not np.any(seg):
        return None

    patches = _sample_random_patches(seg, nb_patch=nb_total_patch)
    patches = _keep_most_centered_patches(seg, patches, nb_centered_patch)

    costs = map(
        lambda patch: _get_cost_of_patch(img, depth, normal, patch),
        patches,
    )
    return np.mean(list(costs))


def get_iqm_center_cost(
    mask: dict,
    img: np.array,
    depth: np.array,
    normal: np.array,
    *,
    nb_total_patch: int = params.dataset.nb_total_patch,
    nb_centered_patch: int = params.dataset.nb_centered_patch,
) -> float:
    """
    Computes the traversal cost of a given segmentation mask.

    After randomly sampling patches, patches are filtered to only keep
    nb_centered_patch most centered ones. Then, it computes the cost of each
    patch using the provided image, depth, and normal data.
    The final cost is the Inter-Quartile Mean (IQM) of all patch costs.

    Args:
        mask (dict): A dictionary containing the segmentation mask.
        img (np.ndarray): The image data.
        depth (np.array): The depth map of the input image.
        normal (np.array): The surface normal map of the input image.
        nb_total_patch (int, optional): The total number of patches to sample.
            Defaults to params.dataset.nb_total_patch (100).
        nb_centered_patch (int, optional): The number of most centered patches
            to keep. Defaults to params.dataset.nb_centered_patch (10).

    Returns:
        float: The cost of the segmentation mask.
    """
    seg = _get_segmentation_bottom_half(mask["segmentation"])
    if not np.any(seg):
        return None

    patches = _sample_random_patches(seg, nb_patch=nb_total_patch)
    patches = _keep_most_centered_patches(seg, patches, nb_centered_patch)

    costs = map(
        lambda patch: _get_cost_of_patch(img, depth, normal, patch),
        patches,
    )
    costs = list(costs)

    q1, q3 = np.percentile(costs, [25, 75])
    filtered_costs = [cost for cost in costs if q1 <= cost <= q3]
    if filtered_costs == []:
        # When not enough values in costs
        return np.mean(costs)

    return np.mean(filtered_costs)


def compute_completeness(masks: list) -> float:
    """
    Computes the completeness of a segmentation, i.e. the ratio of pixels
    that are part of the segmentation mask.

    Args:
        masks (list): A list of segmentation masks, where each mask is a
            dictionary with a "segmentation" key.

    Returns:
        float: The completeness of the segmentation, as a value between 0
            and 1.
    """
    if len(masks) == 0:
        return 0

    segmented_pxl = np.sum(
        reduce(np.logical_or, (m["segmentation"] for m in masks))
    )
    all_pxl = np.prod(masks[0]["segmentation"].shape)

    return segmented_pxl / all_pxl


def filter_intersection(masks: list) -> list:
    """
    Filters intersecting pixels from a segmentation so that each pixel is
    only part of one segmentation mask (i.e. there should be no intersection
    between masks).

    Args:
        masks (list): A list of dictionaries, where each dictionary contains a
            binary mask (with "segmentation" key) and its area (with "area"
            key).

    Returns:
        list: A list of dictionaries, where each dictionary contains an
            exclusive binary mask and its area.
    """
    if len(masks) == 0:
        raise ValueError("Empty list of masks provided.")

    union_mask = np.zeros_like(masks[0]["segmentation"], dtype=bool)
    exclusive_masks = []
    for mask in sorted(masks, key=(lambda x: x["area"])):
        new_segmentation = mask["segmentation"] & ~union_mask
        if new_segmentation.sum() == 0:
            continue

        exclusive_masks.append(
            {
                "segmentation": new_segmentation,
                "area": new_segmentation.sum(),
            }
        )
        union_mask |= mask["segmentation"]

    return exclusive_masks


def downsample_to_grid(img: np.array, *, downsampling_fct=np.max) -> np.array:
    """
    Project a grid onto an image and downsample the image to this grid
    resolution in order to produce a low-resolution bird-eye view. This
    view is useful for plannification algorithms.

    Args:
        img (np.array): The input image to downsample and project.
        downsampling_fct (function, optional): The downsampling function to
            apply to each patch in order to convert it into a single pixel.
            Defaults to `np.max`.

    Returns:
        np.array: A low-resolution and bird-eye view version of the input
            image.
    """

    rectangle_list, grid_list = get_grid_lists()
    grid = np.zeros_like(rectangle_list, dtype=float)

    X, Y = rectangle_list.shape
    for x in range(X):
        for y in range(Y):
            patch = rectangle_list[x, y]
            if patch is None:
                continue

            poly_points = grid_list[x, y]
            poly_mask = np.zeros_like(img, dtype=np.uint8)
            poly_mask = cv2.fillPoly(poly_mask, [poly_points], True)

            pxls = img[poly_mask == True]
            grid[x, y] = downsampling_fct(pxls)

    # XXX: Reverse the grid to have the same orientation as the image.
    # => prevent refactoring the `get_grid_lists` function.
    return grid[::-1, :]


def fill_segmentation(masks: list) -> list:
    """
    Fill segmentation masks from the largest to the smallest in order to
    form a complete segmentation (i.e. the union of all masks has no hole).

    Only necessary modification are kept.

    Args:
        masks (list): A list of dictionaries, where each dictionary contains a
            binary mask (with "segmentation" key) and its area (with "area"
            key).

    Returns:
        list: A list of dictionaries, where each dictionary contains a
            binary mask and its area.
    """
    if len(masks) == 0:
        raise ValueError("Empty list of masks provided.")

    union_mask = reduce(np.logical_or, [m["segmentation"] for m in masks])
    dilation_kernel = np.ones((3, 3), dtype=np.uint8)

    complete_segmentation_list = []
    masks_to_process = sorted(masks, key=(lambda x: x["area"]), reverse=True)

    while not np.all(union_mask):
        mask = masks_to_process.pop(0)
        dilated_segmentation = cv2.dilate(
            mask["segmentation"].astype(np.uint8),
            dilation_kernel,
            iterations=1,
        ).astype(bool)

        # If the dilated segmentation does not change the union mask,
        # then the mask is not close to any holes anymore
        # and it won't help us any further to fill the holes.
        # Save it and move on to the next mask.
        extra = dilated_segmentation & ~union_mask
        if not np.any(extra):
            complete_segmentation_list.append(mask)
        else:
            new_segmentation = mask["segmentation"] | extra
            masks_to_process.append(
                {
                    "segmentation": new_segmentation,
                    "area": new_segmentation.sum(),
                }
            )

        union_mask |= extra.astype(bool)

    complete_segmentation_list.extend(masks_to_process)
    return complete_segmentation_list
