import numpy as np
import torch
from torchvision import transforms
import params.visualparams as viz
from models_development.multimodal_velocity_regression_alt.dataset import (
    DEFAULT_MULTIMODAL_TRANSFORM,
)


def get_model_input(
    crop: np.array,
    depth_crop: np.array,
    normals_crop: np.array,
    velocity: float,
):
    crop = transforms.ToTensor()(crop)
    depth_crop = transforms.ToTensor()(depth_crop)
    normals_crop = transforms.ToTensor()(normals_crop)

    multimodal_image = torch.cat((crop, depth_crop, normals_crop)).float()
    multimodal_image = DEFAULT_MULTIMODAL_TRANSFORM(multimodal_image)
    multimodal_image = torch.unsqueeze(multimodal_image, 0)
    multimodal_image = multimodal_image.to(viz.DEVICE)

    velocity = torch.tensor([velocity]).type(torch.float32).to(viz.DEVICE)
    velocity.unsqueeze_(1)

    return multimodal_image, velocity
