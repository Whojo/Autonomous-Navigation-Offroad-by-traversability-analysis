import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def squared_error(
    predicted_costs: torch.Tensor, true_costs: torch.Tensor
) -> torch.Tensor:
    """
    Apply the squared error function to a predicted and true traversal cost

    Args:
        predicted_costs (Tensor): Predicted traversal cost
        true_costs (Tensor): True traversal cost

    Returns:
        Tensor: The squared error between the predicted and true traversal cost
    """
    return (predicted_costs - true_costs) ** 2


def test(
    model: nn.Module,
    device: str,
    test_loader: torch.utils.data.DataLoader,
    criterion_regression: nn.Module,
) -> tuple:
    """Test the model on the test set

    Args:
        model (nn.Module): The model to test
        device (str): The device to use for the computations
        test_loader (torch.utils.data.DataLoader): The dataloader for the
        test set
        criterion_classification (nn.Module): The loss function for the
        classification task
        criterion_regression (nn.Module): The loss function for the
        regression task
        bins_midpoints (np.ndarray): The midpoints of the bins used for the
        discretization of the traversal cost

    Returns:
        double, double, double, list, list: The average loss, the accuracy,
        the regression loss, the list of the uncertainties, the list of
        regression losses
    """
    test_regression_loss = 0.0

    # Configure the model for testing
    model.eval()

    test_regression_losses = []

    with torch.no_grad():
        # Loop over the testing batches
        for images, traversal_costs, linear_velocities in test_loader:
            images = images.to(device)
            traversal_costs = traversal_costs.to(device)
            linear_velocities = linear_velocities.type(torch.float32).to(
                device
            )

            # Add a dimension to the linear velocities tensor
            linear_velocities.unsqueeze_(1)

            # Perform forward pass
            predicted_traversability_score = model(images, linear_velocities)

            # Compute and accumulate the batch loss
            test_regression_loss += criterion_regression(
                predicted_traversability_score, traversal_costs
            ).item()

            # Compute the loss for each sample
            test_regression_losses.append(
                squared_error(
                    predicted_traversability_score, traversal_costs
                ).to("cpu")
            )

    # Compute the loss and accuracy
    test_regression_loss /= len(test_loader)

    return test_regression_loss, test_regression_losses


def test_models(
    models: list,
    device: str,
    test_loader: torch.utils.data.DataLoader,
    criterion_classification: nn.Module,
    criterion_regression: nn.Module,
    bins_midpoints: np.ndarray,
) -> tuple:
    """Test the models on the test set

    Args:
        models (list): List of models to test
        device (string): The device to use for the computations
        test_loader (torch.utils.data.DataLoader): The dataloader for the test
        set
        criterion_classification (nn.Module): The loss function for the
        classification task
        criterion_regression (nn.Module): The loss function for the regression
        task
        bins_midpoints (np.ndarray): The midpoints of the bins used for the
        discretization of the traversal cost

    Returns:
        double, double, double, list, list: The average loss, the accuracy, the
        regression loss, the list of the uncertainties, the list of regression
        losses
    """
    # Testing
    test_regression_loss = 0.0

    # Configure the models for testing
    for model in models:
        model.eval()

    test_regression_losses = []
    uncertainties = []

    with torch.no_grad():
        # Loop over the testing batches
        for (
            images,
            traversal_costs,
            traversability_labels,
            linear_velocities,
        ) in test_loader:
            images = images.to(device)
            traversal_costs = traversal_costs.to(device)
            linear_velocities = linear_velocities.type(torch.float32).to(
                device
            )

            # Add a dimension to the linear velocities tensor
            linear_velocities = linear_velocities.unsqueeze_(1)

            # expected_traversal_costs = torch.zeros(traversal_costs.shape[0], 1).to(device)

            # uncert = torch.zeros(traversal_costs.shape[0]).to(device)

            probabilities = torch.zeros(traversal_costs.shape[0], 10).to(
                device
            )

            exp_costs = torch.zeros(traversal_costs.shape[0], len(models)).to(
                device
            )

            # Perform forward pass
            for index, model in enumerate(models):
                predicted_traversability_labels = model(
                    images, linear_velocities
                )

                # Apply the softmax function to the predicted traversability labels
                probabilities += nn.Softmax(dim=1)(
                    predicted_traversability_labels
                )

                # Compute the expected traversal cost over the bins
                # expected_traversal_costs += torch.matmul(probabilities, bins_midpoints)

                exp_costs[:, index] = torch.matmul(
                    nn.Softmax(dim=1)(predicted_traversability_labels),
                    bins_midpoints,
                )[:, 0]

            probabilities /= len(models)

            # Compute the expected traversal cost over the bins
            expected_traversal_costs = torch.matmul(
                probabilities, bins_midpoints
            )

            variance = torch.var(exp_costs, dim=1)
            uncertainties.append(variance.to("cpu"))

            # Compute and accumulate the batch loss
            test_regression_loss += criterion_regression(
                expected_traversal_costs[:, 0], traversal_costs
            ).item()

            # Compute the loss for each sample
            test_regression_losses.append(
                squared_error(
                    expected_traversal_costs[:, 0], traversal_costs
                ).to("cpu")
            )

    # Compute the regression loss
    test_regression_loss /= len(test_loader)

    return test_regression_loss, test_regression_losses, uncertainties
