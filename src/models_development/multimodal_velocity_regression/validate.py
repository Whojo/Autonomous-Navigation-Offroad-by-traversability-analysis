import torch
import torch.nn as nn
from tqdm.notebook import tqdm


def validate(model: torch.nn.Module,
             device: str,
             val_loader: torch.utils.data.DataLoader,
             criterion_regression: nn.Module,
             epoch: int) -> tuple:
    """Validate the model for one epoch

    Args:
        model (Model): The model to validate
        device (string): The device to use (cpu or cuda)
        val_loader (Dataloader): The validation data loader
        criterion_classification (Loss): The classification loss to use
        criterion_regression (Loss): The regression loss to use
        bins_midpoints (ndarray): The midpoints of the bins used to discretize
        the traversal costs
        epoch (int): The current epoch
        
    Returns:
        double, double, double: The validation loss, the validation accuracy
        and the validation regression loss
    """
    # Initialize the validation loss and accuracy
    val_regression_loss = 0.
    
    # Configure the model for testing
    # (turn off dropout layers, batchnorm layers, etc)
    model.eval()
    
    # Add a progress bar
    val_loader_pbar = tqdm(val_loader, unit="batch")
    
    # Turn off gradients computation (the backward computational graph is
    # built during the forward pass and weights are updated during the backward
    # pass, here we avoid building the graph)
    with torch.no_grad():
        
        # Loop over the validation batches
        for images,\
            traversal_costs,\
            linear_velocities in val_loader_pbar:

            # Print the epoch and validation mode
            val_loader_pbar.set_description(f"Epoch {epoch} [val]")

            # Move images and traversal scores to GPU (if available)
            images = images.to(device)
            traversal_costs = traversal_costs.to(device)
            linear_velocities = linear_velocities.type(torch.float32).to(device)
            
            # Add a dimension to the linear velocities tensor
            linear_velocities.unsqueeze_(1)
            
            # Perform forward pass (only, no backpropagation)
            predicted_traversability_cost = model(images, linear_velocities)
            # predicted_traversal_scores = nn.Softmax(dim=1)(model(images))

            # Compute loss
            loss = criterion_regression(predicted_traversability_cost,
                                            traversal_costs)

            # Print the batch loss next to the progress bar
            val_loader_pbar.set_postfix(batch_loss=loss.item())

            # Accumulate batch loss to average over the epoch
            val_regression_loss += loss.item()
        
    # Compute the losses and accuracies
    val_regression_loss /= len(val_loader)
    
    return val_regression_loss
