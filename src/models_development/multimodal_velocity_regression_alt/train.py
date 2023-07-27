import torch
import torch.nn as nn
from tqdm.notebook import tqdm


def train(model: nn.Module,
          device: str,
          train_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          criterion_regression: nn.Module,
          epoch: int) -> tuple:
    """Train the model for one epoch

    Args:
        model (Model): The model to train
        device (string): The device to use (cpu or cuda)
        train_loader (Dataloader): The training data loader
        optimizer (Optimizer): The optimizer to use
        criterion_classification (Loss): The classification loss to use
        criterion_regression (Loss): The regression loss to use
        bins_midpoints (ndarray): The midpoints of the bins used to discretize
        the traversal costs
        epoch (int): The current epoch

    Returns:
        double, double, double: The training loss, the training accuracy and
        the training regression loss
    """
    # Initialize the training loss and accuracy
    train_regression_loss = 0.
    
    # Configure the model for training
    # (good practice, only necessary if the model operates differently for
    # training and validation)
    model.train()
    
    # Add a progress bar
    train_loader_pbar = tqdm(train_loader, unit="batch")
    
    # Loop over the training batches
    for images,\
        traversal_costs,\
        linear_velocities in train_loader_pbar:

        # Print the epoch and training mode
        train_loader_pbar.set_description(f"Epoch {epoch} [train]")
        
        # Move images and traversal scores to GPU (if available)
        images = images.to(device)
        traversal_costs = traversal_costs.to(device)
        
        linear_velocities = linear_velocities.type(torch.float32).to(device)
        
        # Add a dimension to the linear velocities tensor
        linear_velocities.unsqueeze_(1)

        # Zero out gradients before each backpropagation pass, to avoid that
        # they accumulate
        optimizer.zero_grad()
        
        # Perform forward pass
        predicted_traversability_cost = model(images, linear_velocities)
        
        # Compute loss 
        loss = criterion_regression(predicted_traversability_cost, traversal_costs)
        
        # Print the batch loss next to the progress bar
        train_loader_pbar.set_postfix(batch_loss=loss.item())
        
        # Perform backpropagation (update weights)
        loss.backward()
        
        # Adjust parameters based on gradients
        optimizer.step()
        
        # Compute and accumulate the batch loss
        train_regression_loss += loss.item()

    scheduler.step()
    
    # Compute the losses and accuracies
    train_regression_loss /= len(train_loader)
        
    return train_regression_loss
