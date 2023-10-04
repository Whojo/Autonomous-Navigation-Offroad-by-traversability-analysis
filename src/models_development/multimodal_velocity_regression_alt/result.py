from tabulate import tabulate
import os
import torch
import matplotlib.pyplot as plt

from typing import List, Any
from pathlib import Path

# Import custom packages and modules
import params.learning


plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def parameters_table(dataset: Path,
                     learning_params: dict) -> List[List[Any]]:
    """Generate a table containing the parameters used to train the network
    
    Args:
        dataset (Path): The path to the dataset
        learning_params (dict): The parameters used to train the network

    Returns:
        table: The table of parameters
    """
    # Generate the description of the training parameters
    data = [
        [
            "Dataset",
            "Train size",
            "Validation size",
            "Test size",
            "Batch size",
            "Nb epochs",
            "Learning rate",
            "Weight decay",
            "Momentum",
        ],
        [
            dataset.name,
            params.learning.TRAIN_SIZE,
            params.learning.VAL_SIZE,
            params.learning.TEST_SIZE,
            learning_params["batch_size"],
            learning_params["nb_epochs"],
            learning_params["learning_rate"],
            learning_params["weight_decay"],
            learning_params["momentum"],
        ],
    ]
    
    # Generate the table
    table = tabulate(data,
                     headers="firstrow",
                     tablefmt="fancy_grid",
                     maxcolwidths=20,
                     numalign="center",)
    
    return table


def generate_log(results_directory: Path,
                 test_regression_loss: float,
                 parameters_table: List[List[Any]],
                 model: torch.nn.Module,
                 regression_loss_values: torch.Tensor) -> None:
    """Create a directory to store the results of the training and save the
    results in it

    Args:
        results_directory (Path): Path to the directory where the results will
        be stored
        test_regression_loss (float): Test loss
        test_accuracy (float): Test accuracy
        parameters_table (table): Table of parameters
        model (nn.Module): The network
        regression_loss_values (Tensor): Regression loss values
        accuracy_values (Tensor): Accuracy values
        test_losses_loss (list): Test loss values when removing samples with
        the highest regression loss
        test_losses_uncertainty (list): Test loss values when removing samples
        with the highest uncertainty
    """    
    os.mkdir(results_directory)

    train_losses = regression_loss_values[0]
    val_losses = regression_loss_values[1]

    with open(results_directory / "results.txt", "w") as loss_file:
        loss_file.write(
            f"Test regression loss: {test_regression_loss}\n"
            f"Train regression loss: {train_losses[-1]}\n"
            f"Validation regression loss: {val_losses[-1]}\n"
        )

    with open(results_directory / "parameters_table.txt", "w") as parameters_file:
        parameters_file.write(parameters_table)

    with open(results_directory / "network.txt", "w") as network_file:
        print(model, file=network_file)

    # Create and save the learning curve
    plt.figure()

    plt.plot(train_losses, "b", label="train loss")
    plt.plot(val_losses, "r", label="validation loss")

    plt.legend()
    plt.xlabel("Epoch")
    
    plt.savefig(results_directory / "learning_curve.png")
    
    # Save the model parameters
    torch.save(model.state_dict(),
               results_directory / params.learning.PARAMS_FILE)


# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    
    # Test the functions
    print(parameters_table())
