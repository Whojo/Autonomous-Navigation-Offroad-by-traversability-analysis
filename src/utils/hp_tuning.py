import matplotlib.pyplot as plt
import pandas as pd
from ray import tune


def get_multiple_choice_seach_space(prefix: str, choices: list) -> dict:
    """
    Generate a search space that allows to select multiple choices.

    Note:
        The default implementation of tune.choice() only allows for single
        choice among a list.
        This function, along with get_multiple_choice(), allows to select
        multiple choices among a list.

    Args:
        prefix (str): The prefix name to be used for the search space.
            This is an implementation detail and does not matter much.
            The only thing that matters is to use it to recover the choices
            with get_multiple_choice().
        choices (list): The list of choices.

    Returns:
        dict: A dictionary representing the search space.
    """
    return {
        f"{prefix}_{i}": tune.choice([True, False])
        for i, aug in enumerate(choices)
    }


def get_multiple_choice(config: dict, prefix: str, choices: []) -> []:
    """
    Returns a list of the selected choices of the given configurations.
    The `config` is supposed to be the object passed to the `trainable`
    function of `tune.Tuner`.

    Args:
        config (dict): The configuration.
        prefix (str): The prefix used to identify the choices in the
            configuration.
        choices (list): The list of available choices.

    Returns:
        []: A list of the selected choices.
    """
    return [
        choice
        for i, choice in enumerate(choices)
        if f"{prefix}_{i}" in config and config[f"{prefix}_{i}"]
    ]


def plot_trials_multiple_choice(
    df: pd.DataFrame, column_prefix: str, metric: str, *, x_range: tuple = None
) -> None:
    """
    Plot the distribution of a `metric` for a multiple choice hyperparameter,
    as defined by `get_multiple_choice_seach_space`.

    Args:
        df (pandas.DataFrame): The dataframe containing the trials data.
        metric (str): The metric to plot.
        column_prefix (str): The column prefix that defines the multiple
            choice hyperparameter. It must be the same used in
            `get_multiple_choice_seach_space`.
        x_range (tuple): The range of values for the x-axis.
            By default, it takes the minimum and maximum values encountered
            in the dataframe, but it can be restricted to smaller range for
            better visualization.

    Returns:
        None
    """
    if x_range is None:
        x_range = (df[metric].min(), df[metric].max())

    best = df.iloc[df[metric].argmin()]
    columns = [col for col in df.columns if column_prefix in col]
    if len(columns) == 0:
        raise ValueError(
            f"Cannot find any column with prefix {column_prefix} in the dataframe"
        )

    _, ax_list = plt.subplots(1, len(columns), figsize=(5 * len(columns), 5))
    if len(columns) == 1:
        ax_list = [ax_list]

    max_y = 0
    for col, ax in zip(columns, ax_list):
        ax.set_title(f"{col} (best: {best[col]})")
        for value, group in df.groupby(col):
            bins, *_ = ax.hist(
                group[metric],
                label=value,
                range=x_range,
                bins=20,
                alpha=0.7,
            )
            max_y = max(max_y, bins.max())

        ax.set_xlabel("Validation loss")
        ax.set_ylabel("Number of trials")
        ax.legend()

    # XXX: increase the y-axis limit by 10% to leave a margin
    max_y = int(max_y * 1.1)
    for ax in ax_list:
        ax.set_ylim(0, max_y)
