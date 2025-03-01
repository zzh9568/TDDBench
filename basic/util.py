"""This file contains information about the utility functions."""
from ast import List

import numpy as np
import sklearn.metrics as metrics
import torch


def get_optimizer(model: torch.nn.Module, configs: dict):
    """Get the optimizer for the given model

    Args:
        model (torch.nn.Module): The model we want to optimize
        configs (dict): Configurations for the optimizer

    Raises:
        NotImplementedError: Check if the optimizer is implemented.

    Returns:
        optim: Optimizer for the given model
    """
    optimizer = configs.get("optimizer", "SGD")
    learning_rate = configs.get("learning_rate", 0.001)
    weight_decay = configs.get("weight_decay", 0)
    momentum = configs.get("momentum", 0)
    print(f"Load the optimizer {optimizer}: ", end=" ")
    print(f"Learning rate {learning_rate}", end=" ")
    print(f"Weight decay {weight_decay} ")

    if optimizer == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    elif optimizer == "Adam":
        return torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer == "AdamW":
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    else:
        raise NotImplementedError(
            f"Optimizer {optimizer} has not been implemented. Please choose from SGD or Adam"
        )


def get_split(
    all_index: List(int), used_index: List(int), size: int, split_method: str
):
    """Select points based on the splitting methods

    Args:
        all_index (list): All the possible dataset index list
        used_index (list): Index list of used points
        size (int): Size of the points needs to be selected
        split_method (str): Splitting (selection) method

    Raises:
        NotImplementedError: If the splitting the methods isn't implemented
        ValueError: If there aren't enough points to select
    Returns:
        np.ndarray: List of index
    """
    if split_method in "no_overlapping":
        selected_index = np.setdiff1d(all_index, used_index, assume_unique=True)
        if size <= len(selected_index):
            selected_index = np.random.choice(selected_index, size, replace=False)
        else:
            raise ValueError("Not enough remaining data points.")
    elif split_method == "uniform":
        if size <= len(all_index):
            selected_index = np.random.choice(all_index, size, replace=False)
        else:
            raise ValueError("Not enough remaining data points.")
    else:
        raise NotImplementedError(
            f"{split_method} is not implemented. The only supported methods are uniform and no_overlapping."
        )

    return selected_index


def sweep(in_signal, out_signal):
    """Calculate the ROC curve and AUC score."""
    all_t = np.concatenate([in_signal, out_signal])
    all_t = np.append(all_t, [np.max(all_t) + 0.0001, np.min(all_t) - 0.0001])
    all_t.sort()
    tpr = []
    fpr = []
    for threshold in all_t:
        tpr.append(np.sum(in_signal < threshold) / len(in_signal))
        fpr.append(np.sum(out_signal < threshold) / len(out_signal))
    return fpr, tpr, metrics.auc(fpr, tpr)


def get_text_model_path(model_name):
    if model_name == "roberta":
        model_path = f"../huggingface/models/FacebookAI/roberta-base"
    elif model_name == "flan-t5":
        model_path = f"../huggingface/models/google/flan-t5-base"
    elif model_name == "distilbert":
        model_path = f"../huggingface/models/distilbert/distilbert-base-uncased"
    elif model_name == "bert":
        model_path = f"../huggingface/models/google-bert/bert-base-uncased"
    elif model_name == "gpt2":
        model_path = f"../huggingface/models/openai-community/gpt2"
    elif model_name == "longformer":
        model_path = f"../huggingface/models/allenai/longformer-base-4096"
    else:
        raise ValueError(
            f"The {model_name} is not supported for the given dataset."
        )
    return model_path