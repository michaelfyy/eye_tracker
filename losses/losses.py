import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def cosine_similarity_loss(pred, target):
    """
    Computes a loss based on cosine similarity using PyTorch's built-in implementation.
    Both pred and target should be non-zero vectors.
    The loss is defined as 1 - cosine_similarity.
    """
    cosine_sim = F.cosine_similarity(pred, target, dim=1)  # PyTorch's built-in cosine similarity
    loss = 1.0 - cosine_sim  # the higher the similarity, the lower the loss
    return torch.mean(loss)

def angular_loss(pred, target):
    """
    Computes the mean angular error (in radians) between predicted and target vectors.
    This loss works on the normalized gaze vectors.
    """
    pred_norm = F.normalize(pred, dim=1)
    target_norm = F.normalize(target, dim=1)
    # Compute cosine similarity and clamp for numerical stability.
    cosine_sim = torch.sum(pred_norm * target_norm, dim=1)
    cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
    angles = torch.acos(cosine_sim)
    return torch.mean(angles)

# Registry for losses per task.
loss_registry = {
    'pupil': {
        'mse': nn.MSELoss(),
        'smooth_l1': nn.SmoothL1Loss()  # Directly using PyTorch's loss module.
    },
    'gaze': {
        'mse': nn.MSELoss(),
        'cosine': cosine_similarity_loss,
        'angular': angular_loss
    }
}

def get_loss(loss_type: str, task: str):
    """
    Retrieve the loss function for the given task.
    
    Args:
        loss_type (str): The type of loss (e.g., 'mse', 'smooth_l1', 'cosine', or 'angular').
        task (str): The task name ('pupil' or 'gaze').
        
    Returns:
        A callable loss function.
    """
    try:
        # For losses implemented as functions, return them directly.
        loss_fn = loss_registry[task][loss_type]
        # If the loss is an instance of nn.Module, wrap it so it can be called.
        if isinstance(loss_fn, nn.Module):
            return lambda pred, target: loss_fn(pred, target)
        return loss_fn
    except KeyError:
        available = list(loss_registry.get(task, {}).keys())
        raise KeyError(f"Loss type '{loss_type}' not found for task '{task}'. Available options: {available}")
