#! Currently kinda jank, gaze loss is never used because gaze gt is just the endpoint, so it uses same as pupil

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
    cosine_sim = F.cosine_similarity(pred, target, dim=1)
    loss = 1.0 - cosine_sim
    return torch.mean(loss)

def angular_loss(pred, target):
    """
    Computes the mean angular error (in radians) between predicted and target vectors.
    This loss works on the normalized gaze vectors.
    """
    pred_norm = F.normalize(pred, dim=1)
    target_norm = F.normalize(target, dim=1)
    cosine_sim = torch.sum(pred_norm * target_norm, dim=1)
    cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
    angles = torch.acos(cosine_sim)
    return torch.mean(angles)

# Updated registry for losses per task.
# For the 2D pupil outputs and the 3D (pupil+gaze) outputs in the multiview setup,
# we add smooth L1 loss options while preserving the old MSE/L1 options.
loss_registry = {
    'pupil': {
        'mse': nn.MSELoss(),
        'l1': nn.L1Loss(),
        'smooth_l1': nn.SmoothL1Loss()
    },
    'gaze': {
        'mse': nn.MSELoss(),
        'l1': nn.L1Loss(),
        'smooth_l1': nn.SmoothL1Loss(),
        'cosine': cosine_similarity_loss,
        'angular': angular_loss
    }
}

def get_loss(loss_conf, task: str):
    """
    Retrieve the loss function for the given task.
    
    For multiview outputs, task keys may include '_2d' or '_3d'.
    We map these to the base tasks:
      - Keys ending with '_2d' are assumed to be "pupil" regression (2D landmark centroid).
      - Keys ending with '_3d' are assumed to be "gaze" regression (6D concatenated pupil/gaze vectors).
    
    Args:
        loss_conf: Either a string representing the loss type or a dictionary with a "type" field.
        task (str): The task name or key.
        
    Returns:
        A callable loss function.
    """
    # Map multiview keys to base task names if necessary.
    if task not in loss_registry:
        if task.endswith('_2d'):
            base_task = 'pupil'
        elif task.endswith('_3d'):
            base_task = 'pupil'
        else:
            base_task = task
    else:
        base_task = task

    # Extract loss type from configuration.
    if isinstance(loss_conf, dict):
        loss_type = loss_conf.get('type', 'mse')
    else:
        loss_type = loss_conf

    try:
        loss_fn = loss_registry[base_task][loss_type]
        # Wrap nn.Module instances for a uniform callable interface.
        if isinstance(loss_fn, nn.Module):
            return lambda pred, target: loss_fn(pred, target)
        return loss_fn
    except KeyError:
        available = list(loss_registry.get(base_task, {}).keys())
        raise KeyError(f"Loss type '{loss_type}' not found for task '{base_task}'. Available options: {available}")
