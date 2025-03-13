import torchvision.transforms.v2 as transforms
import torch

def get_transforms():
    transforms_dict = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.ToImage(),
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.ToImage(),
        ]),
    }
    return transforms_dict

def inverse_transform_2d(pupil_center, original_size, target_size=(224, 224)):
    """
    Convert a 2D coordinate from the resized image space back to the original image coordinates.
    
    Args:
        pupil_center (Tensor): 2D coordinate (x, y) in the resized image.
        original_size (tuple): (height, width) of the original image.
        target_size (tuple): (height, width) of the resized image (default: (224, 224)).
        
    Returns:
        Tensor: The 2D coordinate in the original image space.
    """
    orig_h, orig_w = original_size
    target_h, target_w = target_size
    scale_w = orig_w / target_w
    scale_h = orig_h / target_h
    if not isinstance(pupil_center, torch.Tensor):
        pupil_center = torch.tensor(pupil_center, dtype=torch.float32)
    return pupil_center * torch.tensor([scale_w, scale_h], device=pupil_center.device)
