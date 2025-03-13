import os
import pickle
import torch
import torchvision
import numpy as np
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset

class SynthesEyesDataset(Dataset):
    """SynthesEyes Dataset"""

    def __init__(self, x_files, y_files, transform=None, target_size=(224, 224)):
        """
        Args:
            x_files: List of image file paths.
            y_files: List of corresponding pickle file paths with ground truth.
            transform (callable, optional): Transform to be applied on an image.
        """
        self.img_files = x_files
        self.pkl_files = y_files
        self.transform = transform
        self.target_size = target_size
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image and convert to float tensor.
        img_path = self.img_files[idx]
        image = torchvision.io.read_image(img_path)[:3, ...]  # remove alpha channel if present
        image = transforms.functional.convert_image_dtype(image, torch.float)
        
        # Record original image size (height, width).
        orig_h, orig_w = image.shape[1], image.shape[2]
        target_h, target_w = self.target_size
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h
        
        # Load ground truth data.
        pkl_path = self.pkl_files[idx]
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Get the pupil center as the mean of pupil landmarks (raw coordinates).
        pupil_ldmks_2d = np.array(data['ldmks']['ldmks_pupil_2d'])
        raw_pupil_center = np.mean(pupil_ldmks_2d, axis=0)  # [x, y] in original image coordinates
        # Adjust pupil center to resized image coordinates.
        pupil_center = torch.tensor(
            [raw_pupil_center[0] * scale_w, raw_pupil_center[1] * scale_h],
            dtype=torch.float32
        )

        # Apply image transformation.
        if self.transform:
            image = self.transform(image)

        # Optionally include the original image size for later inverse transformations.
        sample = {'image': image, 'pupil': pupil_center, 'orig_size': torch.tensor([orig_h, orig_w], dtype=torch.float32)}
        return sample