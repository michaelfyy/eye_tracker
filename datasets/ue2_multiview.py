import os
import json
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset
from PIL import Image

def parse_tuple_string(s):
    """
    Parses a string of the form "(v1, v2, ...)" into a tuple of floats.
    
    Example:
        s = "(117.0342, 336.1442, 4.1068)"
        returns: (117.0342, 336.1442, 4.1068)
    """
    s = s.strip().strip("()")
    return tuple(float(x.strip()) for x in s.split(','))

class UE2MultiviewDataset(Dataset):
    """
    Custom PyTorch Dataset for multiview eye tracking prediction.
    
    For each sample (frame), this dataset:
      - Loads the corresponding JSON file (which contains the ground truth for all 4 cameras).
      - Loads the 4 camera images for that frame (expected naming: {frame_num}_cam_{cam_id}.jpg).
      - Computes the 2D pupil center for each camera as the centroid of the iris 2D landmarks,
        adjusted to the resized image coordinate system.
      - Extracts the 3D ground truth for each camera by concatenating the iris center and gaze vector,
        yielding a 6D vector [pupil_x, pupil_y, pupil_z, gaze_x, gaze_y, gaze_z].
    
    The returned sample is a dictionary with:
      - "images": a tensor of shape (views, channels, height, width)
      - "orig_sizes": a tensor of original sizes for each view (shape: views x 2)
      - Ground truth keys for each camera: e.g. "cam_1_2d", "cam_1_3d", etc.
    """
    def __init__(self, json_files, img_dir, transform=None, target_size=(224, 224)):
        """
        Args:
            json_files (list): List of file paths to the JSON ground truth files.
            img_dir (str): Directory containing the image files.
            transform (callable, optional): Transformation to be applied on each image.
            target_size (tuple): Desired output size (height, width) for the images.
        """
        self.json_files = json_files
        self.img_dir = img_dir
        self.transform = transform
        self.target_size = target_size
        # List of camera identifiers in the expected order.
        self.cam_ids = ["cam_1", "cam_2", "cam_3", "cam_4"]

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        # Load the JSON file containing ground truth for all cameras for this frame.
        json_path = self.json_files[idx]
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract frame number from JSON filename (e.g. "1.json" -> "1")
        frame_num = os.path.splitext(os.path.basename(json_path))[0]
        
        sample = {}
        images = []
        orig_sizes = []
        
        # Loop through each camera view
        for cam in self.cam_ids:
            # Construct the image filename; expected format: "{frame_num}_{cam}.jpg"
            img_filename = f"{frame_num}_{cam}.jpg"
            img_path = os.path.join(self.img_dir, img_filename)
            
            # Load and process the image.
            image = torchvision.io.read_image(img_path)  # shape: (C, H, W)
            image = transforms.functional.convert_image_dtype(image, torch.float)
            orig_h, orig_w = image.shape[1], image.shape[2]
            orig_sizes.append(torch.tensor([orig_h, orig_w], dtype=torch.float32))
            
            # Compute scaling factors to map coordinates to the resized image.
            target_h, target_w = self.target_size
            scale_w = target_w / orig_w
            scale_h = target_h / orig_h
            
            if self.transform:
                image = self.transform(image)
            images.append(image)
            
            # Retrieve ground truth for this camera.
            cameras = data.get("cameras", None)
            if cameras is None:
                raise ValueError(f"Missing 'cameras' field in JSON file: {json_path}")
            if cam not in cameras:
                raise ValueError(f"Camera {cam} not found in JSON file: {json_path}")
            cam_data = cameras[cam]
            
            # ----- Process 2D Ground Truth -----
            iris_landmarks = cam_data.get("iris_2d", None)
            if iris_landmarks is None:
                raise ValueError(f"Missing 'iris_2d' for {cam} in JSON file: {json_path}")
            # Parse the iris landmarks strings into tuples of floats.
            landmarks = [parse_tuple_string(s) for s in iris_landmarks]
            landmarks = torch.tensor(landmarks, dtype=torch.float)  # shape (N, 3)
            # Keep only the x and y coordinates.
            landmarks = landmarks[:, :2]
            # Flip the y-coordinate. (Assuming original image coordinate system where y=0 is at the top)
            landmarks[:, 1] = orig_h - landmarks[:, 1]
            # Compute the pupil center as the centroid of the iris landmarks.
            pupil_2d = landmarks.mean(dim=0)
            # Adjust the pupil center according to the resizing.
            pupil_2d = pupil_2d * torch.tensor([scale_w, scale_h], dtype=torch.float)
            sample[f"{cam}_2d"] = pupil_2d
            
            # ----- Process 3D Ground Truth -----
            gt = cam_data.get("ground_truth", None)
            if gt is None:
                raise ValueError(f"Missing 'ground_truth' for {cam} in JSON file: {json_path}")
            iris_center_str = gt.get("iris_center", None)
            gaze_vector_str = gt.get("gaze_vector", None)
            if iris_center_str is None or gaze_vector_str is None:
                raise ValueError(f"Missing 'iris_center' or 'gaze_vector' for {cam} in JSON file: {json_path}")
            # Parse the 3D values.
            iris_center = torch.tensor(parse_tuple_string(iris_center_str), dtype=torch.float)
            gaze_vector = torch.tensor(parse_tuple_string(gaze_vector_str), dtype=torch.float)
            # Concatenate to form a 6D vector: [pupil (iris_center) and gaze_vector].
            pupil_3d = torch.cat([iris_center, gaze_vector])
            sample[f"{cam}_3d"] = pupil_3d
        
        # Stack images along a new dimension (views) to form the input tensor.
        images = torch.stack(images, dim=0)  # shape: (views, channels, height, width)
        sample["images"] = images
        sample["orig_sizes"] = torch.stack(orig_sizes, dim=0)  # shape: (views, 2)
        
        return sample