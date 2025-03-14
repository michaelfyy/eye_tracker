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

class UE2Dataset(Dataset):
    """
    Custom PyTorch Dataset for the UnityEyes2 dataset with multiple camera views.
    
    Expects lists of image file paths and corresponding JSON file paths.
    The JSON file for a given frame contains the ground truth for all 4 cameras.
    For each camera, the JSON now contains an "iris_2d" field (a list of 3D iris landmarks).
    The pupil center is computed as the centroid of the iris landmarks (using only the first two coordinates)
    and then transformed to the resized (224x224) image coordinate system.
    """
    def __init__(self, x_files, y_files, transform=None, target_size=(224, 224)):
        self.image_files = x_files
        self.json_files = y_files
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        # Extract camera id from image filename.
        # Expected filename format: "{frame_number}_cam_{cam_number}.jpg"
        base = os.path.splitext(os.path.basename(image_path))[0]  # e.g. "1_cam_1"
        parts = base.split('_')
        if len(parts) < 3:
            raise ValueError(f"Unexpected image filename format: {image_path}")
        # Construct camera key, e.g. "cam_1"
        camera_id = f"{parts[1]}_{parts[2]}"
        
        # Read the image and record original size.
        image = torchvision.io.read_image(image_path)
        image = transforms.functional.convert_image_dtype(image, torch.float)
        orig_h, orig_w = image.shape[1], image.shape[2]
        target_h, target_w = self.target_size
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h

        # Apply transformations (which include the resizing and normalization).
        if self.transform:
            image = self.transform(image)
        
        # Load the corresponding JSON file.
        json_path = self.json_files[idx]
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Access the specific camera's ground truth.
        cameras = data.get("cameras", None)
        if cameras is None:
            raise ValueError(f"Missing 'cameras' field in JSON file: {json_path}")
        if camera_id not in cameras:
            raise ValueError(f"Camera id {camera_id} not found in JSON file: {json_path}")
        cam_data = cameras.get(camera_id, {})
        
        # Load iris landmarks from the "iris_2d" field.
        iris_landmarks = cam_data.get("iris_2d", None)
        if iris_landmarks is None:
            raise ValueError(f"Missing 'iris_2d' in JSON file: {json_path} for camera {camera_id}")

        # Parse the iris_landmarks strings into tuples of floats.
        iris_landmarks = [parse_tuple_string(s) for s in iris_landmarks]
        # Convert to tensor (assumes iris_landmarks is a list of [x, y, z] lists).
        iris_landmarks = torch.tensor(iris_landmarks, dtype=torch.float)  # shape (N, 3)
        # Truncate the last coordinate, keeping only x and y.
        iris_landmarks = iris_landmarks[:, :2]
        iris_landmarks[:, 1] = 480 - iris_landmarks[:, 1] # Flip y-coordinate to match conventions
        # Compute the pupil center as the centroid of the iris landmarks.
        pupil_center = iris_landmarks.mean(dim=0)
        # Adjust pupil center coordinates according to the resize transformation.
        pupil_center = pupil_center * torch.tensor([scale_w, scale_h], dtype=torch.float)
        
        sample = {
            "image": image,
            "pupil": pupil_center,
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.float32)
        }
        return sample

# For quick testing, you can do:
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToImage(),
    ])
    data_dir = "C:/Users/michaelfeng/Documents/IML/data/10k_multiviewEER/imgs"
    # List image files that follow the new naming convention.
    image_files = []
    for f in os.listdir(data_dir):
        if f.lower().endswith('.jpg'):
            image_files.append(os.path.join(data_dir, f))
    # Sort images by frame number (extracted from filename).
    image_files = sorted(image_files, key=lambda x: int(os.path.basename(x).split('_')[0]))
    # Construct JSON file list using the frame number.
    json_files = [os.path.join(data_dir, f"{os.path.basename(x).split('_')[0]}.json") for x in image_files]
    
    dataset = UE2Dataset(x_files=image_files, y_files=json_files, transform=transform)
    print(f"Total samples: {len(dataset)}")
    sample = dataset[0]
    print("Image tensor shape:", sample["image"].shape)
    print("Pupil center (resized coords):", sample["pupil"])
    print("Original image size:", sample["orig_size"])
