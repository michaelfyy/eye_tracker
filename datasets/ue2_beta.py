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
    Custom PyTorch Dataset for our synthetic dataset.
    
    Expects lists of image file paths and corresponding JSON file paths.
    The JSON files contain the ground truth in the "ground_truth" field:
      - "iris_center": a tuple of floats (e.g., iris center coordinates)
      - "gaze_vec": a tuple of floats (e.g., gaze vector)
    """
    def __init__(self, x_files, y_files, transform=None):
        self.image_files = x_files
        self.json_files = y_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        json_path = self.json_files[idx]
        
        # Open the image.
        image = torchvision.io.read_image(image_path)
        image = transforms.functional.convert_image_dtype(image, torch.float)
        image = torchvision.tv_tensors.Image(image)
        if self.transform:
            image = self.transform(image)
        
        # Load the JSON file.
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract ground truth values.
        gt = data.get("ground_truth", {})
        iris_center_str = gt.get("iris_center", None)
        gaze_vec_str = gt.get("gaze_vec", None)
        if iris_center_str is None or gaze_vec_str is None:
            raise ValueError(f"Missing ground truth in JSON file: {json_path}")
        
        iris_center = torch.tensor(parse_tuple_string(iris_center_str), dtype=torch.float)
        # If the gaze vector has an extra dimension, we keep only the first three.
        gaze_vec = torch.tensor(parse_tuple_string(gaze_vec_str), dtype=torch.float)[:3]
        
        sample = {
            "image": image,
            "pupil": iris_center,  # ground truth iris center
            "gaze": gaze_vec       # ground truth gaze vector
        }
        return sample

# For quick testing, you can do:
if __name__ == "__main__":
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # Example: Provide lists of file paths directly.
    data_dir = "data/ue2_dataset"
    # List jpg files sorted by numeric filename.
    image_files = sorted(
        [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.jpg')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    json_files = [os.path.join(data_dir, os.path.splitext(os.path.basename(x))[0] + '.json')
                  for x in image_files]
    
    dataset = UE2Dataset(x_files=image_files, y_files=json_files, transform=transform)
    print(f"Total samples: {len(dataset)}")
    sample = dataset[0]
    print("Image tensor shape:", sample["image"].shape)
    print("Iris center:", sample["pupil"])
    print("Gaze vector:", sample["gaze"])