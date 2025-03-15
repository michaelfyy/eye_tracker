import os
import json
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset

def parse_tuple_string(s):
    """
    Parses a string of the form "(v1, v2, ...)" into a tuple of floats.
    Example:
        s = "(117.0342, 336.1442, 4.1068)"
        returns: (117.0342, 336.1442, 4.1068)
    """
    s = s.strip().strip("()")
    return tuple(float(x.strip()) for x in s.split(','))

class UE1Dataset(Dataset):
    """
    Custom PyTorch Dataset for the UnityEyes1 dataset.
    
    Expects lists of image file paths and corresponding JSON file paths.
    The ground truth is extracted as the centroid of the iris_2d landmarks.
    The pupil center is computed as the centroid of the iris landmarks (using only the first two coordinates),
    adjusted for resizing, and the original image size is returned.
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
        # Read the image and record original size.
        image = torchvision.io.read_image(image_path)
        image = transforms.functional.convert_image_dtype(image, torch.float)
        orig_h, orig_w = image.shape[1], image.shape[2]
        target_h, target_w = self.target_size
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h

        # Apply transformations (including resizing and normalization).
        if self.transform:
            image = self.transform(image)

        # Load the corresponding JSON file.
        json_path = self.json_files[idx]
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Load iris landmarks from the "iris_2d" field.
        iris_landmarks = data.get("iris_2d", None)
        if iris_landmarks is None:
            raise ValueError(f"Missing 'iris_2d' in JSON file: {json_path}")

        # Parse the iris_landmarks strings into tuples of floats.
        iris_landmarks = [parse_tuple_string(s) for s in iris_landmarks]
        iris_landmarks = torch.tensor(iris_landmarks, dtype=torch.float)  # shape (N, 3)
        # Use only the first two coordinates (x, y)
        iris_landmarks = iris_landmarks[:, :2]
        # Flip y-coordinate to match conventions (assuming an original image height of 480)
        iris_landmarks[:, 1] = 480 - iris_landmarks[:, 1]
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

if __name__ == "__main__":
    # Quick test of the dataset.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToImage(),
    ])
    data_dir = "path/to/your/ue1/data"  # Update this path accordingly.
    # List image files (assumed to be named as {frame_num}.jpg)
    image_files = []
    for f in os.listdir(data_dir):
        if f.lower().endswith('.jpg'):
            image_files.append(os.path.join(data_dir, f))
    # Sort images by frame number.
    image_files = sorted(image_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    # Construct JSON file list using the frame number.
    json_files = [os.path.join(data_dir, f"{os.path.splitext(os.path.basename(x))[0]}.json") for x in image_files]
    
    dataset = UE1Dataset(x_files=image_files, y_files=json_files, transform=transform)
    print(f"Total samples: {len(dataset)}")
    sample = dataset[0]
    print("Image tensor shape:", sample["image"].shape)
    print("Pupil center (resized coords):", sample["pupil"])
    print("Original image size:", sample["orig_size"])
