import os
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.v2 as transforms
from sklearn.model_selection import train_test_split

from datasets.dummy import DummyDataset
from datasets.syntheseyes import SynthesEyesDataset
from datasets.ue2 import UE2Dataset
from datasets.transforms import get_transforms

def load_data(config):
    transform = get_transforms()

    if config['data']['dataset'] == "dummy":
        train_dataset = DummyDataset()
        val_dataset = DummyDataset()
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True), 
            'val': DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
        }
        return dataloaders

    elif config['data']['dataset'] == "syntheseyes":
        img_files = []
        pkl_files = []
        for subdir, _, files in os.walk(config['data']['data_dir']):
            for file in files:
                if file.endswith('.png'):
                    img_path = os.path.join(subdir, file)
                    pkl_path = img_path.replace('.png', '.pkl')
                    if os.path.exists(pkl_path):  # Ensure corresponding pickle file exists
                        img_files.append(img_path)
                        pkl_files.append(pkl_path)
        x_train, x_val, y_train, y_val = train_test_split(img_files, pkl_files, train_size=config['data']['train_split'], shuffle=True)
        train_dataset = SynthesEyesDataset(x_files=x_train, y_files=y_train, transform=transform['train'])
        val_dataset = SynthesEyesDataset(x_files=x_val, y_files=y_val, transform=transform['test'])
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True), 
            'val': DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
        }
        return dataloaders

    elif config['data']['dataset'] == "ue2":
        # Combined approach (existing)
        data_dir = config['data']['data_dir']
        image_files = []
        for f in os.listdir(data_dir):
            if f.lower().endswith('.jpg'):
                image_files.append(os.path.join(data_dir, f))
        image_files = sorted(image_files, key=lambda x: int(os.path.basename(x).split('_')[0]))
        json_files = [os.path.join(data_dir, f"{os.path.basename(x).split('_')[0]}.json") for x in image_files]
        for jf in json_files:
            if not os.path.exists(jf):
                raise FileNotFoundError(f"Expected JSON file {jf} not found in {data_dir}")
        x_train, x_val, y_train, y_val = train_test_split(image_files, json_files, train_size=config['data']['train_split'], shuffle=True)
        train_dataset = UE2Dataset(x_files=x_train, y_files=y_train, transform=transform['train'])
        val_dataset = UE2Dataset(x_files=x_val, y_files=y_val, transform=transform['test'])
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True),
            'val': DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
        }
        return dataloaders

    elif config['data']['dataset'] == "ue2_separate":
        # New branch: group images by camera id
        data_dir = config['data']['data_dir']
        image_files = []
        for f in os.listdir(data_dir):
            if f.lower().endswith('.jpg'):
                image_files.append(os.path.join(data_dir, f))
        image_files = sorted(image_files, key=lambda x: int(os.path.basename(x).split('_')[0]))
        grouped_images = {}
        grouped_jsons = {}
        for img in image_files:
            base = os.path.splitext(os.path.basename(img))[0]  # e.g. "123_cam_1"
            parts = base.split('_')
            if len(parts) < 3:
                continue
            camera_id = f"{parts[1]}_{parts[2]}"  # e.g. "cam_1"
            if camera_id not in grouped_images:
                grouped_images[camera_id] = []
                grouped_jsons[camera_id] = []
            grouped_images[camera_id].append(img)
            json_path = os.path.join(data_dir, f"{os.path.basename(img).split('_')[0]}.json")
            grouped_jsons[camera_id].append(json_path)
        dataloaders_dict = {}
        for cam, imgs in grouped_images.items():
            jsons = grouped_jsons[cam]
            x_train, x_val, y_train, y_val = train_test_split(imgs, jsons, train_size=config['data']['train_split'], shuffle=True)
            train_dataset = UE2Dataset(x_files=x_train, y_files=y_train, transform=transform['train'])
            val_dataset = UE2Dataset(x_files=x_val, y_files=y_val, transform=transform['test'])
            dataloaders_dict[cam] = {
                'train': DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True),
                'val': DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
            }
        return dataloaders_dict

    else:
        raise ValueError(f"Invalid dataset: {config['data']['dataset']}")

def load_dummy_data(config):
    if config['data']['dataset'] == "dummy":
        dataset = DummyDataset()
    else:
        raise ValueError(f"Invalid dataset: {config['data']['dataset']}")
    from torch.utils.data import DataLoader
    test_loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False)
    return test_loader