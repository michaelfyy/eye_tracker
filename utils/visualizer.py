import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_predictions(model, dataloader, run_dir, device, num_samples=5, image_size=(224, 224)):
    """
    Visualizes predictions on a few samples from the test dataloader.
    For each sample, overlays the predicted pupil location (red dot) and ground-truth (green dot).
    
    Args:
        model: The trained model.
        dataloader: DataLoader for the test set.
        run_dir: Directory where visualizations are saved.
        device: torch.device.
        num_samples: Number of samples to visualize.
        image_size: Tuple (width, height) of the resized images.
    """
    model.eval()
    samples = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            pupil_labels = batch['pupil'].to(device)
            outputs = model(images)
            for i in range(images.size(0)):
                samples.append({
                    'image': images[i].cpu(),
                    'pupil_gt': pupil_labels[i].cpu(),
                    'pupil_pred': outputs['pupil'][i].cpu()
                })
                if len(samples) >= num_samples:
                    break
            if len(samples) >= num_samples:
                break

    for idx, sample in enumerate(samples):
        # Convert image tensor (C, H, W) to numpy (H, W, C)
        img = sample['image']
        np_img = img.permute(1, 2, 0).numpy()
        # Assume image pixels are in [0,1] after preprocessing.
        width, height = image_size
        # Denormalize pupil coordinates (assumed normalized to the resized image dimensions)
        pupil_gt = sample['pupil_gt'].numpy() * np.array([width, height])
        pupil_pred = sample['pupil_pred'].numpy() * np.array([width, height])
        
        plt.figure(figsize=(8, 8))
        plt.imshow(np_img)
        plt.scatter(pupil_gt[0], pupil_gt[1], color='green', label='GT Pupil')
        plt.scatter(pupil_pred[0], pupil_pred[1], color='red', label='Pred Pupil')
        plt.legend()
        plt.title(f"Sample {idx+1}")
        plt.savefig(f"{run_dir}/vis_sample_{idx+1}.png")
        plt.close()
