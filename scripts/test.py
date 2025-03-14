import os
import sys
import time
import json
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torchvision.transforms.v2 as transforms
import torch.nn.functional as F
import argparse

# Ensure parent directory is in the path.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.config import parse_args
from utils.dataloader import load_data
from models import get_model
from utils.logger import Logger
from datasets.transforms import inverse_transform_2d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def denormalize_and_resize(img_tensor, orig_size):
    """
    Given an image tensor (C, H, W) that is normalized and resized to 224x224,
    denormalizes it (assuming ImageNet stats) and then resizes it back to its original size.
    """
    img_np = img_tensor.cpu().detach().numpy()
    # Denormalize using ImageNet statistics.
    mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    img_np = np.transpose(img_np, (1,2,0))
    # orig_size is a tensor (2,) [height, width]
    orig_h = int(orig_size[0].item())
    orig_w = int(orig_size[1].item())
    img_resized = cv2.resize(img_np, (orig_w, orig_h))
    return img_resized

def annotate_predictions(np_img, gt_point, pred_point):
    """
    Draws a green circle for GT and a red circle for prediction.
    np_img should be an RGB image (in [0,1]) at the original resolution.
    """
    img_bgr = cv2.cvtColor((np_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if gt_point is not None:
        gt_xy = (int(gt_point[0]), int(gt_point[1]))
        cv2.circle(img_bgr, gt_xy, 5, (0, 255, 0), -1)
    pred_xy = (int(pred_point[0]), int(pred_point[1]))
    cv2.circle(img_bgr, pred_xy, 5, (0, 0, 255), -1)
    return img_bgr

def main():
    parser = argparse.ArgumentParser(description="Test a trained model on the dataset and visualize predictions.")
    parser.add_argument("--config", type=str, default="ue2.yaml", help="Path to the config file.")
    args = parser.parse_args()
    
    config = parse_args()  # load configuration from file
    
    # If testing checkpoint is "latest", replace with the common checkpoint file.
    if config['testing']['checkpoint'] == "latest":
        common_ckpt = os.path.join(config['logging']['run_dir'], f"latest_best_model_{config['data']['dataset']}.pt")
        if os.path.exists(common_ckpt):
            config['testing']['checkpoint'] = common_ckpt
        else:
            raise FileNotFoundError("Latest checkpoint not found. Please run training first.")
    
    # Create test output directory under runs/tests.
    test_output_dir = os.path.join(config['logging']['run_dir'], "tests", "test_" + datetime.now().strftime('%Y-%m-%d_%H-%M'))
    os.makedirs(test_output_dir, exist_ok=True)
    logger = Logger(test_output_dir, config)
    logger.log({"message": "Starting test on dataset", "dataset": config['data']['dataset']})
    
    # Load test dataset (using validation split).
    dataloaders = load_data(config)
    test_loader = dataloaders['val']
    num_test_samples = len(test_loader.dataset)
    logger.log({"message": f"Loaded test dataset with {num_test_samples} samples."})
    
    # Load model from checkpoint.
    model = get_model(config).to(device)
    checkpoint_path = config['testing']['checkpoint']
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    logger.log({"message": "Model loaded for testing", "checkpoint": checkpoint_path})
    
    all_pixel_errors = []
    visualization_paths = []
    sample_count = 0
    start_time = time.time()
    
    # Process test batches.
    for batch in test_loader:
        images = batch['image'].to(device)           # (B, C, 224, 224)
        pupil_labels = batch['pupil'].to(device)       # (B, 2) in resized coords
        orig_sizes = batch['orig_size']                # (B, 2) original size tensors
        with torch.no_grad():
            outputs = model(images)
        for i in range(images.size(0)):
            pred = outputs['pupil'][i]
            gt = pupil_labels[i]
            orig_size = orig_sizes[i]
            # Inverse-transform predictions and GT to original coordinates.
            pred_orig = inverse_transform_2d(pred, orig_size, target_size=(224,224))
            gt_orig = inverse_transform_2d(gt, orig_size, target_size=(224,224))
            pixel_error = torch.norm(pred_orig - gt_orig, p=2).item()
            all_pixel_errors.append(pixel_error)
            # Visualize up to 10 samples.
            if sample_count < 10:
                # Get the original image by denormalizing and resizing back.
                img_resized = denormalize_and_resize(images[i], orig_size)
                annotated = annotate_predictions(img_resized, gt_orig.cpu().numpy(), pred_orig.cpu().numpy())
                vis_path = os.path.join(test_output_dir, f"sample_{sample_count}.png")
                cv2.imwrite(vis_path, annotated)
                visualization_paths.append(vis_path)
                sample_count += 1
    
    total_time = time.time() - start_time
    avg_error = sum(all_pixel_errors) / len(all_pixel_errors) if all_pixel_errors else None
    
    # Plot per-sample pixel errors.
    plt.figure(figsize=(10,5))
    plt.plot(range(len(all_pixel_errors)), all_pixel_errors, marker='o')
    plt.xlabel("Sample Index")
    plt.ylabel("Pixel Error (in original coordinates)")
    plt.title("Per-sample Pixel Error")
    error_plot_path = os.path.join(test_output_dir, "per_sample_pixel_error.png")
    plt.savefig(error_plot_path)
    plt.close()
    
    results = {
        "num_test_samples": num_test_samples,
        "average_pixel_error": avg_error,
        "per_sample_pixel_errors": all_pixel_errors,
        "inference_time_sec": total_time,
        "average_time_per_sample_sec": total_time / num_test_samples if num_test_samples > 0 else None,
        "visualization_samples": visualization_paths
    }
    results_path = os.path.join(test_output_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.log({"message": "Testing complete", "results_summary": {"num_test_samples": num_test_samples,
                                                                    "average_pixel_error": avg_error,
                                                                    "inference_time_sec": total_time}})
    print("Testing complete. Average pixel error: {:.3f}".format(avg_error) if avg_error is not None else "No valid test samples.")

if __name__ == "__main__":
    main()
