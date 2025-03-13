import os
import sys
import json
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Ensure parent directory is in the path.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.config import parse_args
from utils.dataloader import load_test_data
from utils.logger import Logger
from metrics.metrics import compute_mse_error, compute_angular_error, compute_cosine_similarity
from utils.visualizer import visualize_predictions
from models import get_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test():
    # Load configuration.
    config = parse_args()
    
    # Create a test run directory.
    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    run_dir = os.path.join(config['logging']['run_dir'], f"test_run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    logger = Logger(run_dir, config)
    
    # Load test data
    test_loader = load_test_data(config)
    
    # Load the model and its weights.
    model = get_model(config).to(device)
    checkpoint_path = config['testing']['checkpoint']  # Must be specified in config file.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    logger.log({"message": "Model loaded for testing.", "checkpoint": checkpoint_path})
    print(f"Testing on device: {device}")
    
    # Initialize metric accumulators.
    total_samples = 0
    total_pupil_mse = 0.0
    total_gaze_angular = 0.0
    total_gaze_cosine = 0.0
    
    for batch in test_loader:
        images = batch['image'].to(device)
        pupil_labels = batch['pupil'].to(device)
        gaze_labels = batch['gaze'].to(device)
        
        with torch.no_grad():
            outputs = model(images)
        
        batch_size = images.size(0)
        total_samples += batch_size
        
        total_pupil_mse += compute_mse_error(outputs['pupil'], pupil_labels) * batch_size
        total_gaze_angular += compute_angular_error(outputs['gaze'], gaze_labels) * batch_size
        total_gaze_cosine += compute_cosine_similarity(outputs['gaze'], gaze_labels) * batch_size
    
    avg_pupil_mse = total_pupil_mse / total_samples
    avg_gaze_angular = total_gaze_angular / total_samples
    avg_gaze_cosine = total_gaze_cosine / total_samples
    
    test_metrics = {
        "test_samples": total_samples,
        "avg_pupil_mse": avg_pupil_mse,
        "avg_gaze_angular_deg": avg_gaze_angular,
        "avg_gaze_cosine": avg_gaze_cosine
    }
    
    logger.log(test_metrics)
    print("Test Metrics:")
    print(json.dumps(test_metrics, indent=4))
    
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f)
    
    # Optionally, generate visualizations.
    visualize_predictions(model, test_loader, run_dir, device, num_samples=5, image_size=tuple(config.get('image_size', [224,224])))
    
if __name__ == "__main__":
    test()
