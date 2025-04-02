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
import glob

# Ensure parent directory is in the path.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.config import parse_args
from models import get_model
from utils.logger import Logger
from datasets.transforms import inverse_transform_2d

# --- Helper Functions ---

def preprocess_frame(frame):
    """
    Preprocess a single RGB frame:
      - Convert to tensor and float type.
      - Permute dimensions to (C, H, W).
      - Resize to (224, 224), normalize, and convert to image tensor.
    Returns:
      processed_frame: Tensor of shape (C, 224, 224)
      orig_size: Tuple (original_height, original_width)
    """
    orig_size = (frame.shape[0], frame.shape[1])
    frame_tensor = torch.tensor(frame)
    frame_tensor = transforms.functional.convert_image_dtype(frame_tensor, torch.float)
    # Convert from (H, W, C) to (C, H, W)
    frame_tensor = frame_tensor.permute(2, 0, 1)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToImage(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    processed_frame = preprocess(frame_tensor)
    return processed_frame, orig_size

def annotate_frame(orig_frame, gt_point, pred_point):
    """
    Annotate the frame with ground truth and predicted 2D points.
    If gt_point is None, only prediction is drawn.
    """
    frame_bgr = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2BGR)
    if gt_point is not None:
        gt_point_int = (int(gt_point[0]), int(gt_point[1]))
        cv2.circle(frame_bgr, gt_point_int, 5, (0, 255, 0), -1)
        cv2.putText(frame_bgr, "GT", (gt_point_int[0] + 10, gt_point_int[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    pred_point_int = (int(pred_point[0]), int(pred_point[1]))
    cv2.circle(frame_bgr, pred_point_int, 5, (0, 0, 255), -1)
    cv2.putText(frame_bgr, "Pred", (pred_point_int[0] + 10, pred_point_int[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame_bgr

def extract_multiview_frames(video_paths):
    """
    Given a dictionary mapping camera id (e.g., "cam_1") to video file paths,
    extract and synchronize frames across all videos.
    Returns a dict mapping camera id to list of frames.
    """
    caps = {}
    frames_dict = {cam: [] for cam in video_paths}
    for cam, path in video_paths.items():
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {path}")
        caps[cam] = cap

    while True:
        current_frames = {}
        ret_all = True
        for cam, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                ret_all = False
                break
            current_frames[cam] = frame
        if not ret_all:
            break
        for cam, frame in current_frames.items():
            frames_dict[cam].append(frame)
    for cap in caps.values():
        cap.release()
    return frames_dict

# --- Main Prediction Function ---

def predict_and_evaluate():
    config = parse_args()
    logger = Logger(os.path.join(config['logging']['run_dir'], "predictions"), config)

    # Determine checkpoint: if "latest", load common checkpoint.
    if config['testing']['checkpoint'] == "latest":
        common_ckpt = os.path.join(config['logging']['run_dir'], f"latest_best_model_{config['data']['dataset']}.pt")
        if os.path.exists(common_ckpt):
            config['testing']['checkpoint'] = common_ckpt
        else:
            raise FileNotFoundError("Latest checkpoint not found. Please run training first.")

    # Assume video files for multiview are in config['data']['video_root']
    video_root = config['data']['video_root']
    video_files = glob.glob(os.path.join(video_root, "*.mp4"))
    # Map videos to camera ids based on filename, e.g., "e1.mp4" -> "cam_1"
    video_paths = {}
    for vf in video_files:
        base = os.path.splitext(os.path.basename(vf))[0]
        if base.startswith("e") and len(base) > 1:
            num = base[1:]
            cam = f"cam_{num}"
            video_paths[cam] = vf
    if len(video_paths) < 4:
        raise ValueError("Not all camera videos found. Expected videos for cam_1, cam_2, cam_3, and cam_4.")

    # Create predictions output directory.
    predictions_output_root = os.path.join(config['logging']['run_dir'], "predictions", datetime.now().strftime('%Y-%m-%d_%H-%M'))
    os.makedirs(predictions_output_root, exist_ok=True)
    logger.log({"message": "Starting multiview prediction.", "video_paths": video_paths})

    # Extract synchronized frames for each camera.
    frames_dict = extract_multiview_frames(video_paths)
    # Determine number of frames as the minimum across cameras.
    num_frames = min(len(frames) for frames in frames_dict.values())
    logger.log({"message": f"Extracted {num_frames} synchronized frames per camera."})

    # Prepare multiview inputs and record original sizes for each view.
    multiview_samples = []  # List of tensors, each of shape (views, channels, 224, 224)
    orig_sizes_per_view = {cam: [] for cam in video_paths}  # Per-camera list of original sizes
    # Also keep the original frames for annotation.
    orig_frames_per_view = {cam: frames_dict[cam][:num_frames] for cam in video_paths}

    # Sort cameras to have a consistent view order: cam_1, cam_2, cam_3, cam_4.
    sorted_cams = sorted(video_paths.keys())
    for i in range(num_frames):
        sample_views = []
        for cam in sorted_cams:
            frame = frames_dict[cam][i]
            proc_frame, orig_size = preprocess_frame(frame)
            sample_views.append(proc_frame)
            orig_sizes_per_view[cam].append(orig_size)
        sample_tensor = torch.stack(sample_views, dim=0)
        multiview_samples.append(sample_tensor)
    # Create batch tensor.
    inputs = torch.stack(multiview_samples, dim=0)  # Shape: (N, views, C, 224, 224)
    inputs = inputs.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    logger.log({"message": "Prepared multiview input tensor.", "input_shape": list(inputs.shape)})

    # Load the multiview model.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(config).to(device)
    checkpoint_path = config['testing']['checkpoint']
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    logger.log({"message": "Model loaded for multiview prediction.", "checkpoint": checkpoint_path})

    # Run inference.
    start_time = time.time()
    with torch.no_grad():
        predictions = model(inputs)
    total_inference_time = time.time() - start_time
    avg_time_per_frame = total_inference_time / num_frames
    logger.log({"message": "Inference complete.",
                "total_inference_time_sec": total_inference_time,
                "avg_time_per_frame_sec": avg_time_per_frame})

    # Expected prediction keys: "cam_1_2d", "cam_1_3d", ..., "cam_4_2d", "cam_4_3d"
    output_keys = [f"{cam}_2d" for cam in sorted_cams] + [f"{cam}_3d" for cam in sorted_cams]

    # --- Process 2D predictions for visualization ---
    # For each camera, annotate the original frame with the predicted 2D point.
    annotated_frames = {cam: [] for cam in sorted_cams}
    # For evaluation metrics (e.g., pixel error), if annotations exist, they could be loaded here.
    # For now, we assume no ground truth annotations are available.
    per_frame_errors = {cam: [] for cam in sorted_cams}

    for idx in range(num_frames):
        for cam in sorted_cams:
            # Get predicted 2D output for this camera and frame.
            pred_2d = predictions[f"{cam}_2d"][idx].cpu()
            # Convert from resized (224,224) space back to original coordinates.
            orig_size = orig_sizes_per_view[cam][idx]
            pred_orig = inverse_transform_2d(pred_2d, orig_size, target_size=(224, 224))
            pred_orig_np = pred_orig.cpu().numpy()
            # If ground truth annotation exists for this frame, use it; otherwise, set to None.
            gt_point = None
            # (Insert code to load/parse ground truth if available.)
            # Annotate the original frame.
            orig_frame = orig_frames_per_view[cam][idx]
            annotated = annotate_frame(orig_frame, gt_point, pred_orig_np)
            annotated_frames[cam].append(annotated)
            # Compute pixel error if ground truth is available.
            if gt_point is not None:
                error = np.linalg.norm(np.array(pred_orig_np) - np.array(gt_point))
                per_frame_errors[cam].append(error)
            else:
                per_frame_errors[cam].append(None)

    # Save annotated videos and generate evaluation plots for each camera.
    for cam in sorted_cams:
        cam_video_path = os.path.join(predictions_output_root, f"{cam}_annotated.mp4")
        if len(annotated_frames[cam]) == 0:
            continue
        height, width, _ = annotated_frames[cam][0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(cam_video_path, fourcc, config['testing'].get('frame_rate', 30), (width, height))
        for frame in annotated_frames[cam]:
            out_video.write(frame)
        out_video.release()
        logger.log({"message": f"Annotated video saved for {cam}.", "video_path": cam_video_path})

        # Generate per-frame pixel error plot if errors are available.
        valid_errors = [e for e in per_frame_errors[cam] if e is not None]
        if valid_errors:
            plt.figure(figsize=(10,5))
            plt.plot(range(num_frames), [e if e is not None else 0 for e in per_frame_errors[cam]], marker='o')
            plt.xlabel("Frame number")
            plt.ylabel("Pixel error")
            plt.title(f"Per-frame Pixel Error for {cam}")
            plt.grid(True)
            error_plot_path = os.path.join(predictions_output_root, f"{cam}_per_frame_pixel_error.png")
            plt.savefig(error_plot_path)
            plt.close()
            logger.log({"message": f"Pixel error plot saved for {cam}.", "plot_path": error_plot_path})
            # Also save histogram.
            plt.figure(figsize=(8,5))
            plt.hist(valid_errors, bins=20, edgecolor='black')
            plt.xlabel("Pixel error")
            plt.ylabel("Frequency")
            plt.title(f"Pixel Error Histogram for {cam}")
            hist_path = os.path.join(predictions_output_root, f"{cam}_pixel_error_histogram.png")
            plt.savefig(hist_path)
            plt.close()
            logger.log({"message": f"Pixel error histogram saved for {cam}.", "histogram_path": hist_path})

    # --- Process 3D predictions: Save raw outputs ---
    predictions_3d = {}
    for cam in sorted_cams:
        key = f"{cam}_3d"
        # Convert predictions for this key to a list of lists.
        predictions_3d[cam] = predictions[key].cpu().tolist()
    predictions_3d_path = os.path.join(predictions_output_root, "multiview_3d_predictions.json")
    with open(predictions_3d_path, "w") as f:
        json.dump(predictions_3d, f, indent=4)
    logger.log({"message": "3D predictions saved.", "predictions_3d_path": predictions_3d_path})

    # Save overall evaluation results.
    evaluation_results = {
        "total_frames": num_frames,
        "total_inference_time_sec": total_inference_time,
        "avg_time_per_frame_sec": avg_time_per_frame,
        # Optionally, add aggregated 2D error metrics if ground truth was available.
    }
    evaluation_results_path = os.path.join(predictions_output_root, "evaluation_results.json")
    with open(evaluation_results_path, "w") as f:
        json.dump(evaluation_results, f, indent=4)
    logger.log({"message": "Overall evaluation results saved.", "results_path": evaluation_results_path})
    print("Multiview prediction complete. Results saved to", predictions_output_root)

if __name__ == "__main__":
    predict_and_evaluate()
