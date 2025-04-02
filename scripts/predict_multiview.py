import os
import sys
import time
import json
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import argparse
import torchvision.transforms.v2 as transforms
import torch.nn.functional as F

# Ensure parent directory is in the path.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.config import parse_args
from models import get_model
from utils.logger import Logger
from utils.video_helper import extract_frames, parse_annotations
from datasets.transforms import inverse_transform_2d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess_frame(frame):
    """
    Preprocess a single frame:
      - Assumes the frame is in RGB (extracted via extract_frames).
      - Converts to tensor, permutes dimensions to (C, H, W), resizes to (224, 224) and normalizes.
    Returns:
      processed_frame: Tensor of shape (C, 224, 224)
      orig_size: Tuple (height, width)
    """
    orig_size = (frame.shape[0], frame.shape[1])
    frame_tensor = torch.tensor(frame)
    frame_tensor = transforms.functional.convert_image_dtype(frame_tensor, torch.float)
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
    Annotate an RGB frame with ground truth (if available) and predicted 2D point.
    The frame is converted to BGR for visualization.
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

def get_camera_id_from_filename(filename):
    """
    Given a video filename (e.g. "e1.mp4"), return the corresponding camera id (e.g. "cam_1").
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    if base.startswith("e") and len(base) > 1:
        num = base[1:]
        return f"cam_{num}"
    return None

def load_multiview_videos(video_root):
    """
    Scan the video_root folder for mp4 files and map them to camera ids.
    Expects filenames like "e1.mp4", "e2.mp4", etc.
    Returns a dict mapping camera id to video path.
    """
    video_files = glob.glob(os.path.join(video_root, "*.mp4"))
    video_paths = {}
    for vf in video_files:
        cam = get_camera_id_from_filename(vf)
        if cam is not None:
            video_paths[cam] = vf
    if len(video_paths) < 4:
        raise ValueError("Not all camera videos found. Expected videos for cam_1, cam_2, cam_3, and cam_4.")
    return video_paths

def extract_multiview_frames(video_paths):
    """
    For each camera video, use the helper extract_frames (which converts BGR to RGB)
    and return a dictionary mapping camera id to list of frames.
    """
    frames_dict = {}
    for cam, path in video_paths.items():
        frames = extract_frames(path)
        frames_dict[cam] = frames
    return frames_dict

def predict_and_evaluate():
    config = parse_args()
    # Create a timestamped predictions output folder.
    predictions_output_root = os.path.join(config['logging']['run_dir'], "predictions", datetime.now().strftime('%Y-%m-%d_%H-%M'))
    os.makedirs(predictions_output_root, exist_ok=True)
    # Create a logger that writes to this folder.
    logger = Logger(predictions_output_root, config)

    # Determine checkpoint (support "latest" option).
    if config['testing']['checkpoint'] == "latest":
        common_ckpt = os.path.join(config['logging']['run_dir'], f"latest_best_model_{config['data']['dataset']}.pt")
        if os.path.exists(common_ckpt):
            config['testing']['checkpoint'] = common_ckpt
        else:
            raise FileNotFoundError("Latest checkpoint not found. Please run training first.")

    # Load multiview videos.
    video_root = config['data']['video_root']
    video_paths = load_multiview_videos(video_root)
    logger.log({"message": "Multiview video paths loaded.", "video_paths": video_paths})

    # Load annotations for each camera.
    # Assume that config['data']['annotations_folder'] gives the folder where annotation XML files are stored.
    # For each camera video (e.g. "e1.mp4" for cam_1), we look for an annotation file named "e1_annotations.xml".
    sorted_cams = sorted(video_paths.keys())
    annotations = {}
    for cam in sorted_cams:
        video_file = video_paths[cam]
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        annot_file = os.path.join(video_root, config['data']['annotations_folder'], f"{video_name}_annotations.xml")
        if os.path.exists(annot_file):
            annotations[cam] = parse_annotations(annot_file)
            logger.log({"message": f"Loaded annotations for {cam} from {annot_file}"})
        else:
            annotations[cam] = {}
            logger.log({"message": f"No annotations found for {cam}."})

    # Extract frames for each camera (using extract_frames for proper RGB conversion).
    frames_dict = extract_multiview_frames(video_paths)
    # Synchronize: use the minimum frame count across cameras.
    num_frames = min(len(frames) for frames in frames_dict.values())
    logger.log({"message": f"Extracted {num_frames} synchronized frames per camera."})

    # Prepare multiview input tensor, record per-camera original sizes and original frames.
    multiview_samples = []  # List to collect per-frame multiview tensors.
    orig_sizes_per_view = {cam: [] for cam in sorted_cams}
    orig_frames_per_view = {cam: frames_dict[cam][:num_frames] for cam in sorted_cams}

    for i in range(num_frames):
        sample_views = []
        for cam in sorted_cams:
            frame = frames_dict[cam][i]
            proc_frame, orig_size = preprocess_frame(frame)
            sample_views.append(proc_frame)
            orig_sizes_per_view[cam].append(orig_size)
        sample_tensor = torch.stack(sample_views, dim=0)  # (views, C, 224, 224)
        multiview_samples.append(sample_tensor)
    # Stack to create batch: (N, views, C, H, W)
    inputs = torch.stack(multiview_samples, dim=0).to(device)
    logger.log({"message": "Prepared multiview input tensor.", "input_shape": list(inputs.shape)})

    # Load the multiview model.
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

    # Expected prediction keys: e.g., "cam_1_2d", "cam_1_3d", ..., "cam_4_2d", "cam_4_3d"
    # Process 2D predictions for visualization.
    annotated_frames = {cam: [] for cam in sorted_cams}
    per_frame_errors = {cam: [] for cam in sorted_cams}  # Will store error per frame if GT exists

    # For each frame and each camera, convert prediction from resized space to original image space
    for idx in range(num_frames):
        for cam in sorted_cams:
            pred_2d = predictions[f"{cam}_2d"][idx].cpu()
            orig_size = orig_sizes_per_view[cam][idx]
            pred_orig = inverse_transform_2d(pred_2d, orig_size, target_size=(224, 224))
            pred_orig_np = pred_orig.cpu().numpy()
            # Check if annotations for this camera contain this frame.
            if idx in annotations[cam]:
                ann = annotations[cam][idx][0]
                gt_point = ann[1]  # gt is expected to be in original image coordinates.
                gt_tensor = torch.tensor(gt_point, dtype=torch.float)
                error = torch.norm(pred_orig - gt_tensor, p=2).item()
                per_frame_errors[cam].append(error)
            else:
                per_frame_errors[cam].append(None)
                gt_point = None
            annotated = annotate_frame(orig_frames_per_view[cam][idx], gt_point, pred_orig_np)
            annotated_frames[cam].append(annotated)

    # Save per-camera annotated videos and evaluation plots.
    for cam in sorted_cams:
        video_path = os.path.join(predictions_output_root, f"{cam}_annotated.mp4")
        if len(annotated_frames[cam]) > 0:
            height, width, _ = annotated_frames[cam][0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(video_path, fourcc, config['testing'].get('frame_rate', 30), (width, height))
            for frame in annotated_frames[cam]:
                out_video.write(frame)
            out_video.release()
            logger.log({"message": f"Annotated video saved for {cam}.", "video_path": video_path})
        valid_errors = [e for e in per_frame_errors[cam] if e is not None]
        if valid_errors:
            plt.figure(figsize=(10, 5))
            plt.plot(range(num_frames), [e if e is not None else 0 for e in per_frame_errors[cam]], marker='o')
            plt.xlabel("Frame number")
            plt.ylabel("Pixel error (in original coordinates)")
            plt.title(f"Per-frame Pixel Error for {cam}")
            plt.grid(True)
            error_plot_path = os.path.join(predictions_output_root, f"{cam}_per_frame_pixel_error.png")
            plt.savefig(error_plot_path)
            plt.close()
            logger.log({"message": f"Pixel error plot saved for {cam}.", "plot_path": error_plot_path})
            plt.figure(figsize=(8, 5))
            plt.hist(valid_errors, bins=20, edgecolor='black')
            plt.xlabel("Pixel error")
            plt.ylabel("Frequency")
            plt.title(f"Pixel Error Histogram for {cam}")
            hist_path = os.path.join(predictions_output_root, f"{cam}_pixel_error_histogram.png")
            plt.savefig(hist_path)
            plt.close()
            logger.log({"message": f"Pixel error histogram saved for {cam}.", "histogram_path": hist_path})

    # Process 3D predictions: save raw outputs.
    predictions_3d = {}
    for cam in sorted_cams:
        key = f"{cam}_3d"
        predictions_3d[cam] = predictions[key].cpu().tolist()
    predictions_3d_path = os.path.join(predictions_output_root, "multiview_3d_predictions.json")
    with open(predictions_3d_path, "w") as f:
        json.dump(predictions_3d, f, indent=4)
    logger.log({"message": "3D predictions saved.", "predictions_3d_path": predictions_3d_path})

    # Create overall evaluation metrics.
    overall_evaluation = {
        "total_frames": num_frames,
        "total_inference_time_sec": total_inference_time,
        "avg_time_per_frame_sec": avg_time_per_frame,
        "checkpoint_used": checkpoint_path,
        "per_camera": {}
    }
    for cam in sorted_cams:
        cam_errors = [e for e in per_frame_errors[cam] if e is not None]
        avg_error = float(np.mean(cam_errors)) if cam_errors else None
        overall_evaluation["per_camera"][cam] = {
            "average_pixel_error": avg_error,
            "frame_pixel_errors": per_frame_errors[cam]
        }
    overall_evaluation_path = os.path.join(predictions_output_root, "overall_evaluation.json")
    with open(overall_evaluation_path, "w") as f:
        json.dump(overall_evaluation, f, indent=4)
    logger.log({"message": "Overall evaluation results saved.", "results_path": overall_evaluation_path})

    print("Multiview prediction complete. Results saved to", predictions_output_root)

if __name__ == "__main__":
    predict_and_evaluate()
