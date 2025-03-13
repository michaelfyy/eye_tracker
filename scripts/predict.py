import os
import sys
import time
import json
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torchvision
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
from utils.video_helper import extract_frames, parse_annotations
from datasets.transforms import inverse_transform_2d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess_frame(frame):
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

def get_camera_id_from_video(video_path):
    # Assumes video filename is like "e1.mp4" meaning camera "cam_1"
    base = os.path.splitext(os.path.basename(video_path))[0]
    if base.startswith("e") and len(base) > 1:
        num = base[1:]
        return f"cam_{num}"
    return None

def process_video(video_path, annotation_path, config, predictions_output_dir, checkpoint_override=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    logger = Logger(predictions_output_dir, config)
    logger.log({"message": f"Processing video: {video_path}"})
    
    frames = extract_frames(video_path)
    num_frames = len(frames)
    logger.log({"message": f"Extracted {num_frames} frames."})
    
    annotations = {}
    if annotation_path and os.path.exists(annotation_path):
        annotations = parse_annotations(annotation_path)
        logger.log({"message": f"Loaded annotations from {annotation_path}", "annotated_frames": len(annotations)})
    else:
        logger.log({"message": "No annotations found for this video."})
    
    processed_frames = []
    orig_sizes = []
    for frame in frames:
        proc_frame, orig_size = preprocess_frame(frame)
        processed_frames.append(proc_frame)
        orig_sizes.append(orig_size)
    inputs = torch.stack(processed_frames).to(device)
    
    model = get_model(config).to(device)
    # For ue2_separate, use the checkpoint for the specific camera
    if config['data']['dataset'] == "ue2_separate":
        cam_id = get_camera_id_from_video(video_path)
        if cam_id is None:
            raise ValueError("Could not determine camera id from video filename.")
        common_ckpt = os.path.join(config['logging']['run_dir'], f"latest_best_model_{config['data']['dataset']}_{cam_id}.pt")
        checkpoint_path = checkpoint_override if checkpoint_override else common_ckpt
    else:
        checkpoint_path = config['testing']['checkpoint'] if not checkpoint_override else checkpoint_override
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    logger.log({"message": "Model loaded for prediction.", "checkpoint": checkpoint_path})
    
    start_time = time.time()
    with torch.no_grad():
        predictions = model(inputs)
    total_inference_time = time.time() - start_time
    avg_time_per_frame = total_inference_time / num_frames
    logger.log({"message": "Inference complete.",
                "total_inference_time_sec": total_inference_time,
                "avg_time_per_frame_sec": avg_time_per_frame})
    
    pupil_preds = predictions['pupil']
    frame_errors = []
    annotated_frames = []
    evaluated_frame_count = 0
    error_sum = 0.0
    for i in range(num_frames):
        pred_resized = pupil_preds[i].cpu()
        orig_size = orig_sizes[i]
        pred_orig = inverse_transform_2d(pred_resized, orig_size, target_size=(224, 224))
        pred_orig_np = pred_orig.cpu().numpy()
        if i in annotations:
            ann = annotations[i][0]
            gt_point = ann[1]
            gt_tensor = torch.tensor(gt_point, dtype=torch.float)
            error = torch.norm(pred_orig - gt_tensor, p=2).item()
            frame_errors.append({"frame": i, "pixel_error": error})
            evaluated_frame_count += 1
            error_sum += error
        else:
            frame_errors.append({"frame": i, "pixel_error": None})
            gt_point = None
        annotated = annotate_frame(frames[i], gt_point, pred_orig_np)
        annotated_frames.append(annotated)
    
    avg_pixel_error = error_sum / evaluated_frame_count if evaluated_frame_count > 0 else None
    
    # Write annotated video directly from memory to predictions_output_dir.
    height, width, _ = annotated_frames[0].shape
    output_video_path = os.path.join(predictions_output_dir, f"{video_name}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, config['testing'].get('frame_rate', 30), (width, height))
    for frame in annotated_frames:
        out_video.write(frame)
    out_video.release()
    logger.log({"message": "Annotated video created.", "annotated_video_path": output_video_path})
    
    # Generate evaluation plots.
    pixel_error_plot_path = os.path.join(predictions_output_dir, f"{video_name}_per_frame_pixel_error.png")
    plt.figure(figsize=(10,5))
    plt.plot([e["frame"] for e in frame_errors],
             [e["pixel_error"] if e["pixel_error"] is not None else 0 for e in frame_errors],
             marker='o')
    plt.xlabel("Frame number")
    plt.ylabel("Pixel error (in original coordinates)")
    plt.title("Per-frame Pixel Error")
    plt.grid(True)
    plt.savefig(pixel_error_plot_path)
    plt.close()
    logger.log({"message": "Saved per-frame pixel error plot.", "plot_path": pixel_error_plot_path})
    
    histogram_path = os.path.join(predictions_output_dir, f"{video_name}_pixel_error_histogram.png")
    valid_errors = [e["pixel_error"] for e in frame_errors if e["pixel_error"] is not None]
    plt.figure(figsize=(8,5))
    plt.hist(valid_errors, bins=20, edgecolor='black')
    plt.xlabel("Pixel error")
    plt.ylabel("Frequency")
    plt.title("Histogram of Pixel Errors")
    plt.savefig(histogram_path)
    plt.close()
    logger.log({"message": "Saved pixel error histogram.", "histogram_path": histogram_path})
    
    evaluation_results = {
        "video_file": os.path.basename(video_path),
        "total_frames": num_frames,
        "evaluated_frames": evaluated_frame_count,
        "average_pixel_error": avg_pixel_error,
        "frame_pixel_errors": frame_errors,
        "total_inference_time_sec": total_inference_time,
        "avg_time_per_frame_sec": avg_time_per_frame
    }
    evaluation_results_path = os.path.join(predictions_output_dir, f"{video_name}_evaluation_results.json")
    with open(evaluation_results_path, "w") as f:
        json.dump(evaluation_results, f, indent=4)
    logger.log({"message": "Saved evaluation results.", "results_path": evaluation_results_path})
    
    return evaluation_results

def predict_and_evaluate_all():
    config = parse_args()
    if config['testing']['checkpoint'] == "latest":
        if config['data']['dataset'] != "ue2_separate":
            common_ckpt = os.path.join(config['logging']['run_dir'], f"latest_best_model_{config['data']['dataset']}.pt")
            if os.path.exists(common_ckpt):
                config['testing']['checkpoint'] = common_ckpt
            else:
                raise FileNotFoundError("Latest checkpoint not found. Please run training first.")
        # For ue2_separate, the checkpoint will be determined per video.
    
    predictions_output_root = os.path.join(config['logging']['run_dir'], "predictions", datetime.now().strftime('%Y-%m-%d_%H-%M'))
    os.makedirs(predictions_output_root, exist_ok=True)
    
    video_root = config['data']['video_root']
    annotations_folder = config['data']['annotations_folder']
    video_files = glob.glob(os.path.join(video_root, "*.mp4"))
    
    overall_results = {}
    for video_file in video_files:
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        predictions_output_dir = os.path.join(predictions_output_root, video_name)
        os.makedirs(predictions_output_dir, exist_ok=True)
        annot_file = os.path.join(video_root, annotations_folder, f"{video_name}_annotations.xml")
        # For ue2_separate, determine camera id from video filename.
        checkpoint_override = None
        if config['data']['dataset'] == "ue2_separate":
            cam_id = None
            # Assuming video file "e1.mp4" corresponds to "cam_1", "e2.mp4" to "cam_2", etc.
            if video_name.startswith("e") and len(video_name) > 1:
                num = video_name[1:]
                cam_id = f"cam_{num}"
            if cam_id:
                common_ckpt = os.path.join(config['logging']['run_dir'], f"latest_best_model_{config['data']['dataset']}_{cam_id}.pt")
                if os.path.exists(common_ckpt):
                    checkpoint_override = common_ckpt
                else:
                    raise FileNotFoundError(f"Checkpoint for {cam_id} not found. Please run training first.")
        evaluation_results = process_video(video_file, annot_file, config, predictions_output_dir, checkpoint_override)
        overall_results[video_name] = evaluation_results
    overall_results_path = os.path.join(predictions_output_root, "overall_evaluation.json")
    with open(overall_results_path, "w") as f:
        json.dump(overall_results, f, indent=4)
    print("Evaluation for all videos complete. Results saved to", overall_results_path)

if __name__ == "__main__":
    predict_and_evaluate_all()
