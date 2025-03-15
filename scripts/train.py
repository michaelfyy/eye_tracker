import os
import sys
import json
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

# Ensure parent directory is in the path.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.config import parse_args
from utils.dataloader import load_data
from utils.logger import Logger
from models import get_model
from losses.losses import get_loss
from metrics.metrics import compute_pixel_error

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    config = parse_args()
    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    
    # Early stopping parameters:
    early_stop_patience = config['training'].get('early_stop', 5)
    
    if config['data']['dataset'] == "ue2_separate":
        dataloaders_dict = load_data(config)
        for cam, loaders in dataloaders_dict.items():
            run_dir = os.path.join(config['logging']['run_dir'], f"run_{run_timestamp}_{cam}")
            os.makedirs(run_dir, exist_ok=True)
            logger = Logger(run_dir, config)
            logger.log({"message": f"Training model for camera {cam}"})
            
            model = get_model(config).to(device)
            if config['training'].get('from_checkpoint', False):
                # Use individual checkpoint for this camera.
                if cam == "cam_1":
                    cp = config['training'].get('checkpoint_cam_1', None)
                elif cam == "cam_2":
                    cp = config['training'].get('checkpoint_cam_2', None)
                elif cam == "cam_3":
                    cp = config['training'].get('checkpoint_cam_3', None)
                elif cam == "cam_4":
                    cp = config['training'].get('checkpoint_cam_4', None)
                else:
                    cp = None
                if cp == "latest":
                    cp = os.path.join(config['logging']['run_dir'], f"latest_best_model_{config['data']['dataset']}_{cam}.pt")
                if not cp or not os.path.exists(cp):
                    raise ValueError(f"from_checkpoint is True but checkpoint for {cam} is not provided or does not exist.")
                model.load_state_dict(torch.load(cp, map_location=device, weights_only=True))
                logger.log({"message": f"Loaded model weights from checkpoint for {cam}: {cp}"})
            logger.log({"message": f"Model {config['model']} loaded for camera {cam}."})
            
            criterion_pupil = get_loss(config['loss']['pupil'], 'pupil')
            optimizer = optim.Adam(
                model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
            
            best_val_loss = float('inf')
            best_epoch_metrics = None
            metrics = {"epoch": [], "train_loss": [], "train_pixel_error": [], "val_loss": [], "val_pixel_error": []}
            no_improvement_count = 0
            num_epochs = config['training']['num_epochs']
            
            for epoch in range(num_epochs):
                model.train()
                train_running_loss = 0.0
                train_running_pixel_error = 0.0
                train_samples = 0
                for batch in loaders['train']:
                    images = batch['image'].to(device)
                    pupil_labels = batch['pupil'].to(device)
                    orig_sizes = batch['orig_size'].to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion_pupil(outputs['pupil'], pupil_labels)
                    loss.backward()
                    optimizer.step()
                    
                    batch_size = images.size(0)
                    train_running_loss += loss.item() * batch_size
                    batch_pixel_error = compute_pixel_error(outputs['pupil'], pupil_labels, orig_sizes)
                    train_running_pixel_error += batch_pixel_error * batch_size
                    train_samples += batch_size
                train_loss = train_running_loss / train_samples
                train_pixel_error = train_running_pixel_error / train_samples
                
                model.eval()
                val_running_loss = 0.0
                val_running_pixel_error = 0.0
                val_samples = 0
                with torch.no_grad():
                    for batch in loaders['val']:
                        images = batch['image'].to(device)
                        pupil_labels = batch['pupil'].to(device)
                        orig_sizes = batch['orig_size'].to(device)
                        outputs = model(images)
                        loss = criterion_pupil(outputs['pupil'], pupil_labels)
                        batch_pixel_error = compute_pixel_error(outputs['pupil'], pupil_labels, orig_sizes)
                        batch_size = images.size(0)
                        val_running_loss += loss.item() * batch_size
                        val_running_pixel_error += batch_pixel_error * batch_size
                        val_samples += batch_size
                val_loss = val_running_loss / val_samples
                val_pixel_error = val_running_pixel_error / val_samples
                
                metrics['epoch'].append(epoch + 1)
                metrics['train_loss'].append(train_loss)
                metrics['train_pixel_error'].append(train_pixel_error)
                metrics['val_loss'].append(val_loss)
                metrics['val_pixel_error'].append(val_pixel_error)
                logger.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_pixel_error": train_pixel_error,
                    "val_loss": val_loss,
                    "val_pixel_error": val_pixel_error
                })
                print(f"Camera {cam} | Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f}, Pixel Err: {train_pixel_error:.6f} | Val Loss: {val_loss:.6f}, Pixel Err: {val_pixel_error:.6f}")
                
                # Early stopping check on validation loss.
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch_metrics = {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_pixel_error": train_pixel_error,
                        "val_loss": val_loss,
                        "val_pixel_error": val_pixel_error
                    }
                    no_improvement_count = 0
                    checkpoint_path = os.path.join(run_dir, "best_model.pt")
                    torch.save(model.state_dict(), checkpoint_path)
                    logger.log({"message": "Best model saved", "epoch": epoch + 1, "val_loss": val_loss})
                    common_ckpt = os.path.join(config['logging']['run_dir'], f"latest_best_model_{config['data']['dataset']}_{cam}.pt")
                    shutil.copy(checkpoint_path, common_ckpt)
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= early_stop_patience:
                        print(f"Early stopping triggered for camera {cam} at epoch {epoch+1}")
                        logger.log({"message": f"Early stopping triggered for camera {cam} at epoch {epoch+1}"})
                        break
                        
                with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=4)
                plt.figure()
                plt.plot(metrics['epoch'], metrics['train_loss'], label='Train Loss')
                plt.plot(metrics['epoch'], metrics['val_loss'], label='Val Loss')
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"Training and Validation Loss for {cam}")
                plt.legend()
                plt.savefig(os.path.join(run_dir, "loss_plot.png"))
                plt.close()
                plt.figure()
                plt.plot(metrics['epoch'], metrics['train_pixel_error'], label='Train Pixel Error')
                plt.plot(metrics['epoch'], metrics['val_pixel_error'], label='Val Pixel Error')
                plt.xlabel("Epoch")
                plt.ylabel("Pixel Error")
                plt.title(f"Training and Validation Pixel Error for {cam}")
                plt.legend()
                plt.savefig(os.path.join(run_dir, "pixel_error_plot.png"))
                plt.close()
            
            logger.log({"Best Epoch Metrics for " + cam: best_epoch_metrics})
            print(f"Camera {cam} training complete. Best Epoch: {best_epoch_metrics['epoch']}, Best Val Loss: {best_val_loss:.6f}")
            logger.log({"best validation loss": f"{best_val_loss:.6f}"})
            
    else:
        # For syntheseyes and ue2 (combined), use a single dataloader.
        dataloaders = load_data(config)
        run_dir = os.path.join(config['logging']['run_dir'], f"run_{run_timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        logger = Logger(run_dir, config)
        model = get_model(config).to(device)
        if config['training'].get('from_checkpoint', False):
            cp = config['training'].get('checkpoint_path', None)
            if cp == "latest":
                cp = os.path.join(config['logging']['run_dir'], f"latest_best_model_{config['data']['dataset']}.pt")
            if not cp or not os.path.exists(cp):
                raise ValueError("from_checkpoint is True but checkpoint_path is not provided or does not exist.")
            model.load_state_dict(torch.load(cp, map_location=device, weights_only=True))
            logger.log({"message": f"Loaded model weights from checkpoint: {cp}"})
        logger.log({"message": f"Model {config['model']} loaded."})
        print(f"Using device: {device}")
        criterion_pupil = get_loss(config['loss']['pupil'], 'pupil')
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        best_val_loss = float('inf')
        best_epoch_metrics = None
        metrics = {"epoch": [], "train_loss": [], "train_pixel_error": [], "val_loss": [], "val_pixel_error": []}
        no_improvement_count = 0
        num_epochs = config['training']['num_epochs']
        for epoch in range(num_epochs):
            model.train()
            train_running_loss = 0.0
            train_running_pixel_error = 0.0
            train_samples = 0
            for batch in dataloaders['train']:
                images = batch['image'].to(device)
                pupil_labels = batch['pupil'].to(device)
                orig_sizes = batch['orig_size'].to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion_pupil(outputs['pupil'], pupil_labels)
                loss.backward()
                optimizer.step()
                batch_size = images.size(0)
                train_running_loss += loss.item() * batch_size
                batch_pixel_error = compute_pixel_error(outputs['pupil'], pupil_labels, orig_sizes)
                train_running_pixel_error += batch_pixel_error * batch_size
                train_samples += batch_size
            train_loss = train_running_loss / train_samples
            train_pixel_error = train_running_pixel_error / train_samples
            model.eval()
            val_running_loss = 0.0
            val_running_pixel_error = 0.0
            val_samples = 0
            with torch.no_grad():
                for batch in dataloaders['val']:
                    images = batch['image'].to(device)
                    pupil_labels = batch['pupil'].to(device)
                    orig_sizes = batch['orig_size']
                    outputs = model(images)
                    loss = criterion_pupil(outputs['pupil'], pupil_labels)
                    batch_pixel_error = compute_pixel_error(outputs['pupil'], pupil_labels, orig_sizes)
                    batch_size = images.size(0)
                    val_running_loss += loss.item() * batch_size
                    val_running_pixel_error += batch_pixel_error * batch_size
                    val_samples += batch_size
            val_loss = val_running_loss / val_samples
            val_pixel_error = val_running_pixel_error / val_samples
            metrics['epoch'].append(epoch + 1)
            metrics['train_loss'].append(train_loss)
            metrics['train_pixel_error'].append(train_pixel_error)
            metrics['val_loss'].append(val_loss)
            metrics['val_pixel_error'].append(val_pixel_error)
            logger.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_pixel_error": train_pixel_error,
                "val_loss": val_loss,
                "val_pixel_error": val_pixel_error
            })
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f}, Pixel Err: {train_pixel_error:.6f} | Val Loss: {val_loss:.6f}, Pixel Err: {val_pixel_error:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch_metrics = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_pixel_error": train_pixel_error,
                    "val_loss": val_loss,
                    "val_pixel_error": val_pixel_error
                }
                no_improvement_count = 0
                checkpoint_path = os.path.join(run_dir, "best_model.pt")
                torch.save(model.state_dict(), checkpoint_path)
                logger.log({"message": "Best model saved based on validation evaluation", "epoch": epoch+1, "val_loss": val_loss})
                common_ckpt = os.path.join(config['logging']['run_dir'], f"latest_best_model_{config['data']['dataset']}.pt")
                shutil.copy(checkpoint_path, common_ckpt)
            else:
                no_improvement_count += 1
                if no_improvement_count >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    logger.log({"message": f"Early stopping triggered at epoch {epoch+1}"})
                    break
            
            with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
            plt.figure()
            plt.plot(metrics['epoch'], metrics['train_loss'], label='Train Loss')
            plt.plot(metrics['epoch'], metrics['val_loss'], label='Val Loss')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.savefig(os.path.join(run_dir, "loss_plot.png"))
            plt.close()
            plt.figure()
            plt.plot(metrics['epoch'], metrics['train_pixel_error'], label='Train Pixel Error')
            plt.plot(metrics['epoch'], metrics['val_pixel_error'], label='Val Pixel Error')
            plt.xlabel("Epoch")
            plt.ylabel("Pixel Error")
            plt.title("Training and Validation Pixel Error")
            plt.legend()
            plt.savefig(os.path.join(run_dir, "pixel_error_plot.png"))
            plt.close()
        logger.log({"Best Epoch Metrics (lowest validation loss)": best_epoch_metrics})
        print("Training complete. Best Epoch: {}, Best Validation Loss: {:.6f}".format(best_epoch_metrics["epoch"], best_val_loss))
        logger.log({"best validation loss": f"{best_val_loss:.6f}"})

if __name__ == "__main__":
    train()
