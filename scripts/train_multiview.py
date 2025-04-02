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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    config = parse_args()
    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    run_dir = os.path.join(config['logging']['run_dir'], f"run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    logger = Logger(run_dir, config)

    # Load multiview data (this branch only supports ue2_multiview)
    dataloaders = load_data(config)
    logger.log({"message": "Loaded ue2_multiview dataset."})

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

    # Define the eight output keys.
    output_keys = [
        "cam_1_2d", "cam_2_2d",
        "cam_3_2d", "cam_4_3d",
        "cam_1_3d", "cam_2_3d",
        "cam_3_3d", "cam_4_3d"
    ]

    # Create a loss function for each output key.
    loss_fns = {}
    for key in output_keys:
        # Assumes config['loss'][key] exists with a "type" and "weight"
        loss_fns[key] = get_loss(config['loss'][key], key)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    best_val_loss = float('inf')
    best_epoch_metrics = None
    metrics = {"epoch": [], "train_loss": [], "val_loss": []}
    no_improvement_count = 0
    num_epochs = config['training']['num_epochs']
    early_stop_patience = config['training'].get('early_stop', 5)

    for epoch in range(num_epochs):
        model.train()
        train_running_loss = 0.0
        train_samples = 0

        for batch in dataloaders['train']:
            images = batch['images'].to(device)  # Shape: (batch, views, channels, height, width)
            optimizer.zero_grad()

            outputs = model(images)
            loss = 0.0
            # Sum the loss over all output keys.
            for key in output_keys:
                # Get the predicted and ground truth values.
                pred = outputs[key]
                target = batch[key].to(device)
                current_loss = loss_fns[key](pred, target)
                # Multiply by the loss weight specified in config.
                weight = config['loss'][key]['weight']
                loss += weight * current_loss

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            train_running_loss += loss.item() * batch_size
            train_samples += batch_size

        train_loss = train_running_loss / train_samples

        model.eval()
        val_running_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in dataloaders['val']:
                images = batch['images'].to(device)
                outputs = model(images)
                loss = 0.0
                for key in output_keys:
                    pred = outputs[key]
                    target = batch[key].to(device)
                    current_loss = loss_fns[key](pred, target)
                    weight = config['loss'][key]['weight']
                    loss += weight * current_loss
                batch_size = images.size(0)
                val_running_loss += loss.item() * batch_size
                val_samples += batch_size

        val_loss = val_running_loss / val_samples

        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        logger.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early stopping check.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss
            }
            no_improvement_count = 0
            checkpoint_path = os.path.join(run_dir, "best_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            logger.log({"message": "Best model saved", "epoch": epoch + 1, "val_loss": val_loss})
            common_ckpt = os.path.join(config['logging']['run_dir'], f"latest_best_model_{config['data']['dataset']}.pt")
            shutil.copy(checkpoint_path, common_ckpt)
        else:
            no_improvement_count += 1
            if no_improvement_count >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                logger.log({"message": f"Early stopping triggered at epoch {epoch+1}"})
                break

        # Save metrics and plots.
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

    logger.log({"Best Epoch Metrics": best_epoch_metrics})
    print("Training complete. Best Epoch: {}, Best Val Loss: {:.6f}".format(best_epoch_metrics["epoch"], best_val_loss))
    logger.log({"best validation loss": f"{best_val_loss:.6f}"})

if __name__ == "__main__":
    train()
