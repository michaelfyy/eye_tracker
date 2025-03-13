import os
import torch
import json

class Logger:
    def __init__(self, log_dir, config):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "log.txt")
        # Save configuration for record
        with open(os.path.join(log_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

    def log(self, info):
        # Simple console and file logging
        message = ", ".join([f"{k}: {v}" for k, v in info.items()])
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")

    def save_checkpoint(self, model, epoch):
        path = os.path.join(self.log_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(model.state_dict(), path)
        print(f"Checkpoint saved to {path}")
