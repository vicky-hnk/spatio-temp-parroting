import json
import os
import pandas as pd

class MetricsLogger:
    """
    A unified logger to save all scalar, per-epoch metrics to a single CSV file.
    Large artifacts like attention matrices are saved separately.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.metrics_filepath = os.path.join(output_dir, 'epoch_metrics.csv')
        self.metrics = []


    def log_epoch(self, epoch_data: dict):
        """
        Collects metrics for the current epoch.
        Expects a dictionary, e.g., {'epoch': 1, 'train_loss': 0.5, ...}
        """
        self.metrics.append(epoch_data)


    def save(self):
        """Saves all collected metrics to the CSV file."""
        if not self.metrics:
            return
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.metrics_filepath, index=False)
        print(f"Epoch metrics saved to {self.metrics_filepath}")


    def log_attention(self, attn_tensor, epoch, attn_type):
        """Logs large artifacts like attention matrices to their own files."""
        save_dir = os.path.join(self.output_dir, "logs", "attention")
        os.makedirs(save_dir, exist_ok=True)
        log_data = {'epoch': epoch, 'attn_matrix': attn_tensor.tolist()}
        file_path = os.path.join(save_dir, f"attn_{attn_type}_epoch_{epoch:04d}.json")
        with open(file_path, 'w') as f:
            json.dump(log_data, f)