import json
import time
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ConstantLR, ReduceLROnPlateau
import numpy as np
import functools

from src.util.train_utils import EarlyStopping
from src.util.hpc_utils import MetricsLogger
from src.data_loader.spatio_temporal_dataset import SpatioTemporalDataset
from src.runtime.check_jacobian import get_jacobian


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _bf16_supported():
    try:
        if hasattr(torch.cuda, "is_bf16_supported"):
            return torch.cuda.is_bf16_supported()
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    except (RuntimeError, AssertionError):
        return False


def safe_masked_mae_loss(y_pred, y_true_with_nans):
    mask = ~torch.isnan(y_true_with_nans)
    mask = mask.float()  # Convert boolean mask to float (1.0 for valid, 0.0 for NaN)
    y_true_safe = torch.nan_to_num(y_true_with_nans, nan=0.0)
    abs_error = torch.abs(y_pred - y_true_safe)
    masked_abs_error = abs_error * mask
    loss = torch.sum(masked_abs_error) / (torch.sum(mask) + 1e-8)
    return loss

def masked_mae_loss_scaled(y_pred: torch.Tensor, y_true_scaled: torch.Tensor, y_mask_raw: torch.Tensor) -> torch.Tensor:
    """
    y_pred, y_true_scaled: scaled space, same shape (e.g., [B, H, N, 1] or [B, 1, N, H])
    y_mask_raw: boolean/float mask from raw labels, 1=valid, 0=missing (same shape as y_true)
    """
    mask = y_mask_raw.to(y_pred.dtype)
    abs_err = (y_pred - y_true_scaled).abs()
    masked = abs_err * mask
    denominator = mask.sum().clamp_min(1.0)
    return masked.sum() / denominator


def masked_mae_loss_unscaled(y_pred_scaled: torch.Tensor, y_true_scaled: torch.Tensor, mask: torch.Tensor,
        scaler_mean: torch.Tensor, scaler_std: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes masked MAE in ORIGINAL units while the model outputs are in scaled units.
    Broadcast-robust, zero-std safe, and device-safe.
    """
    if y_pred_scaled.shape != y_true_scaled.shape:
        raise ValueError(f"Shape mismatch: pred {y_pred_scaled.shape} vs true {y_true_scaled.shape}")

    device = y_pred_scaled.device
    dtype = y_pred_scaled.dtype
    B, H, N, C = y_pred_scaled.shape

    if mask.dim() == 3:
        mask = mask.unsqueeze(-1)
    elif mask.dim() != 4:
        raise ValueError(f"mask must be [B,H,N] or [B,H,N,1], got {mask.shape}")
    mask = mask.to(device=device, dtype=dtype)
    mean_t = torch.as_tensor(scaler_mean, device=device, dtype=dtype)
    std_t = torch.as_tensor(scaler_std, device=device, dtype=dtype).clamp_min(eps)

    def _view_for_broadcast(t):
        if t.ndim == 0:
            return t.view(1, 1, 1, 1)
        if t.ndim == 1:
            if t.numel() == N:
                return t.view(1, 1, N, 1)
            if t.numel() == C:
                return t.view(1, 1, 1, C)
            raise ValueError(f"1D scaler shape {tuple(t.shape)} doesn't match N={N} or C={C}")
        if t.ndim == 2:
            if t.shape == (N, C):
                return t.view(1, 1, N, C)
            raise ValueError(f"2D scaler shape {tuple(t.shape)} must be (N,C)=({N},{C})")
        raise ValueError(f"Unsupported scaler ndim={t.ndim} with shape {tuple(t.shape)}")

    mu = _view_for_broadcast(mean_t)
    sig = _view_for_broadcast(std_t)

    # --- unscale ---
    y_pred_unscaled = y_pred_scaled * sig + mu
    y_true_unscaled = y_true_scaled * sig + mu

    # --- masked MAE in original units ---
    abs_err = (y_pred_unscaled - y_true_unscaled).abs()  # [B,H,N,C]
    masked_abs = abs_err * mask  # broadcast over C
    denom = mask.sum().clamp_min(1.0)
    return masked_abs.sum() / denom


class Trainer:
    """
    Handles Training and Testing of Pytorch Models.
    """

    def __init__(self, model, config, run_name, run_output_dir):
        """Initialize training function Spatio-Temporal Models."""

        self.scheduler_type = None
        self.scaler = None
        """SETTING: GENERAL"""
        self.model_trained = None
        self.model = model
        self.config = config
        self.run_name = run_name
        self.run_output_dir = run_output_dir

        """SETTING: DATASET"""
        self.pred_len = self.config.get('pred_len')

        """SETTING: TRAINING"""
        # Set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)

        """SETTING: AMP"""
        amp_cfg = (self.config.get("amp") or "auto").lower()
        if amp_cfg == "bf16":
            self.amp_dtype = torch.bfloat16
        elif amp_cfg == "fp16":
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.bfloat16 if (_bf16_supported()) else torch.float16
        # TF32 only when CUDA + Ampere+
        if self.device_type == 'cuda':
            try:
                major_capability, _ = torch.cuda.get_device_capability()
                allow_tf32 = major_capability >= 8
            except (RuntimeError, AssertionError):
                allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
        # AMP compatibility shim: GradScaler + autocast
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.grad_scaler = torch.amp.GradScaler('cuda',
                enabled=(self.device_type == 'cuda' and self.amp_dtype == torch.float16))
            self.autocast = functools.partial(torch.amp.autocast, device_type='cuda',
                dtype=self.amp_dtype, enabled=(self.device_type == 'cuda'))
        else:
            from torch.cuda.amp import GradScaler as CudaGradScaler, autocast as cuda_autocast
            self.grad_scaler = CudaGradScaler(enabled=(self.device_type == 'cuda' and self.amp_dtype == torch.float16))
            self.autocast = functools.partial(cuda_autocast, dtype=self.amp_dtype, enabled=(self.device_type == 'cuda'))

        """SETTING: OPTIMIZATION"""
        # Accumulation / clipping
        self.grad_accum_steps = int(self.config.get("grad_accum_steps", 1))
        self.grad_clip_norm = float(self.config.get("grad_clip_norm", 5.0))

        # Optimizer initialization
        lr = self.config.get('learning_rate', 0.001)
        if self.config.get('optimizer') == 'SGD':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        elif self.config.get('optimizer') == 'AdamW':
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Loss function initialization
        self.loss_name = self.config.get('loss', 'MAE')
        if self.config.get('loss') == "MAE":
            self.loss_criterion = torch.nn.L1Loss()
        elif self.config.get('loss') == "Smooth":
            self.loss_criterion = torch.nn.SmoothL1Loss()
        elif self.config.get('loss') == "masked":
            self.loss_criterion = safe_masked_mae_loss
        elif self.config.get('loss') == "mask_scaled":
            self.loss_criterion = masked_mae_loss_scaled
        elif self.config.get('loss') == "unscaled":
            self.loss_criterion = masked_mae_loss_unscaled
        else:
            self.loss_criterion = torch.nn.MSELoss()


    def configure_dataset(self, flag):
        set_shuffle = flag in {"train"} and self.config["shuffle"]
        batch_size = self.config.get("batch_size", None)
        if batch_size is None:
            raise ValueError("batch_size not provided. Set batch_size.")
        drop_last = flag in {'train'}
        if flag == "train":
            dataset = SpatioTemporalDataset(config=self.config, flag=flag)
            if not isinstance(dataset, Dataset):
                raise ValueError(
                    "Dataset must be an instance of torch.utils.data.Dataset.")
            self.scaler = dataset.scaler
        else:
            dataset = SpatioTemporalDataset(config=self.config, flag=flag, scaler=self.scaler)
            if not isinstance(dataset, Dataset):
                raise ValueError(
                    "Dataset must be an instance of torch.utils.data.Dataset.")
        data_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=set_shuffle,
            num_workers=self.config.get("num_workers", 0),
            drop_last=drop_last)
        print(
            f"Data loaded for: {flag} - task\nThe length of the dataset is: "
            f"{len(dataset)}")
        return dataset, data_loader

    def get_training_data(self):
        training_set, training_loader = self.configure_dataset(flag="train")
        print("Train data loading completed!\n Data Set: ", training_set)
        validation_set, validation_loader = self.configure_dataset(flag="val")
        print("Validation data loading completed!\n Data Set: ",
              validation_set)
        return training_set, training_loader, validation_set, validation_loader

    def set_training_args(self):

        # SCHEDULER
        self.scheduler_type = self.config.get('scheduler', None)
        print(f"Initializing scheduler: {self.scheduler_type}")
        if self.scheduler_type == 'sequential_cosine':
            warmup_epochs = self.config.get('warmup_epochs', 5)
            hold_epochs = self.config.get('hold_epochs', 8)
            warmup = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0,
                              total_iters=warmup_epochs)
            hold = ConstantLR(self.optimizer, factor=1.0, total_iters=hold_epochs)
            cosine = CosineAnnealingLR(self.optimizer,
                                       T_max=self.config.get('epochs') - warmup_epochs - hold_epochs,
                                       eta_min=self.config.get('min_learning_rate', 1e-6))

            schedule = SequentialLR(self.optimizer, schedulers=[warmup, hold, cosine],
                                    milestones=[warmup_epochs, warmup_epochs + hold_epochs])
        elif self.scheduler_type == 'ReduceOnPlateau':
            schedule = ReduceLROnPlateau(self.optimizer, mode='min', factor=self.config.get('scheduler_factor', 0.1),
                                         patience=self.config.get('scheduler_patience', 10))
        elif self.scheduler_type is None:
            schedule = None
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")

        # Early stopping
        stop = EarlyStopping(patience=self.config.get('stop_patience'),
                             delta=self.config.get('stop_delta'),
                             mode='min')
        return schedule, stop

    def _to_device(self, *tensors):
        out = []
        for t in tensors:
            if torch.is_tensor(t):
                out.append(t.to(self.device, non_blocking=True))
            else:
                out.append(t)
        return out if len(out) > 1 else out[0]

    def train(self):
        torch.autograd.set_detect_anomaly(False)
        logger = MetricsLogger(self.run_output_dir)

        # ---LOAD DATA---
        _, train_loader, _, val_loader = self.get_training_data()

        # ---TRAINING SETTINGS---
        scheduler, early_stopping = self.set_training_args()

        # ---RUN EPOCHS---
        avg_train_loss = None
        start_time = time.time()
        model_name = self.model.__class__.__name__
        print(f"\n{'=' * 46}\nTraining of {model_name} "
                f"successfully started.\n{'=' * 46}\n")

        for epoch in range(self.config.get('epochs')):
            # --- init epoch accumulators ---
            all_dirichlet_batches, all_time_attn = [], []
            train_loss, val_loss = [], []
            epoch_time = time.time()

            # --- TRAIN ---
            self.model.train()
            for step, (train_batch_x, train_batch_y, train_raw_y, train_adj,
                       train_mask) in enumerate(train_loader, start=1):

                # No accumulation
                self.optimizer.zero_grad(set_to_none=True)

                train_batch_x, train_batch_y = map(lambda x: x.float(), [train_batch_x, train_batch_y])
                train_batch_x, train_batch_y, train_adj, train_mask = self._to_device(
                    train_batch_x, train_batch_y, train_adj, train_mask)

                with torch.amp.autocast(self.device_type, dtype=self.amp_dtype, enabled=(self.device_type == 'cuda')):
                    # desired outputs shape is (B, T, N, C)
                    outputs, dirichlet, space_attn, time_attn = self.model(train_batch_x, train_adj)
                    if outputs is None:
                        break
                    if self.loss_name == "mask_scaled":
                        loss = self.loss_criterion(outputs, train_batch_y, train_mask)
                    elif self.loss_name == "unscaled":
                        scaler_mean = torch.tensor(self.scaler.mean_, dtype=torch.float32).to(self.device)
                        scaler_std = torch.tensor(self.scaler.scale_, dtype=torch.float32).to(self.device)
                        if scaler_mean is None or scaler_std is None:
                            raise RuntimeError("No scaler applied - cannot compute unscaled loss. "
                                               "Choose different loss criterion.")
                        loss = self.loss_criterion(outputs, train_batch_y, train_mask, scaler_mean, scaler_std)
                    else:
                        loss = self.loss_criterion(outputs, train_batch_y)

                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                train_loss.append(float(loss.detach().item()))

                # metrics per batch
                if self.config.get("dirichlet", False) and dirichlet is not None:
                    all_dirichlet_batches.append(torch.stack([d.detach().cpu() for d in dirichlet]))
                if self.config.get("log_attn", False) and time_attn is not None:
                    all_time_attn.append(time_attn.detach().cpu())

            avg_train_loss = float(np.average(train_loss)) if len(train_loss) else float('nan')

            # --- EPOCH_WISE VALIDATION ---
            self.model.eval()
            with torch.no_grad():
                for _, (val_batch_x, val_batch_y, val_raw_y, val_adj, val_mask) in enumerate(val_loader):
                    val_batch_x = val_batch_x.float()
                    val_batch_y = val_batch_y.float()
                    val_batch_x, val_batch_y, val_adj, val_mask = self._to_device(
                        val_batch_x, val_batch_y, val_adj, val_mask)

                    with torch.amp.autocast(self.device_type, dtype=self.amp_dtype,
                                            enabled=(self.device_type == 'cuda')):
                        val_outputs, _, _, _ = self.model(val_batch_x, val_adj)

                        if self.loss_name == "mask_scaled":
                            val_loss_item = self.loss_criterion(val_outputs, val_batch_y, val_mask)
                        elif self.loss_name == "unscaled":
                            val_loss_item = self.loss_criterion(val_outputs, val_batch_y, val_mask, scaler_mean, scaler_std)
                        else:
                            val_loss_item = self.loss_criterion(val_outputs, val_batch_y)

                    val_loss.append(float(val_loss_item.detach().item()))
            avg_val_loss = float(np.average(val_loss)) if len(val_loss) else float('nan')
            print(f"Validation loss is {avg_val_loss}")

            # --- EARLY STOPPING (epoch-level) ---
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print("No improvement - training stopped early.")
                break

            # --- SCHEDULER STEP ---
            lr_before = self.optimizer.param_groups[0]['lr']
            if scheduler:
                if self.scheduler_type == "ReduceOnPlateau":
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            lr_after = self.optimizer.param_groups[0]['lr']
            if lr_before != lr_after:
                print(f"Learning rate changed from {lr_before} to {lr_after}")

            # --- LOG EPOCH METRICS ---
            epoch_data = {
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }

            if self.config.get("dirichlet", False) and all_dirichlet_batches:
                num_layers = len(all_dirichlet_batches[0])
                for i in range(num_layers):
                    avg_energy = torch.mean(torch.stack([b[i] for b in all_dirichlet_batches])).item()
                    epoch_data[f'dirichlet_layer_{i + 1}'] = avg_energy

            if self.config.get("log_jacob", False) and (epoch % int(self.config.get("jacobian_every", 10)) == 0):
                jacobian = get_jacobian(
                    self.model, val_loader,
                    probes=int(self.config.get("jacobian_samples", 1)),
                    device=self.device)
                epoch_data['jacobian_fro'] = float(jacobian.item())

            logger.log_epoch(epoch_data)

            if self.config.get("log_attn", False) and all_time_attn:
                avg_time_attn = torch.mean(torch.stack(all_time_attn), dim=0)
                attn_detached = avg_time_attn.detach().to(torch.float32).cpu()
                h, w = map(int, attn_detached.shape[-2:])  # seq_len x seq_len
                epoch_data["time_attn_shape"] = [h, w]
                epoch_data["time_attn"] = [[round(float(v), 6) for v in row] for row in attn_detached]

            print(f"Epoch: {epoch + 1} | Time: {time.time() - epoch_time:.2f}s | "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ---TRAINING SUMMARY---
        total_time = time.time() - start_time
        total_params = sum(p.numel() for p in self.model.parameters())

        # Create a final summary JSON for the entire run
        summary_data = {
            'num_parameters': total_params,
            'total_training_time_sec': total_time,
            'final_train_loss': avg_train_loss,
            'final_val_loss': avg_val_loss,
            'stopped_at_epoch': epoch
        }

        # Log and print results
        summary_path = os.path.join(self.run_output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        logger.save()
        print(f"\n{'=' * 46}\nTraining finished. Total time: {total_time:.2f}s\n{'=' * 46}\n")

        self.model_trained = self.model
        return self.model_trained
