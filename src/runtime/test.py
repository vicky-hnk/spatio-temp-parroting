import torch
from src.util.train_utils import Metrics

def _to_tensor(x, device, dtype=torch.float32):
    t = torch.as_tensor(x)
    # Only convert if needed to avoid extra copies
    if t.dtype != dtype or t.device != device:
        t = t.to(device=device, dtype=dtype)
    return t

def _prepare_supports(adj, device):
    """
    Return (supports_list, is_list) where supports_list is List[Tensor] on device.
    Does not change ranks/shapes beyond wrapping in a list if needed.
    """
    if isinstance(adj, (list, tuple)):
        return [_to_tensor(a, device) for a in adj], True
    return [_to_tensor(adj, device)], False

def _prepare_single_adj(adj, device):
    """
    Return a single Tensor adjacency on device.
    If given a list/tuple, pick the first (common convention).
    """
    if isinstance(adj, (list, tuple)):
        adj = adj[0]
    return _to_tensor(adj, device)


class Tester:
    def __init__(self, trained_model, config, get_data_fn):
        self.trained_model = trained_model
        self.config = config
        self.get_data_fn = get_data_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def get_test_data(self):
        test_set, test_loader = self.get_data_fn(flag="test")
        print(f"--- Test data loaded. Number of test batches: {len(test_loader)} ---")
        return test_set, test_loader

    @torch.no_grad()
    def test(self):
        # --- LOAD TEST DATA ---
        test_set, test_loader = self.get_test_data()
        self.trained_model.eval()

        preds_unscaled = []
        trues_raw = []
        supports_device = None
        first_batch = True

        # Dataloader must yield: (x, y_scaled, y_raw, adj, mask)
        for _, (test_batch_x, _, test_batch_y_raw, test_adj, _) in enumerate(test_loader):
            test_batch_x = test_batch_x.float().to(self.device)

            adj_device = _prepare_single_adj(test_adj, device=self.device)
            y_pred = self.trained_model(test_batch_x, adj_device)[0]

            test_outputs_scaled = y_pred
            batch_size, horizon, nodes, channels = test_outputs_scaled.shape

            # Move N to last dim for the scaler and flatten B,H,C together: [B,H,C,N] -> [-1, N]
            pred_2d = (test_outputs_scaled.permute(0, 1, 3, 2).reshape(-1, nodes).detach().cpu().numpy())
            # Inverse-transform over nodes
            pred_unscaled_2d = test_set.inverse_transform(pred_2d)  # [-1, N]
            # Back to BHNC
            pred_unscaled = (torch.from_numpy(pred_unscaled_2d).float() .view(
                batch_size, horizon, channels, nodes).permute(0, 1, 3, 2))
            preds_unscaled.append(pred_unscaled)
            trues_raw.append(test_batch_y_raw)

        # Concatenate all batch results
        y_pred_t = torch.cat(preds_unscaled, dim=0)
        y_true_t = torch.cat(trues_raw, dim=0)

        # --- METRICS (aggregate) ---
        print("\n--- Calculating Final Metrics ---")
        metrics_instance = Metrics(metric_flags=['masked_mae', 'masked_rmse', 'masked_mape'])
        # Now we compare UN-SCALED predictions to RAW ground truth
        results = metrics_instance.calculate_metrics(values_pred=y_pred_t, values_true=y_true_t)
        return results
