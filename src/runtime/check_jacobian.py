import torch


@torch.no_grad()
def _first_batch(dataloader):
    it = iter(dataloader)
    batch = next(it)
    # unpack common 3-tuple: (x, y, adj)
    # adjust to your dataset structure if different
    if isinstance(batch, (tuple, list)) and len(batch) >= 3:
        x = batch[0]
        adj = batch[3]
    else:
        raise RuntimeError("Unexpected dataloader batch structure.")
    return x, adj

def _slice_first_sample(t, expect_batch_dim: bool):
    """
    If tensor `t` has a batch dimension (dim>=3 for adj, dim>=1 for x) and expect_batch_dim=True,
    slice the first item along dim 0. Otherwise return as-is.
    """
    if expect_batch_dim and t.dim() >= 3 and t.size(0) > 1:
        return t[0:1].contiguous()

def get_jacobian(model, dataloader, probes: int = 2, device = None):
    """
    Estimate Frobenius norm of Jacobian dy/dx via Hutchinson.
    Returns a scalar tensor (fp32) on `device`.
    """
    model_device = next(model.parameters()).device
    device = device or model_device

    # --- get one sample ---
    x, adj = _first_batch(dataloader)
    # --- slice first sample for all inputs ---
    x = x[0:1].to(device=device, dtype=torch.float32).contiguous().requires_grad_(True)
    if type(adj) is torch.Tensor:
        if adj.dim() == 2:
            adj = adj.to(device=device).contiguous()  # (N,N)
        elif adj.dim() >= 3:
            adj = _slice_first_sample(adj.to(device=device), expect_batch_dim=True)
        else:
            raise RuntimeError(f"Unexpected adj shape {tuple(adj.shape)}; need (N,N) or batched.")

    model.eval()
    est = torch.zeros((), device=device, dtype=torch.float32)

    # Do Jacobian math in fp32 regardless of AMP mode
    with torch.amp.autocast(device_type='cuda', enabled=False):

        out = model(x, adj)
        y = out[0] if isinstance(out, (tuple, list)) else out

        for k in range(max(1, probes)):
            # Rademacher probe
            r = torch.empty_like(y, dtype=torch.float32).bernoulli_(0.5).mul_(2.0).add_(-1.0)
            v = (y * r).sum()
            (gx,) = torch.autograd.grad(
                v, x, retain_graph=(k < probes - 1), create_graph=False, allow_unused=False)
            est = est + gx.pow(2).sum()

    fro = (est / max(1, probes)).sqrt()
    return fro.detach()