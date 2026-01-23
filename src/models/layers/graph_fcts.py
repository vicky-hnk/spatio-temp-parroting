# -------------------------
# Graph helpers
# -------------------------
import torch
import pickle
import scipy.sparse as sp
from scipy.sparse import linalg
import numpy as np

def sym_adj(adj):
    """Symmetric normalized adjacency: D^{-1/2} A D^{-1/2}."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    """Row-normalized transition: D^{-1} A."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """L = I - D^{-1/2} A D^{-1/2}."""
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_lap = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return norm_lap

def normalized_laplacian_torch(adj: torch.Tensor, symmetrize: bool = True) -> torch.Tensor:
    """ L = I - D^{-1/2} A D^{-1/2}"""
    if adj.dim() == 2:
        adj = adj.unsqueeze(0)  # [1,N,N]
    # optional: enforce symmetry for PSD Laplacian
    adj = 0.5 * (adj + adj.transpose(-1, -2)) if symmetrize else adj
    adj = adj.to(dtype=adj.dtype)

    deg = adj.sum(-1, keepdim=True).clamp_min(1e-12)    # [B,N,1]
    inv_sqrt_deg = deg.pow(-0.5)                       # [B,N,1]
    a_norm = inv_sqrt_deg * adj * inv_sqrt_deg.transpose(1, 2)  # [B,N,N]

    batch, nodes, _ = a_norm.shape
    i = torch.eye(nodes, dtype=a_norm.dtype, device=a_norm.device).expand(batch, nodes, nodes)
    laplace = i - a_norm
    return laplace if adj.dim() == 3 else laplace.squeeze(0)


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    laplacian = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(laplacian, 1, which='LM')
        lambda_max = float(lambda_max[0])
    laplacian = sp.csr_matrix(laplacian)
    M, _ = laplacian.shape
    identity = sp.identity(M, format='csr', dtype=laplacian.dtype)
    laplacian = (2 / lambda_max * laplacian) - identity
    return laplacian.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, "rb") as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            return pickle.load(f, encoding="latin1")

def load_adj(pkl_filename, adjtype: str):
    """
    Returns (sensor_ids, sensor_id_to_ind, supports_list[np.ndarray N×N])
    adjtype in {"scalap","normlap","symnadj","transition","doubletransition","identity"}
    """
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0], dtype=np.float32))]
    else:
        raise ValueError("adj type not defined")
    return sensor_ids, sensor_id_to_ind, adj


def build_torch_supports_from_adj_pkl(pkl_path: str, node_order = None,
                                      adjtype: str = "doubletransition", device = None):
    """
    Load supports from adj_mx.pkl and return as a list of torch.FloatTensor N×N.
    If node_order is provided, will permute A to match your data column order.
    """
    sensor_ids, id2ind, supports_np = load_adj(pkl_path, adjtype)
    if node_order is not None:
        perm = [id2ind[sid] for sid in node_order]
        supports_np = [np.asarray(S)[np.ix_(perm, perm)] for S in supports_np]
    ts = [torch.tensor(S, dtype=torch.float32) for S in supports_np]
    if device is not None:
        ts = [t.to(device) for t in ts]
    return ts