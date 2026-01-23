import os

import numpy as np
import torch

from src.models.layers.graph_fcts import (build_torch_supports_from_adj_pkl,asym_adj)

def _resolve_path(base_dir: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir, path)

def load_double_transition_supports_from_config(config: dict, df = None, device = None):
    data_conf = config["data_conf"]
    base_dir = data_conf.get("data_path", ".")
    pkl_path = data_conf.get("adj_mx_pkl_path")
    npy_path = data_conf.get("adj_file_name")

    if pkl_path:
        pkl_abs = _resolve_path(base_dir, pkl_path)
        if os.path.exists(pkl_abs):
            node_order = list(df.columns) if df is not None else None
            try:
                supports = build_torch_supports_from_adj_pkl(
                    pkl_path=pkl_abs,
                    node_order=node_order,
                    adjtype="doubletransition",
                    device=device
                )
                return supports
            except KeyError as e:
                raise KeyError(
                    f"Node id {e} from your dataframe wasn't found in {pkl_abs}. "
                    f"Ensure your reduced HDF5 columns are the sensor IDs used by adj_mx.pkl."
                )

    if npy_path:
        npy_abs = _resolve_path(base_dir, npy_path)
        if not os.path.exists(npy_abs):
            raise FileNotFoundError(f"Adjacency file not found at: {npy_abs}")
        A = np.load(npy_abs).astype(np.float32)
        T_fwd = np.asarray(asym_adj(A))
        T_bwd = np.asarray(asym_adj(A.T))
        return [
            torch.tensor(T_fwd, dtype=torch.float32, device=device),
            torch.tensor(T_bwd, dtype=torch.float32, device=device),
        ]

    raise FileNotFoundError("No adjacency source found. "
                            "Provide either data_conf.adj_mx_pkl_path or data_conf.adj_file_name."
    )
