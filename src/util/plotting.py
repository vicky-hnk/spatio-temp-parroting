import csv
import os
import re

import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd




def plot_attn_files(folder, outdir):
    outdir_freq = os.path.join(outdir, 'freq')
    outdir_batch = os.path.join(outdir, 'batch')
    os.makedirs(outdir_freq, exist_ok=True)
    os.makedirs(outdir_batch, exist_ok=True)
    for file in os.listdir(folder):
        if file.endswith(".pt"):
            attn = torch.load(os.path.join(folder, file)).numpy()
            attn = np.abs(attn)

            assert attn.ndim == 4, f"Expected (B,N,N,F), got {attn.shape}"

            for f in range(attn.shape[3]):
                matrix = attn.mean(axis=0)[:, :, f]  # mean over B
                fig, ax = plt.subplots()
                im = ax.imshow(matrix, cmap="viridis")
                fig.colorbar(im, ax=ax)
                ax.set_xlabel("Node i index")
                ax.set_ylabel("Node j index")
                ax.set_title(f"{file}: Mean over batches for Frequency bin {f}")
                fig.tight_layout()
                fig.savefig(os.path.join(outdir_freq, f"{file}_mean_b_f{f}.png"), dpi=200)
                plt.close(fig)

            # --- One plot (mean over B and F) ---

            mat_bf = attn.mean(axis=(0, 3))
            fig, ax = plt.subplots()
            im = ax.imshow(mat_bf, cmap="viridis")
            fig.colorbar(im, ax=ax)
            ax.set_xlabel("Node i index")
            ax.set_ylabel("Node j index")
            ax.set_title(f"{file}: Mean over batches and frequencies")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir_batch, f"{file}_mean_b_f.png"), dpi=200)
            plt.close(fig)

def plot_delta(folder, outdir="../../figures/delta_x", weighted=True):
    os.makedirs(outdir, exist_ok=True)

    # For weighted mean across batches: accumulate sums & counts per epoch
    sum_by_epoch = {}
    cnt_by_epoch = {}

    # For unweighted mean (each batch counts equally)
    means_by_epoch = {}

    for file in os.listdir(folder):
        if not file.endswith(".pt"):
            continue

        # filename: delta_epoch_0000_YYYYMMDD_HHMMSS.pt
        root, _ = os.path.splitext(file)
        parts = root.split("_")
        if len(parts) < 3 or parts[0] != "delta" or parts[1] != "epoch":
            continue
        try:
            epoch = int(parts[2])
        except ValueError:
            continue

        obj = torch.load(os.path.join(folder, file), map_location="cpu")
        if isinstance(obj, tuple) and len(obj) == 2:
            _, pct_sym = obj
        elif isinstance(obj, dict) and "pct_sym" in obj:
            pct_sym = obj["pct_sym"]
        else:
            pct_sym = obj
        if pct_sym is None:
            continue

        vals = pct_sym.detach().float().reshape(-1)
        vals = vals[torch.isfinite(vals)]
        if vals.numel() == 0:
            continue

        if weighted:
            sum_by_epoch[epoch] = sum_by_epoch.get(epoch, 0.0) + vals.sum().item()
            cnt_by_epoch[epoch] = cnt_by_epoch.get(epoch, 0) + int(vals.numel())
        else:
            means_by_epoch.setdefault(epoch, []).append(vals.mean().item())

    # Compute epoch means
    if weighted and sum_by_epoch:
        epochs = sorted(sum_by_epoch.keys())
        means = [sum_by_epoch[e] / max(cnt_by_epoch[e], 1) for e in epochs]
    elif not weighted and means_by_epoch:
        epochs = sorted(means_by_epoch.keys())
        means = [sum(means_by_epoch[e]) / len(means_by_epoch[e]) for e in epochs]
    else:
        print("No valid pct_sym found.")
        return

    # --- plot (NO second plt.figure call) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, means, "-o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("pct_sym (%)")  # already scaled by your 200 * ... formula
    ax.set_title("Average x vs x_real similarity per epoch")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    png_path = os.path.join(outdir, "delta_avg.png")
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Saved figure to {png_path}")

    # --- csv ---
    df = pd.DataFrame({"epoch": epochs, "mean": means})
    csv_path = os.path.join(outdir, "delta_avg.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")


def heatmaps_delta(folder, outdir="../../figures/delta_x", node_axis=0):

    os.makedirs(outdir, exist_ok=True)
    def extract_epoch(fname):
        root, _ = os.path.splitext(fname)
        parts = root.split("_")
        if len(parts) >= 3 and parts[0] == "delta" and parts[1] == "epoch":
            try:
                return int(parts[2])
            except ValueError:
                return None
        return None

    def get_pct_sym(obj):
        if isinstance(obj, tuple) and len(obj) == 2:
            return obj[1]
        if isinstance(obj, dict) and "pct_sym" in obj:
            return obj["pct_sym"]
        return obj

    # Gather files and find last epoch
    files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    epochs = [e for f in files if (e := extract_epoch(f)) is not None]
    if not epochs:
        print("No epoch-encoded .pt files found.")
        return
    last_epoch = max(epochs)

    sums_all = None
    cnt_all = 0
    sums_last = None
    cnt_last = 0
    N_nodes = None

    for f in files:
        epoch = extract_epoch(f)
        if epoch is None:
            continue

        obj = torch.load(os.path.join(folder, f), map_location="cpu")
        pct_sym = get_pct_sym(obj)
        if pct_sym is None:
            continue

        t = pct_sym.detach().float()  # shape (B, T, N)
        if t.dim() != 3:
            print(f"Skipping {f}: expected 3D (B,T,N), got {tuple(t.shape)}")
            continue

        # mask non-finite just in case
        t = torch.where(torch.isfinite(t), t, torch.tensor(0.0, dtype=t.dtype))

        # reduce over B and T → per-node vector (N,)
        per_node = t.mean(dim=(0, 1))  # (N,)
        N = per_node.numel()
        if N_nodes is None:
            N_nodes = N
            sums_all = torch.zeros(N, dtype=torch.float32)
            sums_last = torch.zeros(N, dtype=torch.float32)
        elif N != N_nodes:
            print(f"Skipping {f}: node count {N} != expected {N_nodes}")
            continue

        sums_all += per_node
        cnt_all += 1

        if epoch == last_epoch:
            sums_last += per_node
            cnt_last += 1

    if cnt_all == 0:
        print("No valid pct_sym data found.")
        return

    avg_all = (sums_all / max(cnt_all, 1)).numpy()
    avg_last = (sums_last / max(cnt_last, 1)).numpy() if cnt_last > 0 else np.zeros_like(avg_all)

    # Save CSV
    df = pd.DataFrame({
        "node": np.arange(N_nodes, dtype=int),
        "avg_all_epochs": avg_all,
        "avg_last_epoch": avg_last,
    })
    csv_path = os.path.join(outdir, "per_node_similarity_B_T_N.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved per-node CSV to {csv_path}")

    # Heatmap A: all epochs/batches
    fig1, ax1 = plt.subplots(figsize=(10, 2))
    im1 = ax1.imshow(avg_all[np.newaxis, :], aspect="auto")
    ax1.set_yticks([])
    ax1.set_xlabel("Node index")
    ax1.set_title("Per-node avg similarity (pct_sym %) — ALL epochs & batches")
    c1 = fig1.colorbar(im1, ax=ax1)
    c1.set_label("pct_sym (%)")
    fig1.tight_layout()
    png_all = os.path.join(outdir, "heatmap_per_node_all_epochs_B_T_N.png")
    fig1.savefig(png_all, dpi=200)
    plt.close(fig1)
    print(f"Saved heatmap to {png_all}")

    # Heatmap B: last epoch only
    fig2, ax2 = plt.subplots(figsize=(10, 2))
    im2 = ax2.imshow(avg_last[np.newaxis, :], aspect="auto")
    ax2.set_yticks([])
    ax2.set_xlabel("Node index")
    ax2.set_title(f"Per-node avg similarity (pct_sym %) — LAST epoch (epoch {last_epoch})")
    c2 = fig2.colorbar(im2, ax=ax2)
    c2.set_label("pct_sym (%)")
    fig2.tight_layout()
    png_last = os.path.join(outdir, "heatmap_per_node_last_epoch_B_T_N.png")
    fig2.savefig(png_last, dpi=200)
    plt.close(fig2)
    print(f"Saved heatmap to {png_last}")


def plot_nodes_by_similarity(latlon_csv, pernode_csv, value_col="avg_all_epochs",
                             out_png=None, side_by_side=False, vmin=None,
                             vmax=None, point_size=40):

    # Load data
    pernode = pd.read_csv(pernode_csv)
    loc = pd.read_csv(latlon_csv)

    # Harmonize key column names
    if "node" in loc.columns:
        key_left, key_right = "node", "node"
    else:
        # uploaded file has 'index' as the node index
        key_left, key_right = "node", "index"

    # Merge
    df = pernode.merge(loc, left_on=key_left, right_on=key_right, how="inner")

    # Check required columns
    required_cols = {"latitude", "longitude"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"latlon CSV must contain columns {required_cols}, got {loc.columns.tolist()}")

    # Helper to compute scale
    def _compute_scale(values, vmin_in, vmax_in):
        vals = values.to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 0.0, 1.0
        lo = np.percentile(vals, 2) if vmin_in is None else vmin_in
        hi = np.percentile(vals, 98) if vmax_in is None else vmax_in
        if not np.isfinite(lo): lo = np.nanmin(vals)
        if not np.isfinite(hi): hi = np.nanmax(vals)
        if hi <= lo:
            hi = lo + 1e-6
        return lo, hi

    # Helper to plot a single figure
    def _plot_one(data, col, outpath):
        vals = data[col]
        lo, hi = _compute_scale(vals, vmin, vmax)
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(data["longitude"], data["latitude"],
                        c=vals, s=point_size, edgecolors="none", vmin=lo, vmax=hi)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title(f"Nodes colored by {col}")
        ax.grid(True, linestyle="--", alpha=0.4)
        # Aspect correction by median latitude
        if len(data):
            mid_lat = np.deg2rad(np.median(data["latitude"]))
            if np.cos(mid_lat) > 1e-6:
                ax.set_aspect(1.0/np.cos(mid_lat))
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("pct_sym (%)")
        fig.tight_layout()
        # Save
        if outpath is None:
            base = os.path.splitext(pernode_csv)[0]
            outpath = f"{base}_map_{col}.png"
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        return outpath

    # Decide outputs
    if side_by_side:
        # Two separate plots/files
        out1 = None if out_png is None else os.path.splitext(out_png)[0] + "_all.png"
        out2 = None if out_png is None else os.path.splitext(out_png)[0] + "_last.png"
        p1 = _plot_one(df, "avg_all_epochs", out1)
        p2 = _plot_one(df, "avg_last_epoch", out2)
        return {"all_epochs_png": p1, "last_epoch_png": p2}
    else:
        # Single plot for requested column
        if value_col not in df.columns:
            raise ValueError(f"{value_col} not found in per-node CSV columns: {df.columns.tolist()}")
        p = _plot_one(df, value_col, out_png)
        return {"png": p}

def plot_attn_geo_edges(latlon_csv,
                        attn=None,        # torch.Tensor (B,N,N,F) or (B,N,N) or (N,N)
                        W=None,           # numpy array (N,N) if you already averaged
                        symmetrize=True,
                        topk_global=800,  # keep strongest K edges overall (set None to skip)
                        topk_per_node=None,  # OR strongest k per row (overrides global if set)
                        cmap="viridis",
                        node_color="black",
                        node_size=12,
                        edge_alpha_min=0.25,
                        edge_alpha_max=0.95,
                        edge_w_min=0.4,
                        edge_w_max=2.8,
                        title="Geospatial attention (edges colored by weight)",
                        out_png=None):
    """
    Colors/thickens edges by attention weight; nodes are uniform.
    latlon_csv must have columns: ['index','latitude','longitude'] aligned with node indices 0..N-1.
    """

    # --- locations / positions ---
    loc = pd.read_csv(latlon_csv).sort_values("index").reset_index(drop=True)
    N_loc = len(loc)
    pos = {i: (loc.iloc[i]["longitude"], loc.iloc[i]["latitude"]) for i in range(N_loc)}

    # --- get W (N,N) ---
    if W is None:
        if attn is None:
            raise ValueError("Provide either W (N,N) or attn (B,N,N[,F]).")
        if torch.is_tensor(attn):
            A = attn.detach().float()
        else:
            A = torch.as_tensor(attn, dtype=torch.float32)
        if A.dim() == 4:
            # (B,N,N,F) -> mean over B and F
            W = A.mean(dim=(0, 3)).cpu().numpy()
        elif A.dim() == 3:
            # (B,N,N) -> mean over B
            W = A.mean(dim=0).cpu().numpy()
        elif A.dim() == 2:
            W = A.cpu().numpy()
        else:
            raise ValueError(f"Unsupported attention dims: {tuple(A.shape)}")
    else:
        W = np.asarray(W, dtype=float)

    # ensure square, finite, zero diagonal
    if W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square, got {W.shape}")
    N = W.shape[0]
    if N != N_loc:
        raise ValueError(f"Nodes in W ({N}) != rows in latlon_csv ({N_loc})")
    W = np.where(np.isfinite(W), W, 0.0)
    np.fill_diagonal(W, 0.0)
    if symmetrize:
        W = 0.5 * (W + W.T)

    # --- select edges to draw ---
    edges = []
    if topk_per_node is not None:
        k = int(topk_per_node)
        for i in range(N):
            row = W[i]
            if k >= N:
                idx = np.argsort(row)[::-1]
            else:
                idx = np.argpartition(row, -k)[-k:]
                idx = idx[np.argsort(row[idx])[::-1]]
            for j in idx:
                w = row[j]
                if w > 0 and i != j:
                    edges.append((i, j, float(w)))
    elif topk_global is not None:
        K = int(min(topk_global, N * (N - 1)))
        flat = W.flatten()
        # remove diagonal indices
        mask = ~np.kron(np.eye(N, dtype=bool), np.ones(N, dtype=bool)).ravel()
        vals = flat[mask]
        if vals.size:
            sel = np.argpartition(vals, -K)[-K:]
            sel = sel[np.argsort(vals[sel])[::-1]]
            flat_idx = np.nonzero(mask)[0][sel]
            for f in flat_idx:
                i, j = divmod(f, N)
                w = W[i, j]
                if w > 0 and i != j:
                    edges.append((i, j, float(w)))
    else:
        # draw all positive edges (can be dense!)
        ii, jj = np.where(W > 0)
        edges = [(int(i), int(j), float(W[i, j])) for i, j in zip(ii, jj) if i != j]

    if not edges:
        print("No edges selected.")
        return None

    # --- normalize weights for color/alpha/width ---
    ew = np.array([w for _, _, w in edges])
    vmin = np.percentile(ew, 2)
    vmax = np.percentile(ew, 98)
    if vmax <= vmin:
        vmax = vmin + 1e-9
    norm = Normalize(vmin=vmin, vmax=vmax)
    wn = norm(ew)  # 0..1
    edge_colors = plt.get_cmap(cmap)(wn)
    edge_alphas = edge_alpha_min + (edge_alpha_max - edge_alpha_min) * wn
    edge_widths = edge_w_min + (edge_w_max - edge_w_min) * wn

    # --- build graph & plot ---
    G = nx.Graph() if symmetrize else nx.DiGraph()
    G.add_nodes_from(range(N))
    G.add_weighted_edges_from([(u, v, w) for (u, v, w) in edges])

    fig, ax = plt.subplots(figsize=(10, 8))

    # draw edges one by one to support per-edge alpha
    for (u, v, _), c, a, w in zip(G.edges(data=True), edge_colors, edge_alphas, edge_widths):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               edge_color=[c], width=w,
                               alpha=a, arrows=not symmetrize, ax=ax)

    # draw nodes uniformly
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, ax=ax)

    # geo-ish aspect
    mid_lat = np.deg2rad(loc["latitude"].median())
    if np.cos(mid_lat) > 1e-6:
        ax.set_aspect(1.0/np.cos(mid_lat))
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title(title)

    # colorbar for edges
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.75)
    cbar.set_label("Attention weight (mean over B,F)")

    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"Saved {out_png}")
        return out_png
    return fig, ax

if __name__ == "__main__":

    run_name = "GATLinear-CycleNet-METRLA-20250818_174437"

    attn_folder = os.path.join("..", "..", "src", "logs", run_name, "attn")
    # plot_attn_files(attn_folder, outdir=os.path.join("..", "..", "figures", run_name))

    # plot_delta(folder=os.path.join("..", "..", "src", "logs", run_name, "delta"),
               # outdir=os.path.join("..", "..", "figures", run_name, "delta_x"))

    # heatmaps_delta(folder=os.path.join("..", "..", "src", "logs", run_name, "delta"),
               # outdir=os.path.join("..", "..", "figures", run_name, "delta_x"))

    latlon_csv = os.path.join("..", "..", "data", "data_raw", "METR-LA", "graph_sensor_locations.csv")
    pernode_csv = os.path.join("..", "..", "figures", run_name, "delta_x", "per_node_similarity_B_T_N.csv")
    plot_nodes_by_similarity(latlon_csv, pernode_csv, value_col="avg_all_epochs",
                             out_png=os.path.join("..", "..", "figures", run_name, "delta_x",
                                                   "node_similarity_map.png"))

