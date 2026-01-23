import torch
from torch import nn
import torch.nn.functional as func

from src.models.layers.graph_fcts import normalized_laplacian_torch


def safe_check(tensor, name):
    if torch.isnan(tensor).any():
        print(f"⚠️ NaN found in {name} | shape: {tensor.shape}")
        return True
    if torch.isinf(tensor).any():
        print(f"⚠️ Infinity found in {name} | shape: {tensor.shape}")
        return True
    return False

def normalized_dirichlet_energy(x, adj, reduction="mean", symmetrize=True):
    """Returns scalar tensor (on x.device), by default mean over batch.
    Gets the Rayleigh Quotient."""
    batch, nodes, seq, dim = x.shape
    features = seq * dim

    if adj.dim() == 2:
        adj = adj.unsqueeze(0).expand(batch, -1, -1)
    x = x.reshape(batch, nodes, features).to(torch.float32)

    adj = adj.to(x.dtype)
    if symmetrize:
        adj = 0.5 * (adj + adj.transpose(-1, -2))
    deg = adj.sum(-1)
    inv_sqrt_deg = (deg + 1e-12).rsqrt()
    a_sym = adj * inv_sqrt_deg.unsqueeze(-1) * inv_sqrt_deg.unsqueeze(-2)  # D^{-1/2} A D^{-1/2}
    lx = x - a_sym @ x  # (I - A_sym) x
    num = (x * lx).sum(dim=(1, 2))  # x^T L x
    den = (x * x).sum(dim=(1, 2)).clamp_min(1e-12)  # x^T x
    r = num / den
    return r.mean() if reduction == "mean" else r

def dirichlet_energy_unnorm(x, adj, reduction="mean"):
    batch, nodes, seq, dim = x.shape
    features = seq * dim

    if adj.dim() == 2:
        adj = adj.unsqueeze(0).expand(batch, -1, -1)
    # (optional) force undirected: A = (A + A^T)/2
    adj = 0.5 * (adj + adj.transpose(-1, -2))
    x = x.reshape(batch, nodes, features).to(torch.float32)
    deg = adj.sum(-1)
    # Lx = D x - A x
    Lx = deg.unsqueeze(-1) * x - torch.matmul(adj.to(x.dtype), x)
    e_per_batch = (x * Lx).sum(dim=(1, 2))
    return e_per_batch.mean() if reduction == "mean" else e_per_batch

def _make_k_activation(name: str):
    name = (name or "hardtanh").lower()
    if name == "hardtanh":
        return nn.Hardtanh(min_val=0.0, max_val=1.0)  # exact clamp [0,1]
    elif name == "sigmoid":
        # smooth, strictly (0,1)
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported K_activation: {name}")

class GraphAdvection(nn.Module):
    """
    Implements the Graph Advection Layer from the paper "Advection Diffusion
    Reaction Graph Neural Networks for Spatio-Temporal Data".

    This layer computes learnable, directed edge weights to transport node features
    across the graph, simulating an advection process. It is designed to be a
    component within a larger Graph Neural Network architecture.
    """

    def __init__(self, model_dim, hidden_dim, edge_feature_dim, step_size:float):
        super().__init__()

        # These fully connected layers create the edge weight generation mechanism,
        # as detailed in Algorithm 2 of the paper[cite: 93, 96]. They learn to produce
        # directional edge weights from the features of the nodes connected by an edge.
        self.A1 = nn.Linear(model_dim, hidden_dim)
        self.A2 = nn.Linear(model_dim, hidden_dim)
        self.A3 = nn.Linear(hidden_dim, edge_feature_dim)
        self.A4 = nn.Linear(edge_feature_dim, model_dim)
        self.relu = nn.ReLU()
        self.h = float(step_size)

    def forward(self, x, adj):
        adj = adj.to(x.device)
        batch, num_nodes, time, model_dim = x.shape
        channels_flat = time * model_dim  # Flatten time and feature dims
        x_flat = x.view(batch, num_nodes, channels_flat)

        edge_list = adj.nonzero(as_tuple=False) # batch index, source, target
        num_edges = edge_list.size(0)
        batch_idx, sender_idx, receiver_idx = edge_list[:, 0], edge_list[:, 1], edge_list[:, 2]
        sender_idx, batch_idx, receiver_idx = sender_idx.to(x_flat.device), batch_idx.to(x_flat.device), receiver_idx.to(x_flat.device)
        x_sender= x_flat[batch_idx, sender_idx] # (E, T*D) -- sender's features
        x_receiver = x_flat[batch_idx, receiver_idx] # (E, T*D) -- receiver's features

        # Flatten time and batch edges: (E * time, model_dim)
        x_sender_flat = x_sender.view(num_edges * time, model_dim)
        x_receiver_flat = x_receiver.view(num_edges * time, model_dim)

        # Algorithm 2 - Step 1: Compute edge features
        z_ij = self.A3(self.relu(self.A1(x_sender_flat) + self.A2(x_receiver_flat)))
        z_ji = self.A3(self.relu(self.A1(x_receiver_flat) + self.A2(x_sender_flat)))
        # v_ji_pre = self.A4(self.relu(z_ji - z_ij))

        # Gather the incoming edges (different to paper)
        v_ij_pre = self.A4(self.relu(z_ij - z_ji)).view(num_edges, time, model_dim)
        v_ij_pre = v_ij_pre.view(num_edges, time, model_dim)

        # Row-softmax over neighbors of each sender (mass-conservation of outgoing information)
        sender_nodes_flat = (batch_idx * num_nodes + sender_idx).long()
        index_broadcast = sender_nodes_flat.view(-1, 1, 1).expand_as(v_ij_pre)

        out_shape = (batch * num_nodes, time, model_dim)
        max_logits = torch.full(out_shape, -float('inf'), device=v_ij_pre.device, dtype=v_ij_pre.dtype)
        max_logits.scatter_reduce_(0, index_broadcast, v_ij_pre, reduce='amax', include_self=True)
        sender_max_on_edges = max_logits.index_select(0, sender_nodes_flat)

        # Subtract maximum for stable softmax
        exp_centered = torch.exp(v_ij_pre - sender_max_on_edges).type_as(v_ij_pre)
        denominator = torch.zeros_like(max_logits)
        denominator.scatter_reduce_(0, index_broadcast, exp_centered, reduce='sum', include_self=True)
        denominator_edges = denominator.index_select(0, sender_nodes_flat)
        v_ij = exp_centered / (denominator_edges + 1e-12)
        safe_check(v_ij, "v_ij")

        # Equation 4: Apply the Graph Advection Operator, send mass forward, aggregate at receivers
        messages = v_ij * x_sender.view(num_edges, time, model_dim)  # [E, T, D]
        messages_flat = messages.view(num_edges, channels_flat)

        receiver_nodes_flat = (batch_idx * num_nodes + receiver_idx).long()
        inbound = torch.zeros(batch * num_nodes, channels_flat, device=x.device, dtype=messages.dtype)
        inbound.index_add_(0, receiver_nodes_flat, messages_flat)
        inbound = inbound.view(batch, num_nodes, time, model_dim)

        divergence = inbound - x
        x_updated = x + self.h * divergence
        x_updated = x_updated.view( batch, num_nodes, time, model_dim)

        return x_updated, v_ij.detach()

class GraphDiffusion(nn.Module):
    def __init__(self, model_dim, step_size:float, use_k: bool = False,
                 k_activation: str = "hardtanh"):
        super().__init__()
        self.use_k = use_k
        self.h = step_size
        # Global diffusion coefficient alpha ∈ [0, 1/(2h)]
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        self.scale_alpha = 1.0 / (2.0 * self.h)
        if use_k:
            # Per-channel diffusion gates K_i ∈ [0, 1]
            self.k_raw = nn.Parameter(torch.zeros(model_dim))
            self.k_act = _make_k_activation(k_activation)
        else:
            self.register_parameter("k_raw", None)
            self.k_act = None

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        batch, _, _, model_dim = x.shape
        if laplacian.dim() == 2:
            laplacian = laplacian.unsqueeze(0).expand(batch, -1, -1)  # [B,N,N]
        else:
            laplacian = laplacian
        # α in [0, 1/(2h)]
        alpha = self.scale_alpha * torch.sigmoid(self.alpha_raw)
        # Optional per-channel gates K ∈ [0,1]
        if self.use_k:
            k = self.k_act(self.k_raw).view(1, 1, 1, model_dim)  # [1,1,1,D]
            diff_input = x * k
        else:
            diff_input = x
        # L @ X  (broadcast matmul over [B,N,N] x [B,N,T,D] -> [B,N,T,D])
        diff_term = torch.einsum('bij,bjtd->bitd', laplacian, diff_input)
        # Explicit Euler update
        x_new = x - self.h * alpha * diff_term
        return x_new


class GraphReaction(nn.Module):
    def __init__(self, model_dim, step_size:float, activation = 'relu',
                 dropout: float = 0.0, reaction_mode: str = 'simple'):
        super().__init__()
        self.h = step_size
        self.mode = reaction_mode
        self.act = nn.ReLU() if activation == "relu" else nn.GELU()

        if reaction_mode == "simple":
            self.mlp = nn.Sequential(
                nn.Linear(model_dim, model_dim),
                self.act,
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(model_dim, model_dim))

        elif reaction_mode == "adr":
            self.R1 = nn.Linear(model_dim, model_dim)
            self.R2 = nn.Linear(model_dim, model_dim)
            self.R3 = nn.Linear(model_dim, model_dim)
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        else:
            raise ValueError(f"Unknown mode: {reaction_mode}")

    def forward(self, x, u0):
        if self.mode == "simple":
            out = self.mlp(x)
            return x + self.h * out

        else:
            # ADR-style: additive + multiplicative + skip(U0)
            batch, nodes, seq, features = x.shape
            x_flat = x.view(batch * nodes * seq, features)
            u0_flat = u0.view(batch * nodes * seq, features)
            add = self.R1(x_flat)
            mul = torch.tanh(self.R2(x_flat)) * x_flat
            skip = self.R3(u0_flat)
            out = self.act(add + mul + skip)
            out = self.dropout(out)
            out = out.view(batch, nodes, seq, features)
            return x + self.h * out


class GraphADRBlock(nn.Module):
    def __init__(self, advection: nn.Module = None, diffusion: nn.Module = None, reaction: nn.Module = None,
                 compute_laplacian: bool = True, use_u0_skip: bool = True):
        super().__init__()
        self.advection = advection
        self.diffusion = diffusion
        self.reaction = reaction
        self.compute_laplacian = compute_laplacian
        self.use_u0_skip = use_u0_skip

    def forward(self, x: torch.Tensor, adj: torch.Tensor, laplacian = None):
        u0 = x.detach() if self.use_u0_skip else None
        batch, num_nodes, time, model_dim = x.shape

        if laplacian is None:
            if not self.compute_laplacian:
                raise ValueError("Laplacian not provided but compute_laplacian=False.")
            laplacian = normalized_laplacian_torch(adj)  # [B, N, N]
        else:
            laplacian = laplacian if laplacian.dim() == 3 else laplacian.unsqueeze(0).expand(batch, -1, -1)

        # --- Advection ---
        if self.advection:
            x, adv_attn = self.advection(x=x, adj=adj) # x has shape (B, N, T, D)
        else:
            x, adv_attn = x, None

        # --- Diffusion ---
        if self.diffusion:
            x = self.diffusion(x=x, laplacian=laplacian) # x has shape (B, N, T, D)
        else:
            x = x

        # --- Reaction ---
        if self.reaction:
            x_out = self.reaction(x=x, u0=u0) # x_out has shape (B, N, T, D)
        else:
            x_out = x

        dirichlet = normalized_dirichlet_energy(x=x_out, adj=adj)
        return x_out, adv_attn, dirichlet


class AdaptiveGCN(nn.Module):
    """
    Implements the Adaptive Spatial GCN layer, it learns its own two-directional adjacency matrix from the data.
    """

    def __init__(self, in_channels, out_channels, num_nodes,
                 order=2, embedding_dim=10, dropout=0.1):
        """
        Args:
            c_in (int): Input channels.
            c_out (int): Output channels.
            num_nodes (int): Number of nodes (N).
            order (int): The max hop-distance to explore (e.g., 2 means A*X and A^2*X).
            embedding_dim (int): Dimension for learned node embeddings.
            dropout (float): Dropout rate.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.order = order
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.source_embedding = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        self.target_embedding = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        self.softmax = nn.Softmax(dim=1)
        mlp_in_channels = (self.order * 2 + 1) * self.in_channels
        self.mlp = nn.Linear(mlp_in_channels, out_channels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        """
        batch, num_nodes, t, channels = x.shape
        assert num_nodes == self.num_nodes, "Input node dimension N must match num_nodes"
        out = [x]
        forward_adj = func.relu(torch.matmul(self.source_embedding, self.target_embedding.T))
        forward_adj = self.softmax(forward_adj)  # (N, N)
        backward_adj = func.relu(torch.matmul(self.target_embedding, self.source_embedding.T))
        backward_adj = self.softmax(backward_adj)  # (N, N)
        adj_list = [forward_adj, backward_adj]
        x_permuted = x.permute(0, 2, 1, 3) # shape (B, T, N, D)
        for a in adj_list:  # Loop 1 (forward), Loop 2 (backward)
            x_gcn = torch.einsum('nn, btnc -> btnc', a, x_permuted) # shape (B, T, N, D)
            out.append(x_gcn.permute(0, 2, 1, 3))
            # Loop for 2nd order and higher
            for k in range(2, self.order + 1):
                x_gcn_next = torch.einsum('nn, btnc -> btnc', a, x_gcn) # shape (B, T, N, D)
                out.append(x_gcn_next.permute(0, 2, 1, 3))
                x_gcn = x_gcn_next # shape (B, T, N, D)

        h = torch.cat(out, dim=3) #  (B, T, N, D*order*support_len)
        h = self.mlp(h)
        output = func.dropout(h, self.dropout, training=self.training)

        dirichlet_fwd = normalized_dirichlet_energy(x=output, adj=forward_adj)
        dirichlet_bwd = normalized_dirichlet_energy(x=output, adj=backward_adj)
        dirichlet_avg = (dirichlet_fwd + dirichlet_bwd) / 2.0

        return output, None, dirichlet_avg
