import math

import torch
from torch import nn
import torch.nn.functional as func

from src.models.layers.positional_encoding import SineSPE, SPEFilter

class GraphAttentionLayer(nn.Module):
    """
    Layer to compute attention among timely patches of each node in the graph.
    torch.einsum is used to speed up training.
    As activation functions LeakyReLU is used to maintain nonzero attention
    scores, while ELU is used to stabilize the final feature representations.
    """

    def __init__(self, in_channels, out_channels, dropout=0.4,
                 patch_size: int = 24, alpha: float = 0.2,
                 filter_strategy: str = None, k: int = None):
        """
        :param patch_size: number of time steps for one attention patch
        :param in_channels: number of input features
        :param out_channels: number of output features
        """
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.dropout = dropout
        self.alpha = alpha
        if filter_strategy not in ['scale', 'sparsify', None]:
            raise ValueError(
                f'filter_strategy must None, mask or sparsify not '
                f'{filter_strategy}')
        self.filter_strategy = filter_strategy
        self.k = k

        # Initialize weight matrix
        self.W = nn.Parameter(torch.empty(
            size=(self.in_channels, self.out_channels)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Initialize weight matrix for learnable attention scaling
        self.W_c = nn.Parameter(torch.empty(self.out_channels))
        nn.init.uniform_(self.W_c, a=-0.1, b=0.1)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def _top_k_sparsify(self, attention):
        """
        Retains only the top-k values per node while setting others to zero.
        """
        top_k_values, top_k_indices = torch.topk(attention, self.k, dim=2)
        mask = torch.zeros_like(attention)
        # Set top-k indices to 1
        mask.scatter_(2, top_k_indices, 1)
        return attention * mask

    def forward(self, x, adj):
        """
        x: (batch_size, time_steps, num_nodes, in_features)
        adj: (num_nodes, num_nodes)  # Weighted adjacency matrix (0 to 1)
        """
        batch_size, time_steps, num_nodes, _ = x.shape

        if time_steps % self.patch_size != 0:
            raise ValueError(
                f"Invalid time_steps={time_steps}: The number of time steps "
                f"must be divisible by the patch_size={self.patch_size}. "
            )
        num_patches = time_steps // self.patch_size  # n = T/P

        # Create patches (B, n, N, P, C)
        x_patched = x.view(batch_size, num_patches, num_nodes, self.patch_size,
                           self.in_channels)

        # Feature embedding
        h_patched = torch.matmul(x_patched, self.W)  # (B, n, N, P, C)
        e_ij_list = []
        for n in range(num_patches):
            h_patched_n = h_patched[:, n, :, :, :]  # Select nth patch
            print(f"Processing patch {n} with shape: {h_patched_n.shape}")
            e_ij_n = torch.einsum("bipc, bjpc -> bijpc", h_patched_n,
                                  h_patched_n)  # (B, N, N, P, C)
            e_ij_n = self.leaky_relu(e_ij_n)
            # Apply weighted adjacency to ensure only connected node
            # contribute to attention score (1,N,N,1,1)
            adj_expanded = adj.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            e_ij_n = e_ij_n * adj_expanded.to(self.device)  # (B, N, N, P, C)
            # Normalize per patch
            e_ij_n = func.softmax(e_ij_n, dim=-1)  # Softmax across nodes
            e_ij_list.append(e_ij_n)  # Store per-patch attention scores
        e_ij = torch.stack(e_ij_list, dim=4)  # (B, N, N, P, n, C)
        attention = e_ij.reshape(batch_size, num_nodes, num_nodes,
                                 time_steps, self.out_channels)

        if self.filter_strategy == 'scale':
            attention = self.W_c * attention
            # Apply softmax normalization over neighbors (j)
            attention = torch.softmax(attention, dim=2)
        elif self.filter_strategy == 'sparsify':
            attention = self._top_k_sparsify(attention)
        # Update feature embedding
        print(attention.shape, h_patched.shape)
        # (B, n, N, P, F) -> (B, N, P, n, C)
        h_patched = h_patched.permute(0, 2, 1, 3, 4)
        h_patched = h_patched.reshape(batch_size, num_nodes, time_steps,
                                      self.out_channels)  # (B, N, T, C)
        h_prime = torch.einsum("bnqtc, bntc -> bntc", attention,
                               h_patched)

        return attention, func.elu(h_prime)  # (B, N, N, T, C), (B, N, T, C)


class TemporalAttentionLayer(nn.Module):
    def __init__(self, model_dim, num_features, num_heads, gated: bool = True):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.query_projection = nn.Linear(model_dim, model_dim)
        self.key_projection = nn.Linear(model_dim, model_dim)
        self.value_projection = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(0.1)
        # Initialize SineSPE and SPEFilter
        self.spe_generator = SineSPE(num_heads=self.num_heads, in_features=model_dim,
            num_realizations=256, num_sines=1)
        self.spe_filter = SPEFilter(gated=gated, code_shape=(1, model_dim))

    def forward(self, x):
        # x shape: (batch_size, nodes, time, features)
        batch, nodes, seq_len, features = x.shape

        # Project queries, keys, and values
        queries = self.query_projection(x).view(batch * nodes, seq_len, features)
        keys = self.key_projection(x).view(batch * nodes, seq_len, features)
        values = self.value_projection(x).view(batch * nodes, seq_len, features)

        # Generate positional encoding codes, (B*N, T)
        pe_codes = self.spe_generator.forward((batch * nodes, seq_len))
        # The spe_filter returns transformed queries and keys
        queries_pe, keys_pe = self.spe_filter.forward(queries, keys, pe_codes)

        # (B*N, T, D) @ (B*N, D, T) -> (B*N, T, T)
        attn_scores = torch.einsum("btr,bkr->btk", queries_pe, keys_pe) / math.sqrt(self.model_dim)
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        # (B*N, T, T) @ (B*N, T, D) -> (B*N, T, D)
        attended_output = torch.einsum("bnm,bnd->bnd", attn_scores, values)
        attended_output = attended_output.view(batch, nodes, seq_len, features)
        return attended_output, attn
