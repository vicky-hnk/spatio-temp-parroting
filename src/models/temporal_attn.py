import torch
from torch import nn

from src.models.layers.positional_encoding import SineSPE, SPEFilter


class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        self.mode = config.get("mode", "prod")
        self.discard_last_feature = config["discard_last_feature"]
        self.num_heads = config["num_heads"]
        self.model_dim = config["model_dim"]
        self.num_features_in = config["num_features_in"]
        self.dropout_value = config.get("attn_dropout", 0.1)
        self.dropout = nn.Dropout(self.dropout_value)

        # --- Diagonal mask/drop flags ---
        self.mask_diag = config.get(
            "diag_mask", False
        )  # hard mask (sets diagonal to -inf)
        self.diag_drop = config.get("diag_drop", 0.0)  # probabilistic diagonal dropout
        self.soft_diag_penalty = config.get(
            "diag_penalty", 0.0
        )  # soft bias (negative value)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.model_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_value,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor):
        batch, nodes, seq_len, dim = x.shape
        # (B, N, T, D) -> (B*N, T, D)
        x_reshaped = x.reshape(batch * nodes, seq_len, dim)
        attn_mask = None

        # Diagonal Mask
        if self.mask_diag:
            attn_mask = torch.eye(seq_len, device=self.device, dtype=torch.bool)
        # Diagonal Dropout or Penalty
        elif self.diag_drop > 0 or self.soft_diag_penalty != 0.0:
            attn_mask = torch.zeros(seq_len, seq_len, device=self.device)
            if self.training and self.diag_drop > 0:
                drop_mask = torch.rand(seq_len, device=self.device) < self.diag_drop
                attn_mask[drop_mask, drop_mask] = float("-inf")
            # Apply a constant penalty if requested
            if self.soft_diag_penalty != 0.0:
                attn_mask += (
                    torch.eye(seq_len, device=self.device) * self.soft_diag_penalty
                )

        attn_output, attn_weights = self.attn(
            x_reshaped,
            x_reshaped,
            x_reshaped,
            attn_mask=attn_mask,
            average_attn_weights=False,
        )
        # (B*N, T, D) -> (B, N, T, D)
        x_out = attn_output.reshape(batch, nodes, seq_len, dim)
        aggregated_attn = attn_weights.view(
            batch, nodes, self.num_heads, seq_len, seq_len
        )
        toeplitz_matrix = aggregated_attn.mean(dim=(0, 1, 2))
        return x_out, toeplitz_matrix


class MultiHeadStochasticTemporalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        self.mode = config.get("mode", "prod")
        self.discard_last_feature = config["discard_last_feature"]
        self.num_heads = config["num_heads"]
        self.model_dim = config["model_dim"]
        self.num_features_in = config["num_features_in"]
        self.gated = config.get("gated", True)
        self.num_realizations = config.get("num_realizations", 256)

        # Split model dimension over heads
        if self.model_dim % self.num_heads != 0:
            raise ValueError(
                f"Model dimension ({self.model_dim}) must be divisible by num_heads ({self.num_heads})"
            )

        self.features_per_head = self.model_dim // self.num_heads

        # Initialize SineSPE and SPEFilter
        self.spe_generator = SineSPE(
            num_heads=self.num_heads,
            in_features=self.features_per_head,
            num_realizations=self.num_realizations,
            num_sines=1,
        )
        self.spe_filter = SPEFilter(
            gated=self.gated, code_shape=(self.num_heads, self.features_per_head)
        )

        # Output projection
        self.out_projection = nn.Linear(self.model_dim, self.model_dim)
        self.dropout_value = config.get("attn_dropout", 0.1)
        self.dropout = nn.Dropout(self.dropout_value)

    def forward(self, x: torch.Tensor):
        # x shape: (batch_size, nodes, time, features)
        batch, nodes, seq_len, _ = x.shape
        # Reshape to create heads: (B, N, T, D) -> (B, N, T, H, F_H)
        queries = x.view(batch, nodes, seq_len, self.num_heads, self.features_per_head)
        keys = x.view(batch, nodes, seq_len, self.num_heads, self.features_per_head)
        values = x.view(batch, nodes, seq_len, self.num_heads, self.features_per_head)

        # Permutation and Flattening for PE results in (B*N*H, T, F_H)
        queries_flat = (
            queries.permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(-1, seq_len, self.features_per_head)
        )
        # keys_flat = keys.permute(0, 3, 1, 2, 4).contiguous().view(-1, seq_len, self.features_per_head)

        # Generate positional encoding codes for each head, (1, T, H, F_H, num_realizations)
        qbar_full, kbar_full = self.spe_generator.forward(
            (queries_flat.shape[0], seq_len)
        )
        # -> (B*N, T, H, F_H)
        queries_pe = queries.view(
            batch * nodes, seq_len, self.num_heads, self.features_per_head
        )
        keys_pe = keys.view(
            batch * nodes, seq_len, self.num_heads, self.features_per_head
        )
        # -> (B*N, T, H, num_realizations)
        queries_hat, keys_hat = self.spe_filter.forward(
            queries_pe, keys_pe, (qbar_full, kbar_full)
        )

        queries_hat = torch.clamp(queries_hat, min=-5.0, max=5.0)
        keys_hat = torch.clamp(keys_hat, min=-5.0, max=5.0)

        # Reshape the outputs of the SPE filter for the attention calculation
        # (B*N, T, H, num_realizations) -> (B*N*H, T, num_realizations)
        queries_hat = (
            queries_hat.permute(0, 2, 1, 3)
            .contiguous()
            .view(-1, seq_len, self.num_realizations)
        )
        keys_hat = (
            keys_hat.permute(0, 2, 1, 3)
            .contiguous()
            .view(-1, seq_len, self.num_realizations)
        )

        # (B, N, T, H, F_H) -> (B*N*H, T, F_H)
        values_flat = (
            values.permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(-1, seq_len, self.features_per_head)
        )

        attended_output_flat = None
        toeplitz_matrix = None

        if self.mode == "prod":
            # Linear Complexity (O(L)) approach: Q(KV)
            kv_product = torch.matmul(keys_hat.transpose(-2, -1), values_flat)
            attended_output_flat = torch.matmul(queries_hat, kv_product)
        elif self.mode == "research":
            # Quadratic Complexity (O(L^2)) approach: (QK)V
            attn_scores = torch.matmul(queries_hat, keys_hat.transpose(-2, -1))
            attn = torch.softmax(attn_scores, dim=-1)
            attn = self.dropout(attn)  # (B*N*H, T, T)
            attended_output_flat = torch.matmul(attn, values_flat)

            # Aggregate attention weights for a single Toeplitz matrix
            aggregated_attn = attn_scores.view(
                batch, nodes, self.num_heads, seq_len, seq_len
            )
            toeplitz_matrix = aggregated_attn.mean(dim=(0, 1, 2))
        else:
            raise ValueError("Mode must be 'prod' or 'research'.")

        # Correctly reshape attended output and apply final projection, (B*N*H, T, F_H) -> (B, N, T, H, F_H)
        attended_output = attended_output_flat.view(
            batch, nodes, seq_len, self.num_heads, self.features_per_head
        )

        # Concatenate heads and apply final output projection, (B, N, T, H, F_H) -> (B, N, T, D)
        final_output = attended_output.view(
            batch, nodes, seq_len, self.num_heads * self.features_per_head
        )
        x_updated = self.out_projection(final_output)  # (B, N, T, D)

        return x_updated, toeplitz_matrix
