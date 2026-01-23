from typing import List, Tuple, Optional

import torch
from torch import nn

from src.models.layers.positional_encoding import TAPE
from src.models.temporal_attn import MultiHeadStochasticTemporalAttention, MultiHeadTemporalAttention
from src.models.gnn import AdaptiveGCN


def safe_check(tensor, name):
    if torch.isnan(tensor).any():
        print(f"⚠️ NaN found in {name} | shape: {tensor.shape}")
        return True
    if torch.isinf(tensor).any():
        print(f"⚠️ Infinity found in {name} | shape: {tensor.shape}")
        return True
    return False


class TimeSpaceBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gnn_block_mode = config.get("gnn_block_mode", "full")
        self.temporal_module = config.get("temporal_module", "temp_attn")
        self.without_residual = config.get("without_residual", False)

        if self.temporal_module == "temp_attn":
            self.tmp = MultiHeadStochasticTemporalAttention(config=config)
        elif self.temporal_module == "simple_temp_attn":
            self.tmp = MultiHeadTemporalAttention(config=config)
        else:
            raise RuntimeError(
                f"Invalid temporal_module '{self.temporal_module}'. "
                "This indicates a configuration or implementation error.")

        if self.gnn_block_mode == "Conv":
            self.mp = AdaptiveGCN(in_channels=config["model_dim"], out_channels=config["model_dim"],
                                  num_nodes=config["num_nodes"])
        else:
            raise RuntimeError(
                f"Invalid graph model '{self. self.gnn_block_mode}'.")

        if not callable(self.tmp):
            raise TypeError(f"Temporal module is not callable: {type(self.tmp)}")

        if not callable(self.mp):
            raise TypeError(f"GNN block is not callable: {type(self.mp)}")

        # --- Residual conditioning from temporal features ---
        self.temporal_bias = nn.Linear(1, config["model_dim"])  # bias after TMP
        self.spatial_bias = nn.Linear(1, config["model_dim"])  # bias after MP

        self.temp_norm = nn.LayerNorm(config['model_dim'])
        self.space_norm = nn.LayerNorm(config['model_dim'])
        dropout = config.get('dropout', 0.1)
        self.temporal_dropout = nn.Dropout(dropout)
        self.spatial_dropout = nn.Dropout(dropout)

    def forward(self, x_dyn, x_temp, adj) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # x shape: (batch_size, nodes, time, model_dim)

        if self.tmp is not None:
            x_tmp_out, time_attn = self.tmp(x_dyn) # x has shape (B, N, T, D)
            temporal_bias_tmp = self.temporal_bias(x_temp)  # (B,N,T,D_dyn)
            if self.without_residual:
                x_tmp_res = self.temp_norm(self.temporal_dropout(x_tmp_out + temporal_bias_tmp))
            else:
                x_tmp_res = self.temp_norm(x_dyn + self.temporal_dropout(x_tmp_out + temporal_bias_tmp))
        else:
            x_tmp_res = x_dyn
            time_attn = None

        x_mp_out, space_attn, dirichlet = self.mp(x_tmp_res, adj) # (B, N, T, D)
        temporal_bias_space = self.spatial_bias(x_temp)  # (B,N,T,D_dyn)
        x_updated = self.space_norm(x_tmp_res + self.spatial_dropout(x_mp_out + temporal_bias_space))

        safe_check(x_tmp_res, 'x_tmp_res')
        safe_check(x_updated, 'x_updated')

        return x_updated, dirichlet, space_attn, time_attn


class TimeSpaceAttnModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        self.depth = config["depth"]
        self.num_features_in = config["num_features_in"]
        self.num_features_out = config["num_features_out"]
        self.model_dim = config["model_dim"]
        self.seq_len = config["seq_len"]
        self.pred_len = config["pred_len"]
        self.input_projection = nn.Linear(self.num_features_in-1, self.model_dim)
        self.output_projection = nn.Linear(self.model_dim+1, self.num_features_out)
        self.forecast_projection = nn.Linear(self.seq_len, self.pred_len)
        self.stmp = nn.ModuleList([TimeSpaceBlock(config) for _ in range(self.depth)])

        self.use_global_pe = config.get('use_global_pe', False)
        self.positional_encoding = None
        if self.use_global_pe == "tape":
            self.positional_encoding = TAPE(model_dim=self.model_dim, max_seq_len= self.seq_len,
                                            dropout=config.get('dropout', 0.1))
        elif self.use_global_pe in (None, False, "none"):
            self.positional_encoding = None
        else:
            raise ValueError(
                f"Unknown use_global_pe='{self.use_global_pe}'. Expected 'tape' or None/'none'.")


    def forward(self, x, adj) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # x has shape (B, T, N, C)
        x = x.permute(0, 2, 1, 3)
        x_dyn = x[..., :-1]  # [B,N,T,D_dyn]
        x_temp = x[..., -1:]  # [B,N,T,C]
        x_dyn_proj = self.input_projection(x_dyn)  # (B, N, T, D)

        if self.positional_encoding is not None:
            x_dyn_proj = self.positional_encoding(x_dyn_proj)

        space_attn, time_attn = None, None
        dirichlet_list = []
        for i, stmp_block in enumerate(self.stmp):
            safe_check(x_dyn_proj, f"x_proj{i}")
            x_updated, dirichlet, space_attn, time_attn = stmp_block(x_dyn_proj, x_temp, adj)
            dirichlet_list.append(dirichlet)
            x_dyn_proj = x_updated
        x_cat = torch.cat([x_dyn_proj, x_temp], dim=-1) # x_Cat  has shape(B, N, T, num_features_in])
        x_reduced = self.output_projection(x_cat).permute(0, 1, 3, 2) # -> (B, N, F, T)
        x_out = self.forecast_projection(x_reduced).permute(0, 3, 1, 2) # Final projection to (B, pred_len, N, F)
        return x_out, dirichlet_list, space_attn, time_attn
