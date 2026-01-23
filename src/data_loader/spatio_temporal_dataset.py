"""
This module contains a Python class to load spatio-temporal datasets,
including an adjacency matrix representing spatial relationships and
time-series data for each node in the network.

The datasets must adhere to the following dimensionality requirements:
- **Number of Timestamps x Number of Nodes x Number of Features**

The adjacency matrix should represent weighted edges, defining spatial
dependencies between nodes. This module is designed for datasets where
each node has the same number of features across all timestamps.

We use the term sequence length (seq_len) for the input sequence length,
often referred to as the window size and prediction length (pred_len) for
the output sequence length, often referred to as the horizon.
"""
import os

import torch
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.data_loader.local_data_loader import LocalDataLoader
from src.util.exceptions import InsufficientDataError
from src.models.layers.graph_fcts import (calculate_scaled_laplacian, sym_adj, asym_adj, calculate_normalized_laplacian)


class SpatioTemporalDataset(Dataset):
    """
    A Pytorch class for loading and processing spatio-temporal datasets, with
    support for adjacency matrices and time-series data stored in separate
    files.

    ### Requirements:
    - **Adjacency Matrix**:
        - File must be `.csv` or `.npy`.
        - Represents spatial relationships between nodes with weighted edges.
    - **Time-Series Data**:
        - File must be `.csv`, `.npy`, or `.h5`.
        - Data shape:
        (Number of Timestamps x Number of Nodes).
        - Every node must have the same number of features across all
        timestamps and across nodes.

    ### Attributes:
    ----------
    source_type (str): The type of the data source.
            E.g., 'local', 'remote', etc.
        file_name (str): The name of the file containing the dataset.
        adj_file_name (str): The name of the file containing the adjacency
        matrix.
        data_type (str): The format of the dataset file.
            E.g., 'hdf5', 'csv', 'npy', etc.
        adj_type (str): The format of the adjacency matrix file.
            E.g., 'npy', 'csv'.
        data_path (str): The path to the directory containing the dataset files.
        target (str): The target variable for prediction.
            E.g., 'temperature', 'speed', etc.
        flag (str): The mode of operation.
            E.g., 'train', 'validation', 'test'. Defaults to 'train'.
        features (str): Specifies the features to use.
            'M' for multiple, 'S' for single, 'MS' for multiple with a
            specific single target.
            Defaults to 'MS'.
        scale (str): Whether to scale the data.
            'True' to apply scaling, 'False' otherwise. Defaults to "False".
        adj_norm (str): Whether to normalize the adjacency matrix.
            'True' to apply normalization, 'False' otherwise.
            Defaults to "False".
        time_enc (int): Time encoding type.
            0 for no encoding, other values for specific encoding strategies.
            Defaults to 0.
        log_encoding (bool): Whether to log encode time-based data.
            Defaults to False.
        data_split (tuple, optional): Ratios for splitting the dataset into
        train, validation, and test sets.
            Format: (train_ratio, val_ratio, test_ratio). Defaults to None.
    """

    def __init__(self, config: dict = None, flag: str = 'train', scaler=None):

        # Set configurations
        self.config = config
        self.flag = flag

        # Data Source Settings
        raw_path = config.get("data_conf", {}).get("data_path", None)
        if raw_path is None:
            raise ValueError("No data path provided.")
        self.data_path = os.path.abspath(os.path.expanduser(raw_path))
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Resolved path does not exist: {self.data_path}")
        self.loader = LocalDataLoader(self.data_path)
        self.file_name = config.get("data_conf").get("file_name", None)
        self.adj_file_name = config.get("data_conf").get("adj_file_name", None)
        self.data_type = config.get("data_conf").get("data_type", None)
        self.adj_type = config.get("data_conf").get("adj_type", None)

        # Sequence Lengths
        self.seq_len = config.get("seq_len")
        self.pred_len = config.get("pred_len")

        # Data settings
        self.data_split = config.get("data_split", None)
        if self.data_split is None:
            self.data_split = [0.7, 0.1, 0.2]
        self.slice_start = self.slice_end = None
        self.train_portion = self.data_split[0]
        self.val_portion = self.data_split[1]
        self.test_portion = self.data_split[2]
        if flag not in ['train', 'val', 'test']:
            raise ValueError("Flag should be 'train', 'val', or 'test'")
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        # Normalization, Scaling, Encoding
        self.adj_norm_method = config.get("adj_norm", "False")
        self.adj_scale = config.get("adj_scale", 1)
        self.adj_sparsify = config.get("adj_sparsify", 0.1)

        self.scale = config.get("scale", None)
        if scaler is None:
            if self.scale == "Standard":
                self.scaler = StandardScaler()
            elif self.scale == "MinMax":
                self.scaler = MinMaxScaler()
            else:
                self.scaler = None
                print("No scaling applied")
            self._owns_scaler = True
        else:
            self.scaler = scaler
            self._owns_scaler = False

        print("Applying scaling: ", self._owns_scaler)

        # Data Preparation
        self.__read_data__()


    def load_data(self):
        # Load data set
        if self.data_type == "csv":
            data = self.loader.load_csv(file_name=self.file_name)
        elif self.data_type == "npy":
            data = self.loader.load_npy(self.file_name)
        elif self.data_type == "hdf5":
            data = self.loader.load_hdf5(self.file_name)
        elif self.data_type == "parquet":
            data = self.loader.load_parquet(self.file_name)
        elif self.data_type == "pkl":
            data = self.loader.load_pkl(self.file_name)
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
        # Load the adjacency matrix
        if self.adj_type == "csv":
            adj_matrix = self.loader.load_csv(self.adj_file_name).values
        elif self.adj_type == "npy":
            adj_matrix = self.loader.load_npy(self.adj_file_name)
        elif self.adj_type == "hdf5":
            adj_matrix = self.loader.load_hdf5(self.adj_file_name)
        elif self.adj_type == "parquet":
            adj_matrix = self.loader.load_parquet(self.adj_file_name)
        elif self.adj_type == "pkl":
            adj_matrix = self.loader.load_pkl(self.adj_file_name)
        else:
            raise ValueError(
                f"Unsupported adjacency matrix type: {self.adj_type}")

        return data, adj_matrix

    def get_split(self):
        total_length = self.raw_df.shape[0]
        train_end = int(self.train_portion * total_length)
        val_start = train_end
        val_end = val_start + int(self.val_portion * total_length)
        test_start = val_end
        test_end = test_start + int(self.test_portion * total_length)
        split_start = [0, val_start, test_start]
        split_end = [train_end, val_end, test_end]
        self.slice_start, self.slice_end = (split_start[self.set_type],
                                            split_end[self.set_type])


    def __read_data__(self):
        self.raw_df, self.adj = self.load_data()
        print("Raw data shape is: ", self.raw_df.shape, type(self.raw_df))
        if not isinstance(self.raw_df, pd.DataFrame) or not isinstance(self.raw_df.index, pd.DatetimeIndex):
            raise TypeError("Data must be a pandas DataFrame with a DatetimeIndex.")

        # DEFINE SPLITS
        self.get_split()

        # HANDLE SCALER FITTING
        if self.scaler is not None and self._owns_scaler:
            X_train = self.raw_df.iloc[:int(self.train_portion * len(self.raw_df))].values  # [T_train, N]
            print("Scaling on data with shape: ", X_train.shape)
            if hasattr(self.scaler, "fit"):
                self.scaler.fit(X_train)
            else:
                raise ValueError("Scaler must implement fit/transform/inverse_transform.")

        # SELECT THE CORRECT DATA SLICE
        data_slice = self.raw_df.iloc[self.slice_start:self.slice_end]
        main_data = data_slice.values  # [T_slice, N]

        # -- TRANSFORM --
        if self.scaler is not None:
            main_data = self.scaler.transform(main_data)  # [T_slice, N]

        # reshape to [T_slice, N, 1] for model I/O
        main_data = main_data.reshape(len(main_data), -1, 1).astype(np.float32)

        self.raw_slice_for_masking = data_slice.values.reshape(len(data_slice), -1, 1)

        # FEATURE ENGINEERING & CONCATENATION
        feature_list = [main_data]
        if self.config.get("add_time_in_day", True):
            time_idx = data_slice.index
            time_in_day = (time_idx.hour * 3600 + time_idx.minute * 60 + time_idx.second) / 86400.0
            time_feature = np.tile(time_in_day.values.reshape(-1, 1, 1), (1, main_data.shape[1], 1))
            feature_list.append(time_feature)

        self.processed_data = np.concatenate(feature_list, axis=-1).astype(np.float32)

        # NORMALIZE ADJACENCY MATRIX
        adj_norm_method = self.adj_norm_method

        if self.adj_norm_method == "scaled_laplacian":
            print("Applying scaled Laplacian normalization to adjacency matrix.")
            self.adj = [calculate_scaled_laplacian(self.adj)]  # Return as list for consistency
        elif self.adj_norm_method == "symmetric":
            print("Applying symmetric normalization to adjacency matrix.")
            self.adj = [sym_adj(self.adj)]
        elif self.adj_norm_method == "asymmetric":
            print("Applying asymmetric (transition) normalization to adjacency matrix.")
            self.adj = [asym_adj(self.adj)]
        elif self.adj_norm_method == "double_transition":
            print("Applying double transition normalization to adjacency matrix.")
            self.adj = [asym_adj(self.adj), asym_adj(self.adj.T)]
        elif self.adj_norm_method == "normalized_laplacian":
            print("Applying normalized Laplacian to adjacency matrix.")
            self.adj = [calculate_normalized_laplacian(self.adj)]
        else:
            print("No GNN-specific normalization applied to adjacency matrix.")
            pass

        self.data_x = self.processed_data
        self.data_y = main_data
        self.data_y_raw = self.raw_df

        if self.data_x.shape[0] - self.seq_len - self.pred_len <= 1:
            raise InsufficientDataError(self.seq_len, self.pred_len,
                                        self.data_x.shape[0])

    def __getitem__(self, index):
        x_start = index
        x_end = x_start + self.seq_len
        y_start = x_end
        y_end = y_start + self.pred_len

        seq_x = self.data_x[x_start:x_end]
        seq_y = self.data_y[y_start:y_end]
        seq_y_raw = self.raw_slice_for_masking[y_start:y_end]  # [H, N, 1] raw mph
        seq_y_mask = (seq_y_raw != 0.0).astype(np.float32)

        return (
            torch.FloatTensor(seq_x),
            torch.FloatTensor(seq_y),
            torch.FloatTensor(seq_y_raw),
            self.adj,
            torch.FloatTensor(seq_y_mask)
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if not self.scaler: return data
        return self.scaler.inverse_transform(data)
