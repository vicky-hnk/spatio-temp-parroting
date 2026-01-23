import glob
import os

import pandas as pd
import numpy as np
import pickle

def _validate_file(file_path: str):
    """
    Validates whether a file exists at the given path.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")


class LocalDataLoader:
    """
    Class to load data from local files including CSV, NPY, and HDF5 formats.
    """

    def __init__(self, dir_path: str):
        if not dir_path:
            raise ValueError("Directory path must be specified.")
        self.dir_path = dir_path

    def load_csv(self, file_name: str) -> pd.DataFrame:
        """
        Load data from a CSV file into a Pandas DataFrame.
        """
        file_path = os.path.join(self.dir_path, file_name)
        _validate_file(file_path)
        return pd.read_csv(file_path)

    def load_all_csvs(self) -> dict:
        """
        Load all CSV files in the directory into a dictionary of DataFrames.
        Keys are filenames, and values are DataFrames.
        """
        csv_files = glob.glob(os.path.join(self.dir_path, "*.csv"))
        return {os.path.basename(file): pd.read_csv(file) for file in csv_files}

    def load_npy(self, file_name: str) -> np.ndarray:
        """
        Load data from an NPY file into a NumPy array.
        """
        file_path = os.path.join(self.dir_path, file_name)
        _validate_file(file_path)
        return np.load(file_path)

    def load_hdf5(self, file_name: str) -> pd.DataFrame:
        """
        Load data from an HDF5 file into a Pandas DataFrame.
        """
        file_path = os.path.join(self.dir_path, file_name)
        _validate_file(file_path)
        return pd.read_hdf(file_path)

    def load_pkl(self, file_name: str) -> pd.DataFrame:
        """
        Load data from a pickle (.pkl) file into a list.
        """
        file_path = os.path.join(self.dir_path, file_name)
        _validate_file(file_path)

        with open(file_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        return data

    def load_parquet(self, file_name: str) -> pd.DataFrame:
        """
        Load data from a Parquet file into a Pandas DataFrame.
        """
        file_path = os.path.join(self.dir_path, file_name)
        _validate_file(file_path)
        return pd.read_parquet(file_path)
