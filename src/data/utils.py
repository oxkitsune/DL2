import os
from pathlib import Path
from typing import Optional, Union, Any

import numpy as np
import pandas as pd


def load_file(file_path: Path) -> Union[Any, dict[str, Any]]:
    """
    Load a file in one of the formats provided in the OpenKBP dataset
    """
    if file_path.stem == "voxel_dimensions":
        return np.loadtxt(file_path)

    loaded_file_df = pd.read_csv(file_path, index_col=0)
    if loaded_file_df.isnull().values.any():  # Data is a mask
        loaded_file = np.array(loaded_file_df.index).squeeze()
    else:  # Data is a sparse matrix
        loaded_file = {
            "indices": loaded_file_df.index.values,
            "data": loaded_file_df.data.values,
        }

    return loaded_file


def get_paths(directory_path: Path, extension: Optional[str] = None) -> list[Path]:
    """
    Get the paths of every file contained in `directory_path` that also has the extension `extension` if one is provided.
    """
    all_paths = []

    if not directory_path.is_dir():
        pass
    elif extension is None:
        dir_list = os.listdir(directory_path)
        for name in dir_list:
            if "." != name[0]:  # Ignore hidden files
                all_paths.append(directory_path / str(name))
    else:
        data_root = Path(directory_path)
        for file_path in data_root.glob("*.{}".format(extension)):
            file_path = Path(file_path)
            if "." != file_path.stem[0]:
                all_paths.append(file_path)

    return all_paths
