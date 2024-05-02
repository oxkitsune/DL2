from pathlib import Path
from typing import Optional, TypeVar

import matplotlib.pyplot as plt

from src.data import DataLoader, get_paths

T = TypeVar("T")


def unwrap(var: Optional[T]) -> T:
    if var is None:
        raise Exception("Could not unwrap")

    return var


def run():
    primary_directory = Path().resolve()
    provided_data_dir = primary_directory / "data"
    training_data_dir = provided_data_dir / "train-pats"
    training_plan_paths = get_paths(training_data_dir)
    data_loader_train = DataLoader(training_plan_paths)
    data_loader_train.set_mode("training_model")
    batch = data_loader_train.get_patients(["pt_1"])

    sample_in_batch = 0
    axis_sample = 50

    sample_structure_masks = unwrap(batch.structure_masks)[sample_in_batch]
    batch_features = batch.get_features(ptv_index=7)
    sample_features = batch_features[sample_in_batch]
    sample_oars = sample_features[:, :, :, 1]

    for i in range(7):
        print(unwrap(batch.structure_mask_names)[i])
        plt.imshow(sample_structure_masks[:, :, axis_sample, i])
        plt.show()

    print("Squished OARs")
    plt.imshow(sample_oars[:, :, axis_sample])
    plt.show()


if __name__ == "__main__":
    run()
