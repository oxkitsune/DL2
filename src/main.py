from pathlib import Path

import matplotlib.pyplot as plt

from src.data import DataLoader, get_paths


def run():
    primary_directory = Path().resolve()  # directory where everything is stored
    provided_data_dir = primary_directory / "data"
    training_data_dir = provided_data_dir / "train-pats"
    training_plan_paths = get_paths(
        training_data_dir
    )  # gets the path of each plan's directory
    data_loader_train = DataLoader(training_plan_paths)
    data_loader_train.set_mode("training_model")
    batch = data_loader_train.get_patients(["pt_1"])

    sample_dose = batch.dose[0]
    plt.imshow(sample_dose[:, :, 40])
    plt.show()


if __name__ == "__main__":
    run()
