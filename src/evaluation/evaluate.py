from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from src.data import DataLoader, DataBatch, get_paths
from numpy.typing import NDArray
from tqdm import tqdm
from pathlib import Path


def evaluate(model: nn.Module, data_loader: DataLoader, prediction_dir: Path):
    predict(model, data_loader)
    prediction_paths = get_paths(prediction_dir, extension="csv")
    prediction_loader = DataLoader(prediction_paths)
    dose_evaluator = DoseEvaluator(data_loader, prediction_loader)
    dose_evaluator.evaluate()
    dvh_score, dose_score = dose_evaluator.get_scores()
    return dvh_score, dose_score

def sparse_vector_function(x, indices=None) -> dict[str, NDArray]:
    """Convert a tensor into a dictionary of the non-zero values and their corresponding indices
    :param x: the tensor or, if indices is not None, the values that belong at each index
    :param indices: the raveled indices of the tensor
    :return:  sparse vector in the form of a dictionary
    """
    if indices is None:
        y = {"data": x[x > 0], "indices": np.nonzero(x.flatten())[-1]}
    else:
        y = {"data": x[x > 0], "indices": indices[x > 0]}
    return y

def store_prediction(pred, batch):
    t = pred * batch.possible_dose_mask
    dose_pred = np.squeeze(t)
    dose_to_save = sparse_vector_function(dose_pred)
    dose_df = pd.DataFrame(data=dose_to_save["data"].squeeze(), index=dose_to_save["indices"].squeeze(), columns=["data"])
    (patient_id,) = batch.patient_list
    dose_df.to_csv(f"results/{patient_id}.csv")

def predict(model: nn.Module, data_loader: DataLoader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    for batch in tqdm(data_loader.get_batches(), total=len(data_loader)):
        features = batch.get_flattend_oar_features()
        input = torch.Tensor(features).transpose(1, 4).to(device)

        target = batch.get_target()
        target = torch.Tensor(target).transpose(1, 4).to(device)

        pred = model(input)
        store_prediction(pred.transpose(1, 4).detach().cpu().numpy(), batch)

class DoseEvaluator:
    """Evaluate a full dose distribution against the reference dose on the OpenKBP competition metrics"""

    def __init__(self, reference_data_loader: DataLoader, prediction_loader: Optional[DataLoader] = None):
        self.reference_data_loader = reference_data_loader
        self.prediction_loader = prediction_loader

        # Initialize objects for later
        self.reference_batch: Optional[DataBatch] = None
        self.prediction_batch: Optional[DataBatch] = None

        # Define evaluation metrics for each roi
        oar_dvh_metrics = {oar: ["D_0.1_cc", "mean"] for oar in self.reference_data_loader.rois["oars"]}
        target_dvh_metrics = {target: ["D_99", "D_95", "D_1"] for target in self.reference_data_loader.rois["targets"]}
        self.all_dvh_metrics = oar_dvh_metrics | target_dvh_metrics

        # Make data frames to cache evaluation metrics
        metric_columns = [(m, roi) for roi, metrics in self.all_dvh_metrics.items() for m in metrics]
        self.dose_errors = pd.Series(index=self.reference_data_loader.patient_id_list, data=None, dtype=float)
        self.dvh_metric_differences_df = pd.DataFrame(index=self.reference_data_loader.patient_id_list, columns=metric_columns)
        self.reference_dvh_metrics_df = self.dvh_metric_differences_df.copy()
        self.prediction_dvh_metrics_df = self.dvh_metric_differences_df.copy()

    def evaluate(self):
        """Calculate the  dose and DVH scores for the "new_dose" relative to the "reference_dose"""
        if not self.reference_data_loader.patient_paths:
            raise ValueError("No reference patient data was provided, so no metrics can be calculated")
        if self.prediction_loader:
            Warning("No predicted dose loader was provided. Metrics were only calculated for the reference dose.")
        self._set_data_loader_mode()

        for self.reference_batch in self.reference_data_loader.get_batches():
            self.reference_dvh_metrics_df = self._calculate_dvh_metrics(self.reference_dvh_metrics_df, self.reference_dose)

            self.prediction_batch = self.prediction_loader.get_patients([self.patient_id]) if self.prediction_loader else None
            if self.predicted_dose is not None:
                patient_dose_error = np.sum(np.abs(self.reference_dose - self.predicted_dose)) / np.sum(self.possible_dose_mask)
                self.dose_errors[self.patient_id] = patient_dose_error
                self.prediction_dvh_metrics_df = self._calculate_dvh_metrics(self.prediction_dvh_metrics_df, self.predicted_dose)

    def get_scores(self) -> tuple[NDArray, NDArray]:
        dose_score = np.nanmean(self.dose_errors)
        dvh_errors = np.abs(self.reference_dvh_metrics_df - self.prediction_dvh_metrics_df)
        dvh_score = np.nanmean(dvh_errors.values)
        return dose_score, dvh_score

    def _set_data_loader_mode(self) -> None:
        self.reference_data_loader.set_mode("evaluation")
        if self.prediction_loader:
            self.prediction_loader.set_mode("predicted_dose")

    def _calculate_dvh_metrics(self, metric_df: pd.DataFrame, dose: NDArray) -> pd.DataFrame:
        """
        Calculate the DVH values that were used to evaluate submissions in the competition.
        :param metric_df: A DataFrame with columns indexed by the metric name and the structure name
        :param dose: the dose to be evaluated
        :return: the same metric_df that is input, but now with the metrics for the provided dose
        """
        voxels_within_tenths_cc = np.maximum(1, np.round(100 / self.voxel_size))
        for roi in self.reference_data_loader.full_roi_list:
            roi_mask = self.get_roi_mask(roi)
            if roi_mask is None:
                continue  # Skip over ROIs when the ROI is missing (i.e., not contoured)
            roi_dose = dose[roi_mask]
            for metric in self.all_dvh_metrics[roi]:
                if metric == "D_0.1_cc":
                    roi_size = len(roi_dose)
                    fractional_volume_to_evaluate = 100 - voxels_within_tenths_cc / roi_size * 100
                    metric_value = np.percentile(roi_dose, fractional_volume_to_evaluate)
                elif metric == "mean":
                    metric_value = roi_dose.mean()
                elif metric == "D_99":
                    metric_value = np.percentile(roi_dose, 1)
                elif metric == "D_95":
                    metric_value = np.percentile(roi_dose, 5)
                elif metric == "D_1":
                    metric_value = np.percentile(roi_dose, 99)
                else:
                    raise ValueError(f"Metrics {metric} is not supported.")
                metric_df.at[self.patient_id, (metric, roi)] = metric_value

        return metric_df

    def get_roi_mask(self, roi_name: str) -> Optional[NDArray]:
        roi_index = self.reference_batch.get_index_structure_from_structure(roi_name)
        mask = self.reference_batch.structure_masks[:, :, :, :, roi_index].astype(bool)
        flat_mask = mask.flatten()
        return flat_mask if any(flat_mask) else None

    @property
    def patient_id(self) -> str:
        patient_id, *_ = self.reference_batch.patient_list if self.reference_batch.patient_list else [None]
        return patient_id

    @property
    def voxel_size(self) -> NDArray:
        return np.prod(self.reference_batch.voxel_dimensions)

    @property
    def possible_dose_mask(self) -> NDArray:
        return self.reference_batch.possible_dose_mask

    @property
    def reference_dose(self) -> NDArray:
        return self.reference_batch.dose.flatten()

    @property
    def predicted_dose(self) -> NDArray:
        return self.prediction_batch.predicted_dose.flatten()
