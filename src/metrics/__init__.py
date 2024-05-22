import torch

ROIS = dict(
    oars=[
        "Brainstem",
        "SpinalCord",
        "RightParotid",
        "LeftParotid",
        "Esophagus",
        "Larynx",
        "Mandible",
    ],
    targets=["PTV56", "PTV63", "PTV70"],
)

ALL_ROIS = ROIS["oars"] + ROIS["targets"]

OAR_DVH_METRICS = {oar: ["D_0.1_cc", "mean"] for oar in ROIS["oars"]}
TARGET_DVH_METRICS = {target: ["D_99", "D_95", "D_1"] for target in ROIS["targets"]}

ALL_DVH_METRICS = OAR_DVH_METRICS | TARGET_DVH_METRICS


def dose_score(prediction, target, mask):
    return torch.abs(prediction - target).sum() / mask.sum()


def mean_dvh_error(prediction, batch):
    reference_dvh = dvh_score(batch["dose"], batch)
    prediction_dvh = dvh_score(prediction, batch)

    errors = {}
    for metric, roi in ALL_DVH_METRICS:
        errors[(metric, roi)] = torch.abs(
            reference_dvh[(metric, roi)] - prediction_dvh[(metric, roi)]
        )

    return torch.nanmean(torch.stack(list(errors.values())))


def dvh_score(prediction, batch):
    batch_size = prediction.shape[0]
    dvh_metrics = [
        dvh_score_for_single_prediction(
            prediction[i], batch["voxel_dimensions"][i], batch["structure_masks"][i]
        )
        for i in range(batch_size)
    ]

    dvh_metrics = {
        k: torch.stack([m[k] for m in dvh_metrics]) for k in dvh_metrics[0].keys()
    }
    return dvh_metrics


def dvh_score_for_single_prediction(prediction, voxel_dims, structure_masks):
    voxels_within_tenths_cc = torch.maximum(
        torch.Tensor([1.0, 1.0, 1.0]).to(torch.device(prediction.get_device())),
        torch.round(100.0 / voxel_dims),
    )
    metrics = {}
    for roi_index, roi in enumerate(ALL_ROIS):
        roi_mask = structure_masks[..., roi_index].to(torch.bool)
        roi_dose = prediction[roi_mask]
        roi_size = roi_dose.size(0)
        if roi_mask is None or roi_size == 0:
            continue  # Skip over ROIs when the ROI is missing (i.e., not contoured)

        for metric in ALL_DVH_METRICS[roi]:
            if metric == "D_0.1_cc":
                fractional_volume_to_evaluate = voxels_within_tenths_cc / roi_size
                metric_value = torch.quantile(roi_dose, fractional_volume_to_evaluate)
            elif metric == "mean":
                metric_value = roi_dose.mean()
            elif metric == "D_99":
                metric_value = torch.quantile(roi_dose, 0.01)
            elif metric == "D_95":
                metric_value = torch.quantile(roi_dose, 0.05)
            elif metric == "D_1":
                metric_value = torch.quantile(roi_dose, 0.99)
            else:
                raise ValueError(f"Metrics {metric} is not supported.")
            metrics[(metric, roi)] = metric_value
    return metrics
