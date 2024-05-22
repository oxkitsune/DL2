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
    voxel_dims = batch["voxel_dimensions"]

    voxels_within_tenths_cc = torch.maximum(1, torch.round(100 / voxel_dims))
    metrics = {}
    for roi_index, roi in enumerate(ALL_ROIS):
        roi_mask = batch["structure_masks"][..., roi_index].to(torch.bool)
        roi_dose = prediction[roi_mask]
        for metric in ALL_DVH_METRICS[roi]:
            if metric == "D_0.1_cc":
                roi_size = len(roi_dose)
                fractional_volume_to_evaluate = (
                    100 - voxels_within_tenths_cc / roi_size * 100
                )
                metric_value = torch.percentile(roi_dose, fractional_volume_to_evaluate)
            elif metric == "mean":
                metric_value = roi_dose.mean()
            elif metric == "D_99":
                metric_value = torch.percentile(roi_dose, 1)
            elif metric == "D_95":
                metric_value = torch.percentile(roi_dose, 5)
            elif metric == "D_1":
                metric_value = torch.percentile(roi_dose, 99)
            else:
                raise ValueError(f"Metrics {metric} is not supported.")
            metrics[(metric, roi)] = metric_value
    return metrics
