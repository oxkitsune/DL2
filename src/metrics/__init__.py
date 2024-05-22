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

ALL_DVH_METRICS = {**OAR_DVH_METRICS, **TARGET_DVH_METRICS}


def dose_score(prediction, target, mask):
    return torch.abs(prediction - target).sum() / mask.sum()


def mean_dvh_error(prediction, target, voxel_dim, structure_masks):
    errors = _dvh_error(prediction, target, voxel_dim, structure_masks)
    return torch.nanmean(torch.stack(list(errors.values())))


def _dvh_error(prediction, target, voxel_dim, structure_masks):
    batch_size = prediction.shape[0]
    print("Structure masks shape:", structure_masks.shape)
    reference_dvh_metrics = [
        dvh_score_for_single_prediction(
            target[i].cpu(), voxel_dim[i].cpu(), structure_masks[i].cpu().clone()
        )
        for i in range(batch_size)
    ]
    pred_dvh_metrics = [
        dvh_score_for_single_prediction(
            prediction[i].cpu(), voxel_dim[i].cpu(), structure_masks[i].cpu().clone()
        )
        for i in range(batch_size)
    ]

    dvh_metrics = {
        (metric, roi): torch.stack(
            [
                torch.nanmean(
                    torch.abs(reference[roi][metric] - pred_dvh_metrics[i][roi][metric])
                )
                for _ in range(len(reference_dvh_metrics))
            ]
        )
        for i, reference in enumerate(reference_dvh_metrics)
        for roi in reference.keys()
        for metric in reference[roi].keys()
    }
    return dvh_metrics


def dvh_score_for_single_prediction(prediction, voxel_dims, structure_masks):
    device = torch.device("cpu")  # Force CPU for debugging
    voxels_within_tenths_cc = torch.maximum(
        torch.tensor([1.0, 1.0, 1.0], device=device),
        torch.round(100.0 / voxel_dims),
    )
    metrics = {k: {} for k in ALL_ROIS}
    for roi_index, roi in enumerate(ALL_ROIS):
        if roi_index == 6:
            continue
        print(f"ROI: {roi}, Index: {roi_index}")

        if roi_index >= structure_masks.shape[-1]:
            raise IndexError(
                f"ROI index {roi_index} is out of bounds for structure_masks with shape {structure_masks.shape}"
            )

        roi_mask = structure_masks[:, :, :, roi_index].to(torch.bool)
        print(
            f"Structure mask shape: {structure_masks.shape}, ROI mask shape: {roi_mask.shape}"
        )

        try:
            roi_mask_indices = structure_masks[:, :, :, roi_index].nonzero()
            print(f"Indices of ROI mask: {roi_mask_indices}")
        except Exception as e:
            print(f"Error accessing ROI mask indices: {e}")
            continue

        if not roi_mask.any():
            continue
        roi_dose = prediction.squeeze()[roi_mask]
        roi_size = roi_dose.size(0)
        print(f"ROI dose size: {roi_size}")

        metrics[roi] = {}

        for metric in ALL_DVH_METRICS[roi]:
            print(f"Calculating metric: {metric}")
            if metric == "D_0.1_cc":
                fractional_volume_to_evaluate = voxels_within_tenths_cc / roi_size
                print(f"Fractional volume to evaluate: {fractional_volume_to_evaluate}")
                metric_value = torch.quantile(
                    roi_dose, fractional_volume_to_evaluate.clip(0.0, 1.0)
                )
            elif metric == "mean":
                metric_value = roi_dose.mean()
            elif metric == "D_99":
                metric_value = torch.quantile(roi_dose, 0.01)
            elif metric == "D_95":
                metric_value = torch.quantile(roi_dose, 0.05)
            elif metric == "D_1":
                metric_value = torch.quantile(roi_dose, 0.99)
            else:
                raise ValueError(f"Metric {metric} is not supported.")
            metrics[roi][metric] = metric_value
    return metrics
