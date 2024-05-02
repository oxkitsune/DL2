from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class DataBatch:
    def __init__(
        self,
        dose: Optional[NDArray] = None,
        predicted_dose: Optional[NDArray] = None,
        ct: Optional[NDArray] = None,
        structure_masks: Optional[NDArray] = None,
        structure_mask_names: Optional[list[str]] = None,
        possible_dose_mask: Optional[NDArray] = None,
        voxel_dimensions: Optional[NDArray] = None,
        patient_list: Optional[list[str]] = None,
        patient_path_list: Optional[list[Path]] = None,
    ):
        self.dose = dose
        self.predicted_dose = predicted_dose
        self.ct = ct
        self.structure_masks = structure_masks
        self.structure_mask_names = structure_mask_names
        self.possible_dose_mask = possible_dose_mask
        self.voxel_dimensions = voxel_dimensions
        self.patient_list = patient_list
        self.patient_path_list = patient_path_list

    @classmethod
    def initialize_from_required_data(
        cls, data_dimensions: dict[str, NDArray], batch_size: int
    ) -> DataBatch:
        attribute_values = {}
        for data, dimensions in data_dimensions.items():
            batch_data_dimensions = (batch_size, *dimensions)
            attribute_values[data] = np.zeros(batch_data_dimensions)
        return cls(**attribute_values)

    def set_values(self, data_name: str, batch_index: int, values: NDArray):
        getattr(self, data_name)[batch_index] = values

    def get_index_structure_from_structure(self, structure_name: str):
        if self.structure_mask_names is None:
            raise Exception("Structure_mask_names has not been initialized")

        return self.structure_mask_names.index(structure_name)

    def get_features(self, ptv_index: int) -> NDArray:
        """
        Returns a CT channel containing the CT image, an OAR channel
        containing the labeled OARs and a PTV channel containing the
        selected PTV.
        """
        if self.structure_masks is None:
            raise Exception("Structured masks data not in DataBatch")

        if self.ct is None:
            raise Exception("CT data not in DataBatch")

        ct_channel = self.ct

        # These are all encoded as ones and zeros originally
        oar_channels = self.structure_masks[:, :, :, :, :7]
        labels = np.arange(1, 8)
        labeled_oar_channels = oar_channels * labels
        oar_channel = np.sum(labeled_oar_channels, axis=-1, keepdims=True)

        ptv_channel = self.structure_masks[:, :, :, :, ptv_index, np.newaxis]

        return np.concatenate((ct_channel, oar_channel, ptv_channel), axis=-1)
