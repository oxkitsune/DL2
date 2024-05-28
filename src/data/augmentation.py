import torch
from monai.transforms import Rotate
import numpy as np


class Augment(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)

    def fit(self, feat):
        # Flip
        flips = torch.arange(2, 4)[torch.rand((2,)) > 0.5]
        feat = torch.flip(feat, dims=flips.tolist())
        self.flips = flips

        # Translations
        shifts = torch.randint(-5, 5, (2,)).int()
        feat = torch.roll(feat, shifts=shifts.tolist(), dims=(1, 2))
        self.shifts = shifts

        # Rotations
        rotations = torch.arange(0, 360, 40) * np.pi / 180
        rot = rotations[torch.randint(0, len(rotations), (1,))]
        r = Rotate((0, 0, rot), keep_size=True)

        for feature in range(feat.shape[-1]):
            feat[:, :, :, :, feature] = r(feat[:, :, :, :, feature])

        self.r = r

        return feat

    def augment_structure_masks(self, structure_masks):
        structure_masks = torch.flip(structure_masks, dims=self.flips.tolist())

        # Translations
        structure_masks = torch.roll(
            structure_masks, shifts=self.shifts.tolist(), dims=(1, 2)
        )

        # Rotations
        for structure in range(structure_masks.shape[-1]):
            structure_masks[:, :, :, :, structure] = self.r(
                structure_masks[:, :, :, :, structure]
            )

        return structure_masks

    def augment_dose(self, dose):
        # Flip
        dose = torch.flip(dose, dims=self.flips.tolist())

        # # Translations
        dose = torch.roll(dose, shifts=self.shifts.tolist(), dims=(1, 2))

        # Rotations
        dose = self.r(dose)

        return dose
