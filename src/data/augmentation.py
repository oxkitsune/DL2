import torch
from monai.transforms import Rotate
import numpy as np

class Augment(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)

    def augment_features(self, feat):
        # Rotations
        rotations = torch.arange(0, 360, 40) * np.pi / 180     
        rot = rotations[torch.randint(0, len(rotations), (1,))]
        r = Rotate((0, 0, rot), keep_size=True)
        
        for feature in range(feat.shape[-1]):
            feat[:, :, :, :, feature] = r(feat[:, :, :, :, feature])

        self.r = r

        # Translations
        shifts = torch.randint(-20, 21, (2,)).int()
        feat = torch.roll(feat, shifts=shifts.tolist(), dims=(1, 2))
        self.shifts = shifts

        # Flip
        flips = torch.arange(2, 4)[torch.rand((2,)) > 0.5]
        feat = torch.flip(feat, dims=flips.tolist())
        self.flips = flips
        
        return feat 

    def augment_dose(self, dose):
        # Rotations
        dose = self.r(dose)

        # Translations
        dose = torch.roll(dose, shifts=self.shifts.tolist(), dims=(1, 2))

        # Flip
        dose = torch.flip(dose, dims=self.flips.tolist())

        return dose

    def __call__(self, x):
        if x.dim() == 5:
            return self.augment_features(x)
        elif x.dim() == 4:
            return self.augment_dose(x)

        return None
