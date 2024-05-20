import torch
from torchvision.transforms.functional import rotate

class Augment(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)

    def augment_features(self, feat):
        # Rotations
        rotations = [0, 40, 80, 120, 160, 200, 240, 280, 320]
        rot = rotations[torch.randint(0, len(rotations), (1,))]
        self.rot = rot

        feat = torch.transpose(feat, 0, 3)
        (_, _, _, batch_size, feature_size) = feat.shape

        for feature in range(feature_size):
            for sample in range(batch_size):
                feat[:, :, :, sample, feature] = rotate(feat[:, :, :, sample, feature], rot)

        feat = torch.transpose(feat, 0, 3)

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
        dose = torch.transpose(dose, 0, 3)
        (_, _, _, batch_size) = dose.shape

        for sample in range(batch_size):
            dose[:, :, :, sample] = rotate(dose[:, :, :, sample], self.rot)

        dose = torch.transpose(dose, 0, 3)

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
