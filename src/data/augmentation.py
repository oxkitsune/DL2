import torch
from torchvision.transforms.functional import rotate

class Augment(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)

    def __call__(self, feat):
        aug_features = feat

        # Rotations
        rotations = [0, 40, 80, 120, 160, 200, 240, 280, 320]
        rot = rotations[torch.randint(0, len(rotations), (1,))]

        aug_features = torch.transpose(aug_features, 0, 3)
        (_, _, _, batch_size, feature_size) = aug_features.shape
    
        for feat in range(feature_size):
            for sample in range(batch_size):
                aug_features[:, :, :, sample, feat] = rotate(aug_features[:, :, :, sample, feat], rot)
    
        aug_features = torch.transpose(aug_features, 0, 3)

        # Translations
        shifts = torch.randint(-20, 21, (2,)).int()
        aug_features = torch.roll(aug_features, shifts=shifts.tolist(), dims=(1, 2))
    
        # Flip
        flips = torch.arange(2, 4)[torch.rand((2,)) > 0.5]
        aug_features = torch.flip(aug_features, dims=flips.tolist())
        
        return feat
