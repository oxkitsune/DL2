import torch
from torchvision.transforms.functional import rotate

class Augment(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)

    def __call__(self, batch):
        features = batch["features"]

        print(features)
        print(type(features))

        # Rotations
        rotations = [0, 40, 80, 120, 160, 200, 240, 280, 320]
        rot = rotations[torch.randint(0, len(rotations), (1,))]

        features = torch.transpose(features, 0, 3)
        (_, _, _, batch_size, feature_size) = features.shape
    
        for feat in range(feature_size):
            for sample in range(batch_size):
                features[:, :, :, sample, feat] = rotate(features[:, :, :, sample, feat], rot)
    
        features = torch.transpose(features, 0, 3)

        # Translations
        shifts = torch.randint(-20, 21, (2,)).int()
        features = torch.roll(features, shifts=shifts.tolist(), dims=(1, 2))
    
        # Flip
        flips = torch.arange(2, 4)[torch.rand((2,)) > 0.5]
        features = torch.flip(features, dims=flips.tolist())
        
        return batch
