import numpy as np
import scipy
import torch


class Transform3D(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def __call__(self, imgs):
        # These are either one or zero, flips occur along the inferior-superior
        # axis and right-left axis
        infsup_flip, rl_flip = self.rng.integers(0, 2, 2) * 2 - 1
        imgs = imgs[:, :, ::infsup_flip, ::rl_flip, :]

        # Max translation of 20 voxels according to the paper, these translations
        # occur along the inferior-superior axis and the right-left axis.
        infsup_translation, rl_translation = self.rng.integers(-20, 21, 2)
        imgs = np.roll(imgs, infsup_translation, axis=2)
        imgs = np.roll(imgs, rl_translation, axis=3)

        # Apply random rotation.
        rotations = [0, 40, 80, 120, 160, 200, 240, 280, 320]
        rot = rotations[self.rng.integers(0, len(rotations))]
        imgs = scipy.ndimage.rotate(imgs, rot, axes=(1, 3), mode="nearest", order=0)
        imgs = np.divide(imgs, np.max(imgs), out=np.zeros_like(imgs), where=imgs != 0)

        return imgs
