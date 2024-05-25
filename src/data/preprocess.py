import numpy as np

NUM_RELEVANT_STRUCTURES = 7


def transform_data(ct, structure_masks):
    # Need to add a channel dimension to the CT data
    # in order to end up with a feature map of shape (128, 128, 128, 3)
    ct_channel = np.clip(ct, -1024, 1500) / 1000
    ct_channel = np.expand_dims(ct, -1)

    # These are all encoded as ones and zeros originally
    oar_channels = structure_masks[..., :NUM_RELEVANT_STRUCTURES]
    labels = np.arange(1, NUM_RELEVANT_STRUCTURES + 1)
    labeled_oar_channels = oar_channels * labels
    oar_channel = np.sum(labeled_oar_channels, axis=-1, keepdims=True)

    # According to the original paper, the PTV channel is normalized by 70
    ptv_channels = structure_masks[..., NUM_RELEVANT_STRUCTURES:]
    labels = np.array([56, 63, 70])
    ptv_channel = np.max(ptv_channels * labels, axis=-1, keepdims=True) / 70

    # Combine the channels into a single feature tensor
    # (depth, width, height, channel)
    flattened_features = np.concatenate((ct_channel, oar_channel, ptv_channel), axis=-1)
    return {
        "features": flattened_features,
    }
