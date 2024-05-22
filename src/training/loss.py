import torch
import torch.nn as nn
   

class MomentDVHLoss(nn.Module):
    def __init__(self, moments=[1, 2, 10]):
        super(MomentDVHLoss, self).__init__()
        self.moments = moments

    def forward(self, predicted_dose, target_dose, structure_masks):
        """
        Compute the moment loss for multiple structures in a batch.
        
        :param predicted_dose: Batch of predicted dose tensors. Shape: (batch_size, D, H, W)
        :param target_dose: Batch of target dose tensors. Shape: (batch_size, D, H, W)
        :param structure_masks: List of structure masks for each batch. 
                                Each element is a tensor of shape (batch_size, D, H, W)
        :return: Moment loss value.
        """
        batch_size = predicted_dose.size(0)
        total_moment_loss = 0.0
        
        for structure_mask in structure_masks:
            # Compute Moment Loss for the current structure across the batch
            moment_loss_value = 0.0
            for p in self.moments:
                predicted_moment = self.compute_moment(predicted_dose, structure_mask, p)
                target_moment = self.compute_moment(target_dose, structure_mask, p)
                moment_loss_value += torch.norm(predicted_moment - target_moment, p=2) ** 2
            total_moment_loss += moment_loss_value

        # Average over the batch
        total_moment_loss /= batch_size

        return total_moment_loss

    def compute_moment(self, dose, mask, p):
        """
        Compute the p-th moment of the dose for the given structure mask.
        
        :param dose: Batch of dose tensors. Shape: (batch_size, D, H, W)
        :param mask: Batch of structure mask tensors. Shape: (batch_size, D, H, W)
        :param p: Moment order.
        :return: p-th moment value for each element in the batch.
        """
        # Apply mask to the dose to get dose values within the structure
        structure_dose = dose * mask

        # Compute the sum of the masked doses and the number of voxels in the structure
        structure_dose_sum = torch.sum(structure_dose ** p, dim=[1, 2, 3])
        structure_voxel_count = torch.sum(mask, dim=[1, 2, 3])

        # Compute the p-th moment
        moment = (structure_dose_sum / structure_voxel_count) ** (1 / p)
        
        return moment

