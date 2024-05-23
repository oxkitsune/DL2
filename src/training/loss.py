import torch
import torch.nn as nn


class DVHLoss(nn.Module):
    def __init__(self, n_bins: int):
        super(DVHLoss, self).__init__()
        self.n_bins = n_bins
        self.bin_width = 1 / n_bins
        self.loss = torch.nn.MSELoss()

    def comp_hist(self, predicted_dose, structure_masks):
        """
        Calculate DVH loss: averaged over all OARs. Target hist is already computed
            predicted dose (tensor) -- [N, C, D, H, W] C = 1
            oar (tensor)            -- [N, C, D, H, W] C == n_oars one hot encoded OAR including PTV
        """
        batch_size = predicted_dose.shape[0]
        structure_masks = torch.permute(structure_masks, (0, 4, 1, 2, 3))
        num_voxels = torch.sum(structure_masks, dim=(2, 3, 4)) + 1
        hist = torch.zeros(size=(batch_size, self.n_bins, structure_masks.shape[1]))

        for i in range(self.n_bins):
            diff = torch.sigmoid((predicted_dose - i * self.bin_width) / self.bin_width)
            diff = diff.repeat(1, structure_masks.shape[1], 1, 1, 1) * structure_masks
            num = torch.sum(diff, dim=(0, 2, 3, 4))
            hist[:,i] = (num / num_voxels)

        return hist

    def forward(self, predicted_dose, target_dose, structure_masks):
        # print min max and mean of predicted dose
        # print("Predicted Dose")
        # print(torch.min(predicted_dose))
        # print(torch.max(predicted_dose))
        # print(torch.mean(predicted_dose))
        # print("Target Dose")
        # print(torch.min(target_dose))
        # print(torch.max(target_dose))
        # print(torch.mean(target_dose))
        predicted_hist = self.comp_hist(predicted_dose, structure_masks)
        # print(predicted_hist)
        target_hist = self.comp_hist(target_dose, structure_masks)
        # print(target_hist)
        return self.loss(predicted_hist, target_hist)  / predicted_dose.shape[0]
   

class MomentDVHLoss(nn.Module):
    def __init__(self, moments=[1, 2, 10]):
        super(MomentDVHLoss, self).__init__()
        self.moments = moments
        self.mse = torch.nn.MSELoss()

    def forward(self, predicted_dose, target_dose, structure_masks):
        """
        Compute the moment loss for multiple structures in a batch.
        
        :param predicted_dose: Batch of predicted dose tensors. Shape: (batch_size, D, H, W)
        :param target_dose: Batch of target dose tensors. Shape: (batch_size, D, H, W)
        :param structure_masks: List of structure masks for each batch. 
                                Each element is a tensor of shape (batch_size, D, H, W)
        :return: Moment loss value.
        """
        predicted_dose = predicted_dose.squeeze(1)
        target_dose = target_dose.squeeze(1)

        batch_size = predicted_dose.size(0)
        total_moment_loss = 0.0
        
        for structure_mask_i in range(structure_masks.shape[-1]):
            structure_mask = structure_masks[:, :, :, :, structure_mask_i]

            # Compute Moment Loss for the current structure across the batch
            moment_loss_value = 0.0
            for p in self.moments:
                predicted_moment = self.compute_moment(predicted_dose, structure_mask, p)
                target_moment = self.compute_moment(target_dose, structure_mask, p)
                moment_loss_value += self.mse(predicted_moment, target_moment)

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
        structure_voxel_count = torch.sum(mask, dim=[1, 2, 3]) + 1

        # Compute the p-th moment
        moment = structure_dose_sum / structure_voxel_count
        moment += 1e-6

        moment = moment ** (1 / p)
        if moment[0] == 0:
            print("Moment is zero")
            print(structure_dose_sum)
            print(structure_voxel_count)

        return moment


class RadiotherapyLoss(nn.Module):
    def __init__(
        self, 
        use_mae=True, 
        use_dvh=True, 
        use_moment=True, 
        alpha=1, 
        beta=0.001, 
        gamma=0.0000005
    ):
        super(RadiotherapyLoss, self).__init__()
        self.use_mae = use_mae
        self.use_dvh = use_dvh
        self.use_moment = use_moment
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.mae_loss = torch.nn.L1Loss()
        self.dvh_loss = DVHLoss(n_bins=10)
        self.moment_loss = MomentDVHLoss()
        
    def forward(self, output, target, structure_masks):
        loss = 0
        print("Computing loss")

        if torch.isnan(structure_masks).any():
            print("STRUCTURE MASKS contain NaN values.")
        
        if self.use_mae:
            if torch.isnan(output).any():
                print("Output contain NaN values.")

            if torch.isnan(target).any():
                print("Targets contain NaN values.")

            l = self.alpha * self.mae_loss(output, target)
            print("MAE", l)
            loss += l
        
        if self.use_dvh:
            l = self.gamma * self.dvh_loss(output, target, structure_masks)
            print("DVH", l)
            loss += l
 
        if self.use_moment:
            l =  self.beta * self.moment_loss(output, target, structure_masks)
            print("Moment", l)
            loss += l

        print("final", loss)

        return loss
