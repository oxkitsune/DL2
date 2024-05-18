import torch
import torch.nn as nn


##############################################
# Standard Losses
##############################################

class MAELoss(nn.Module):
    """ Mean Absolute Error Loss """
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, output, target):
        return torch.mean(torch.abs(output - target))
    
class MSELoss(nn.Module):
    """ Mean Squared Error Loss """
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, target):
        return torch.mean((output - target) ** 2)
    
class RMSELoss(nn.Module):
    """ Root Mean Squared Error Loss """
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, output, target):
        return torch.sqrt(torch.mean((output - target) ** 2))
    
##############################################
# Regularization Losses
##############################################

class SigmoidDVHLoss(nn.Module):
    """ DVH Loss """
    def __init__(self, beta=0.1):
        super(SigmoidDVHLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.beta = beta

    def structure_dvh(self, dose_batch, PTV_mask_batch, ptv_values):
        batch_size = dose_batch.size(0)
        unique_ptvs = torch.unique(PTV_mask_batch[PTV_mask_batch != 0])
        dvh_list = []

        for batch_index in range(batch_size):
            dose = dose_batch[batch_index]
            PTV_mask = PTV_mask_batch[batch_index]
            dvh = []

            for PTV in unique_ptvs:
                threshold = ptv_values[int(PTV.item()) - 1]
                structure_mask = (PTV_mask == PTV).float()
                structure_mask_sum = structure_mask.sum()
                if structure_mask_sum > 0:
                    dvh_pred_thresholded = dose - threshold
                    structure_loss = (self.sigmoid(dvh_pred_thresholded / self.beta) * structure_mask).sum() / structure_mask_sum
                    dvh.append(structure_loss)

            dvh = torch.stack(dvh) if dvh else torch.tensor([], device=dose.device)
            dvh_list.append(dvh)

        return torch.stack(dvh_list)

    def forward(self, predicted_dose, target_dose, PTV_mask_batch, ptv_values):
        predicted_dvh = self.structure_dvh(predicted_dose, PTV_mask_batch, ptv_values)
        target_dvh = self.structure_dvh(target_dose, PTV_mask_batch, ptv_values)
        loss = nn.MSELoss()(predicted_dvh, target_dvh)
        return loss
    

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
        
    
class LambertLoss(nn.Module):
    """ Lambert Loss
    
    We constrain the model to follow Lambert's law by penalizing the deviation from the desired exponential pattern.
    
    """
    def __init__(self, beta=0.1, tau=0.1):
        super(LambertLoss, self).__init__()
        self.beta = beta  # regularization weight
        self.tau = tau  #  attenuation constant

    def forward(self, predicted_doses, distances):
        # expected transmission based on Lambert's law       
        expected_transmission = torch.exp(-self.tau * distances)

        # calculate the loss as the mean squared error between predicted and expected transmissions
        loss = torch.mean((predicted_doses - expected_transmission) ** 2)

        # scale the loss by the regularization weight
        return self.beta * loss
    
##############################################
# Combined Loss
##############################################

class RadiotherapyLoss(nn.Module):
    def __init__(self, thresholds, use_mae=True, use_dvh=True, use_moment=True, alpha=0.5, beta=0.1, gamma=0.05, tau=0.1):
        super(RadiotherapyLoss, self).__init__()
        self.use_mae = use_mae
        self.use_dvh = use_dvh
        self.use_moment = use_moment
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.mae_loss = MAELoss()
        self.dvh_loss = SigmoidDVHLoss(thresholds=thresholds, beta=beta)
        self.moment_loss = MomentDVHLoss()
        
    def forward(self, output, target, dvh_pred, dvh_true, OAR_mask, PTV_mask, ptv_values):
        loss = 0

        structure_masks = torch.cat([OAR_mask, PTV_mask], dim=0)
        
        if self.use_mae:
            loss += self.alpha * self.mae_loss(output, target)
        
        if self.use_dvh:
            loss += self.gamma * self.dvh_loss(dvh_pred, dvh_true, PTV_mask, ptv_values)
        
        if self.use_moment:
            loss += self.beta * self.moment_loss(dvh_pred, dvh_true, structure_masks)
            
        return loss
    

    
    
if __name__ == "__main__":
    
    # Test the loss functions
    
    ##############################################
    # Toy model
    ##############################################


    class ToyModel(nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.fc = nn.Linear(1, 1)
            
        def forward(self, x):
            return self.fc(x)
    
    
    # setups with true/false and try to do backward
    for use_dvh in [True, False]:
        for use_lambert in [True, False]:
            print(f"\nUse MAE: {True}, Use DVH: {use_dvh}, Use Lambert: {use_lambert}")
            
            
            model = ToyModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


            # example data with requires_grad=True
            input_data = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])

            # forward pass through the model
            output = model(input_data)

            # test target data
            target = torch.tensor([[1.1], [2.1], [3.1], [4.1], [5.1]])

            # example DVH predictions and truths
            dvh_pred = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])
            dvh_true = torch.tensor([[0.2, 0.3, 0.4, 0.5, 0.6], [0.2, 0.3, 0.4, 0.5, 0.6]])

            # example distances for Lambert Loss
            distances = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

            # example OAR mask
            OAR_mask = torch.tensor([[1, 1, 2, 2, 3], [1, 1, 2, 2, 3]])
            PTV_mask = torch.tensor([[1, 1, 2, 2, 3], [1, 1, 2, 2, 3]])

            # example thresholds
            thresholds = torch.tensor([0.1, 0.2, 0.3])
            ptv_thresholds = torch.tensor([0.4, 0.5, 0.6])
                    
            # perform backward pass
            optimizer.zero_grad()
            
            # if lambert give distances
            if use_lambert:
                loss_fn = RadiotherapyLoss(thresholds, ptv_thresholds, use_mae=True, use_dvh=use_dvh, use_lambert=use_lambert)
                loss = loss_fn(output, target, dvh_pred, dvh_true, OAR_mask, PTV_mask, distances)
                
            else:
                loss_fn = RadiotherapyLoss(thresholds, ptv_thresholds, use_mae=True, use_dvh=use_dvh, use_lambert=use_lambert)
                loss = loss_fn(output, target, dvh_pred, dvh_true, OAR_mask, PTV_mask)
                
            print(loss)
            loss.backward()
            optimizer.step()
                
                
                
