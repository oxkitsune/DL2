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
    def __init__(self, thresholds, ptv_values, beta=0.1):
        super(SigmoidDVHLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.beta = beta # histogram bin width
        self.thresholds = thresholds
        self.ptv_values = ptv_values
    
    def structure_dvh(self, dose_batch, OAR_mask_batch, PTV_mask_batch):
        batch_size = dose_batch.size(0)
        dvh_list = []

        for batch_index in range(batch_size):
            dose = dose_batch[batch_index]
            OAR_mask = OAR_mask_batch[batch_index]
            PTV_mask = PTV_mask_batch[batch_index]
            dvh = torch.Tensor()

            for structure in OAR_mask[OAR_mask != 0].unique():
                threshold = self.thresholds[int(structure.item() - 1)]
                structure_mask = OAR_mask == structure
                dvh_pred_thresholded = dose - threshold
                structure_loss = (self.sigmoid(dvh_pred_thresholded / self.beta) * structure_mask) / structure_mask.sum()
                dvh = torch.cat((dvh, structure_loss), dim=0)

            for PTV in PTV_mask[PTV_mask != 0].unique():
                threshold = self.ptv_values[int(PTV.item() - 1)]
                structure_mask = PTV_mask == PTV
                dvh_pred_thresholded = dose - threshold
                structure_loss = (self.sigmoid(dvh_pred_thresholded / self.beta) * structure_mask) / structure_mask.sum()
                dvh = torch.cat((dvh, structure_loss), dim=0)

            dvh_list.append(dvh)

        return torch.stack(dvh_list)
    
    def forward(self, dvh_pred, dvh_true, OAR_mask, PTV_mask):
        dvh_pred = self.structure_dvh(dvh_pred, OAR_mask, PTV_mask)
        dvh_true = self.structure_dvh(dvh_true, OAR_mask, PTV_mask)
        
        loss = 0
        for pred, true in zip(dvh_pred, dvh_true):
            loss += torch.mean((pred - true) ** 2) / len(pred)

        # TODO divide by 1/n(t)?     
        return loss/len(dvh_pred)
    
class MomentDVHLoss(nn.Module):
    """ DVH Loss """
    def __init__(self, thresholds, beta=0.1):
        super(MomentDVHLoss, self).__init__()
        self.beta = beta # histogram bin width
        self.thresholds = thresholds
        
    
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
    def __init__(self, thresholds, ptv_values, use_mae=True, use_dvh=True, use_lambert=True, alpha=0.5, beta=0.1, gamma=0.05, tau=0.1):
        super(RadiotherapyLoss, self).__init__()
        self.use_mae = use_mae
        self.use_dvh = use_dvh
        self.use_lambert = use_lambert
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.mae_loss = MAELoss()
        self.dvh_loss = SigmoidDVHLoss(thresholds=thresholds, ptv_values=ptv_values, beta=beta)
        self.lambert_loss = LambertLoss(beta=beta, tau=tau)
        
    def forward(self, output, target, dvh_pred, dvh_true, OAR_mask, PTV_mask, distances=None):
        loss = 0
        
        if self.use_mae:
            loss += self.alpha * self.mae_loss(output, target)
        
        if self.use_dvh:
            loss += self.gamma * self.dvh_loss(dvh_pred, dvh_true, OAR_mask, PTV_mask)
        
        if self.use_lambert:
            loss += self.beta * self.lambert_loss(output, distances)
            
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
                
                
                
