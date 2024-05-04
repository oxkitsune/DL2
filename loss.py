import torch
import torch.nn as nn
from scipy.special import lambertw


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

class DVHLoss(nn.Module):
    """ DVH Loss """
    def __init__(self):
        super(DVHLoss, self).__init__()

    def forward(self, dvh_pred, dvh_true):
        return torch.mean(torch.abs(dvh_pred - dvh_true))
    
class LambertLoss(nn.Module):
    """ Lambert Loss
    
    We constrain the model to follow Lambert's law by penalizing the deviation from the desired exponential pattern.
    
    """
    def __init__(self, beta=0.1):
        super(LambertLoss, self).__init__()
        self.beta = beta  # regularization weight

    def forward(self, predicted_doses):
        # incremental doses derived from predicted doses
        increments = predicted_doses[1:] - predicted_doses[:-1]
        
        desired_pattern = torch.exp(increments)  # target exponential pattern
        lambert_term = torch.from_numpy(lambertw(increments.detach().numpy()).real).to(predicted_doses.device)
        regularization_loss = torch.mean((lambert_term - desired_pattern) ** 2)
        return self.beta * regularization_loss
    
##############################################
# Combined Loss
##############################################

class RadiotherapyLoss(nn.Module):
    def __init__(self, use_mae=True, use_dvh=True, use_lambert=True, alpha=0.5, beta=0.1, gamma=0.05):
        super(RadiotherapyLoss, self).__init__()
        self.use_mae = use_mae
        self.use_dvh = use_dvh
        self.use_lambert = use_lambert
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.mae_loss = MAELoss()
        self.dvh_loss = DVHLoss()
        self.lambert_loss = LambertLoss(beta=beta)
        
    def forward(self, output, target, dvh_pred, dvh_true):
        loss = 0
        
        if self.use_mae:
            loss += self.alpha * self.mae_loss(output, target)
        
        if self.use_dvh:
            loss += self.gamma * self.dvh_loss(dvh_pred, dvh_true)
        
        if self.use_lambert:
            loss += self.beta * self.lambert_loss(output)
            
        return loss