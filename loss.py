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
    def __init__(self, use_mae=True, use_dvh=True, use_lambert=True, alpha=0.5, beta=0.1, gamma=0.05, tau=0.1):
        super(RadiotherapyLoss, self).__init__()
        self.use_mae = use_mae
        self.use_dvh = use_dvh
        self.use_lambert = use_lambert
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.mae_loss = MAELoss()
        self.dvh_loss = DVHLoss()
        self.lambert_loss = LambertLoss(beta=beta, tau=tau)
        
    def forward(self, output, target, dvh_pred, dvh_true, distances=None):
        loss = 0
        
        if self.use_mae:
            loss += self.alpha * self.mae_loss(output, target)
        
        if self.use_dvh:
            loss += self.gamma * self.dvh_loss(dvh_pred, dvh_true)
        
        if self.use_lambert:
            loss += self.beta * self.lambert_loss(output, distances)
            
        return loss
    
##############################################
# Dummy toy model
##############################################


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc = nn.Linear(1, 1)
        
    def forward(self, x):
        return self.fc(x)
    
    
if __name__ == "__main__":
    # Test the loss functions
    
    
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
            dvh_pred = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
            dvh_true = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6])

            # example distances for Lambert Loss
            distances = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
                    
            # perform backward pass
            optimizer.zero_grad()
            
            
            # if lambert give distances
            if use_lambert:
                loss_fn = RadiotherapyLoss(use_mae=True, use_dvh=use_dvh, use_lambert=use_lambert)
                loss = loss_fn(output, target, dvh_pred, dvh_true, distances)
                
            else:
                loss_fn = RadiotherapyLoss(use_mae=True, use_dvh=use_dvh, use_lambert=use_lambert)
                loss = loss_fn(output, target, dvh_pred, dvh_true)
                
            print(loss)
            loss.backward()
            optimizer.step()
                
                
                
