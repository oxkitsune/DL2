"""
Strategy: compute lambert loss per axial slice (beam) and sum them up
for one beam, the equation is: T = e^{-\mu d}, where mu is the attenuation coefficient, d is the travelled distance, 
and T is the transmission factor, which can be computed by dividing I_d by I_0

So to compute the loss, we need:
- the attenuation coefficient mu
- the travelled distance d
- the initial intensity I_0
- the final intensity I_d

We want to compute the difference between the predicted I_d and the I_d that follows the equation

We know the following things:
- The initial intensity is the value at the exact border of the true dose map. I_0 can differ between beams
- The true final intensity is the value at the center. We could also do at the other border, makes less sense I think since beams can be symmetrical
- The travelled distance is the distance between the border and the center. We have the exact voxel dimensions per patient, need those to compute the travelled distance
- The attenuation coefficient depends on the ct scan/organs at risk. 
    - We could do an assumption based on the ct scan. 
    - Number has to be between 0 and 1, otherwise dose would increase
    - Not sure how to do this yet 
    - For now, assume constant.
    - Could we compute it based on the true data using the law?

Loss function:
- Input: True dose of beam, predicted dose of beam, voxel dimensions
- Step 1: Compute center of beam
- Step 2: Compute travelled distance
- Step 3: Get I_0 and I_d from true dose map
- Step 4: Determine attenuation coefficient from true dose map
- Step 5: Get I_0 and I_d from predicted dose map
- Step 6: Using equation, compute what I_d should be
- Step 7: Compute loss between predicted I_d and true I_d
- Output: Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LambertLoss(nn.Module):
    def __init__(self):
        super(LambertLoss, self).__init__()
        
    def forward_1d(self, true_dose, predicted_dose, voxel_dimensions):
        # For now do only 1d. #TODO: make it 3d

        # Step 1: Compute center of beam for every true dose map
        center = torch.tensor(true_dose[0].shape) / 2  #TODO now assumes same center for every elem in batch
        # convert center to int
        center = center.int()
        
        # Step 2: Compute travelled distance
        travelled_distance = voxel_dimensions[:, 0] * center # assume x and y voxel dimensions are the same, is fair I think #TODO distance should be from border
        # does the unit of the distance matter? Think not, since we're only interested in the ratio
        
        # Step 3: Get I_0 and I_d from true dose map
        I_0_true = true_dose[:, 0]
        I_d_true = true_dose[:, center]

        # Step 4: Determine attenuation coefficient from true dose map
        # - ln(T)/d = mu
        T_true = I_d_true / I_0_true
        mu = -torch.log(T_true) / travelled_distance

        # Step 5: Get I_0 and I_d from predicted dose map
        I_0_predicted = predicted_dose[:, 0] #TODO assumes border is at 0
        I_d_predicted = predicted_dose[:, center]
        
        # Step 6: Using equation, compute what I_d should be
        T_pred_law = torch.exp(-mu * travelled_distance)
        T_pred = I_d_predicted / I_0_predicted
        
        # Step 7: Compute loss between predicted I_d and true I_d
        loss = F.mse_loss(T_pred_law, T_pred)
        
        return loss
    
    def forward(self, true_dose, predicted_dose, voxel_dimensions):
        # For now do only 2d. #TODO: make it 3d

        # Step 1: Compute center of beam for every true dose map
        center = torch.tensor(true_dose[0].shape) / 2  #TODO now assumes same center for every elem in batch
        # convert center to int
        center = center.int()
        
        # Step 2: Compute travelled distance
        # first compute the distance between point 0, 0 and the center
        unit_distance = torch.sqrt(center[0] ** 2 + center[1] ** 2) #TODO distance should be from border
        travelled_distance = voxel_dimensions[:, 0] * unit_distance # assume x and y voxel dimensions are the same, is fair I think 
        # does the unit of the distance matter? Think not, since we're only interested in the ratio
        
        # Step 3: Get I_0 and I_d from true dose map
        I_0_true = true_dose[:, 0, 0]
        I_d_true = true_dose[:, center[0], center[1]]

        # Step 4: Determine attenuation coefficient from true dose map
        # - ln(T)/d = mu
        T_true = I_d_true / I_0_true
        mu = -torch.log(T_true) / travelled_distance

        # Step 5: Get I_0 and I_d from predicted dose map
        I_0_predicted = predicted_dose[:, 0, 0] #TODO assumes border is at 0, 0
        I_d_predicted = predicted_dose[:, center[0], center[1]]
        
        # Step 6: Using equation, compute what I_d should be
        T_pred_law = torch.exp(-mu * travelled_distance)
        T_pred = I_d_predicted / I_0_predicted
        
        # Step 7: Compute loss between predicted I_d and true I_d
        loss = F.mse_loss(T_pred_law, T_pred)
        
        return loss
    

if __name__ == "__main__":
    
    # Test the loss function
    
    ##############################################
    # Toy model
    ##############################################


    class ToyModel(nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.fc = nn.Linear(1, 1)
            
        def forward(self, x):
            return self.fc(x)
    
    
    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


    # # example data with requires_grad=True
    # input_data = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])

    # # forward pass through the model
    # output = model(input_data)

    # # test target data
    # target = torch.tensor([[1.1], [2.1], [3.1], [4.1], [5.1]])

    # example DVH predictions and truths
    # dvh_pred = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])
    # dvh_true = torch.tensor([[0.2, 0.3, 0.4, 0.5, 0.6], [0.2, 0.3, 0.4, 0.5, 0.6]])

    true_dose_1d = torch.tensor([[0.5, 0.4, 0.3, 0.2, 0.1], [0.5, 0.4, 0.3, 0.2, 0.1]])
    predicted_dose_1d = torch.tensor([[0.5, 0.4, 0.5, 0.2, 0.1], [0.5, 0.4, 0.5, 0.2, 0.1]])
    voxel_dim_1d = torch.tensor([[3.5, 3.5, 2], [3.5, 3.5, 2]])

    # TODO only takes 1 pixel into account, not like a bandwidth or something
    true_dose_2d = torch.tensor([[[0.5, 0.4, 0.5], [0.2, 0.1, 0.1], [0.5, 0.4, 0.5]], [[0.5, 0.4, 0.5], [0.2, 0.1, 0.1], [0.5, 0.4, 0.5]]])
    predicted_dose_2d = torch.tensor([[[0.5, 0.4, 0.5], [0.2, 0.1, 0.1], [0.5, 0.4, 0.5]], [[0.5, 0.4, 0.5], [0.2, 0.1, 0.1], [0.5, 0.4, 0.5]]])
    voxel_dim_2d = torch.tensor([[3.5, 3.5, 2], [3.5, 3.5, 2]])

    # perform backward pass
    optimizer.zero_grad()
    
    loss_fn = LambertLoss()
    loss = loss_fn(true_dose_2d, predicted_dose_2d, voxel_dim_2d)
        
    print(loss)
    loss.backward()
    optimizer.step()
                
                
                
