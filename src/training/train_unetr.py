import torch

def train_unetr(data_loader, model, epochs, ptv_index):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device {device}")
    criterion = torch.nn.L1Loss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001) # use lr 1e-4 and AdamW
    
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        data_loader.shuffle_data()
        
        for batch in data_loader.get_batches():
            features = batch.get_flattend_oar_features(ptv_index=ptv_index)
            input = torch.Tensor(features).transpose(1, 4).to(device)
            target = batch.dose
            target = torch.Tensor(target).transpose(1, 4).to(device)
            
            print(input.shape)
            print(target.shape)
            
            optimizer.zero_grad()
            outputs = model(input)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            print(f"Model loss at epoch {epoch} is {loss.item():.3f}")