# Training settings

import os
from  src.models.vanilla_vae import VAE
from  src.models.gmm_vae import VAE_GMM
from dataloader import loader
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import yaml
from sklearn.manifold import TSNE


model_classes = {
    'VAE': VAE,  # Add other model classes if needed
    'VAE_GMM': VAE_GMM
}

class Train:
    def __init__(self, config):
        self.lr = config['model params']['lr']
        self.model_name = config['model params']['name']
        self.epochs = config['model params']['epochs']
        self.latent_dim = config['model params']['latent_dim']
        self.batch_size = config['loader_params']['batch_size']
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_loader, self.val_loader = loader(config['loader_params']['input_directory'], self.batch_size)

    def train(self):
        print("started training")
        model = model_classes[self.model_name]().to(self.device)
       # skip_and_latent_params = [model.skip_weight1, model.latent_weight]
       # skip_and_latent_ids = list(map(id, skip_and_latent_params))
        optimizer = optim.Adam(model.parameters(), lr=float(self.lr))
        #optimizer = torch.optim.Adam([
            # Filter out the parameters based on their id
           # {'params': filter(lambda p: id(p) not in skip_and_latent_ids, model.parameters())}
           # {'params': [model.skip_weight1], 'lr': 0.1},  # Smaller learning rate for skip weights
           # {'params': [model.latent_weight], 'lr': 0.0001}  # Larger learning rate for latent weight
       # ])
        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            model.train()
            train_loss = 0
            for batch_idx, data in enumerate(self.train_loader):
                data = data.to(self.device)
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(data)
                loss = model.loss_function( epoch, recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
    
            print(f'Epoch {epoch + 1}, Loss: {train_loss / len(self.train_loader.dataset)}')
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in self.val_loader:
                    data = data.to(self.device)
                    recon_batch, mu, logvar = model(data)
                    loss = model.loss_function(epoch, recon_batch, data, mu, logvar)
                    val_loss += loss.item()

            train_loss /= len(self.train_loader)
            val_loss /= len(self.val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(config['log_params']['assets'], self.model_name +  '_' + f'best_vae_epoch_f.pth'))
                print(f'Best model saved for epoch {epoch} with validation loss {best_val_loss:.4f}')
                

# Load the YAML file
with open('/home/drovco/Bhumika/VAEs/configs/vae.yaml', 'r') as file:
    config = yaml.safe_load(file)
obj = Train(config)
obj.train()                                             