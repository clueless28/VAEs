# Training settings
from vae import VAE
from dataloader import loader
import torch
import os
import torch.optim as optim
import torch.nn.functional as F
import yaml


class Train:
    def __init__(self, config):
        self.lr = config['model params']['lr']
        self.epochs = config['model params']['epochs']
        self.latent_dim = config['model params']['latent_dim']
        self.batch_size = config['loader_params']['batch_size']
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_loader, self.val_loader = loader(config['loader_params']['input_directory'], self.batch_size)

    def train(self):
        model = VAE().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=float(self.lr))
        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            model.train()
            train_loss = 0
            for batch_idx, data in enumerate(self.train_loader):
                data = data.to(self.device)
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(data)
                loss = model.loss_function(recon_batch, data, mu, logvar)
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
                    loss = model.loss_function(recon_batch, data, mu, logvar)
                    val_loss += loss.item()

            train_loss /= len(self.train_loader)
            val_loss /= len(self.val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(config['log_params']['assets'],f'best_vae_epoch.pth'))
                print(f'Best model saved for epoch {epoch} with validation loss {best_val_loss:.4f}')


# Load the YAML file
with open('/home/drovco/Bhumika/VAEs/configs/vae.yaml', 'r') as file:
    config = yaml.safe_load(file)
obj = Train(config)
obj.train()                                             
