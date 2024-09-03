import torch
import os
import yaml
import torch.nn as nn
import matplotlib.pyplot as plt
from vae import VAE
from dataloader import loader

class Test:
    def __init__(self, config):
        self.load_weights = config['log_params']['assets']
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.latent_dim = config['model params']['latent_dim']
        self.input_path = config['loader_params']['input_directory']
        self.batch_size = config['loader_params']['batch_size']


    def test(self):
        model = VAE().to(self.device)
        model.load_state_dict(torch.load(os.path.join(self.load_weights,f'best_vae_epoch.pth'), weights_only=True))  # Ensure correct epoch or best model is loaded
        model.eval()
        train_loader, val_loader = loader(self.input_path, self.batch_size)
        with torch.no_grad():
            data = next(iter(val_loader)).to(self.device)
            recon, _, _ = model(data)
            recon = recon.cpu()
            # Plot original and reconstructed images
            fig, axs = plt.subplots(2, 8, figsize=(15, 5))
            for i in range(8):
                axs[0, i].imshow(data[i].cpu().permute(1, 2, 0))
                axs[0, i].axis('off')
                axs[1, i].imshow(recon[i].permute(1, 2, 0))
                axs[1, i].axis('off')
            plt.show()
            plt.savefig(os.path.join(config['log_params']['assets'], 'test.png'))

    # Load the YAML file
with open('/home/drovco/Bhumika/VAEs/configs/vae.yaml', 'r') as file:
    config = yaml.safe_load(file)
obj = Test(config)
obj.test()   