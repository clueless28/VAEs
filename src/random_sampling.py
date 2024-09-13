import torch
import os
import yaml
import torchvision.transforms as transforms
import torch.nn as nn
from  src.models.vq_vae import VQVAE
from  src.models.gmm_vae import VAE_GMM
import matplotlib.pyplot as plt
from src.models.vanilla_vae import VAE
from PIL import Image
from dataloader import loader
import numpy as np

model_classes = {
    'VAE': VAE,  # Add other model classes if needed
    'VAE_GMM': VAE_GMM,
    'VQVAE': VQVAE
}
with open('/home/drovco/Bhumika/VAEs/configs/vae.yaml', 'r') as file:
    config = yaml.safe_load(file)

load_weights = config['log_params']['assets']
model_name = config['model params']['name']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
latent_dim = config['model params']['latent_dim']
input_path = config['loader_params']['input_directory']
batch_size = config['loader_params']['batch_size']
model = model_classes[model_name]().to(device)
model.load_state_dict(torch.load(os.path.join(config['log_params']['assets'], model_name +  '_' + f'best_vae_epoch.pth'), weights_only=True))  # Ensure correct epoch or best model is loaded

# Function to generate random samples from the VAE
def generate_random_sample(model, z_dim=256, num_samples=1):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        z = torch.normal(0, 1, size=(num_samples, z_dim)).to(next(model.parameters()).device)
        z = model.fc3(z).view(-1, 256, 8, 8)  # Reshape to match the decoder input shape
        generated_samples = model.decoder(z)  # Pass through decoder to get images
    return generated_samples

# Assuming your VAE model is already trained and named 'vae'
num_samples = 5  # Number of random samples you want to generatedsd
random_samples = generate_random_sample(model, z_dim=256, num_samples=num_samples)
# Save the figure
plt.imshow(random_samples[0].cpu().permute(1, 2, 0).numpy() )
plt.show()
plt.savefig(os.path.join(config['log_params']['assets'], model_name  + '_sampled_output.png'))