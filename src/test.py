import torch
import os
import yaml
import torchvision.transforms as transforms
import torch.nn as nn
from  src.models.vq_vae import VQVAE
import matplotlib.pyplot as plt
from src.models.vanilla_vae import VAE
from  src.models.gmm_vae import VAE_GMM
from PIL import Image
from dataloader import loader
import numpy as np
model_classes = {
    'VAE': VAE,  # Add other model classes if needed
    'VAE_GMM': VAE_GMM,
    'VQVAE': VQVAE
}

class Test:
    def __init__(self, config):
        self.load_weights = config['log_params']['assets']
        self.model_name = config['model params']['name']
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.latent_dim = config['model params']['latent_dim']
        self.input_path = config['loader_params']['input_directory']
        self.batch_size = config['loader_params']['batch_size']
    def test(self):
        model = model_classes[self.model_name]().to(self.device)
        model.load_state_dict(torch.load(os.path.join(config['log_params']['assets'], self.model_name +  '_' + f'best_vae_epoch_f.pth'), weights_only=True))  # Ensure correct epoch or best model is loaded
        model.eval()
        image_dir = ['/home/drovco/Bhumika/abstract_art_512/abstract_abdul-qader-al-raes_258.jpg', '/home/drovco/Bhumika/abstract_art_512/abstract_yves-klein_6956.jpg', '/home/drovco/Bhumika/abstract_art_512/abstract_yves-klein_6946.jpg', '/home/drovco/Bhumika/abstract_art_512/abstract_yves-gaucher_4910.jpg'  ]
        transform = transforms.Compose([
                        transforms.Resize((128,128)),
                        transforms.ToTensor()
                    ])
        test_data = []
        for path in image_dir:
            img = Image.open(path)
            img = transform(img)
            test_data.append(img)
            
        with torch.no_grad():
            fig, axs = plt.subplots(len(test_data), 2, figsize=(10, 5 * len(test_data)))  # Create subplots for all images
            for i, data in enumerate(test_data):
                # Add batch dimension and move to the appropriate device
                data = data.to(self.device).unsqueeze(0)
                
                # Get the reconstructed image from the model
                recon, _, _ = model(data)
                
                # Move the reconstructed image and original data back to the CPU
                recon = recon.cpu()
                data = data.cpu()

                # Plot original image
                original_img = data[0].permute(1, 2, 0).numpy()  # Change shape to (H, W, C) for plotting
                axs[i, 0].imshow(np.clip(original_img, 0, 1))  # Clip values to be within [0, 1]
                axs[i, 0].axis('off')
                axs[i, 0].set_title('Original Image')

                # Plot reconstructed image
                recon_img = recon[0].permute(1, 2, 0).numpy()  # Change shape to (H, W, C) for plotting
                axs[i, 1].imshow(np.clip(recon_img, 0, 1))  # Clip values to be within [0, 1]
                axs[i, 1].axis('off')
                axs[i, 1].set_title('Reconstructed Image')

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

        # Save the figure
        plt.savefig(os.path.join(config['log_params']['assets'], self.model_name + '_output.png'))


    # Load the YAML file
with open('/home/drovco/Bhumika/VAEs/configs/vae.yaml', 'r') as file:
    config = yaml.safe_load(file)
obj = Test(config)
obj.test()   