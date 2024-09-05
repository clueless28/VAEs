import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=256*8*8, z_dim=128):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),   #(32,3,64,64)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), #(32,64, 32,32)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), #(32,128, 16,16)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),   #(32,256,8,8)
            nn.ReLU(),   #Let's say the size after the encoder is [batch_size, 256, 4, 4]. In that case, h_dim should be 256*4*4
        )
        
        # Fully connected layers for mean and log variance
        self.fc1 = nn.Linear(h_dim, z_dim)  # Mean vector
        self.fc2 = nn.Linear(h_dim, z_dim)  # Log variance vector
        self.fc3 = nn.Linear(z_dim, h_dim)  # Map latent vector back to high-dimensional space
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Ensure output is in the range [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x).view(x.size(0), -1)  # Flatten the output
        mu, logvar = self.fc1(h), self.fc2(h)  # Mean and log variance
        z = self.reparameterize(mu, logvar)  # Sample from latent space
        z = self.fc3(z).view(-1, 256, 8, 8)  # Reshape for decoding
        return self.decoder(z), mu, logvar  # Decode to

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.mse_loss(recon_x, x, reduction='sum')  # Reconstruction loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL Divergence
        return BCE + KLD
        
