
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_GMM(nn.Module):
    def __init__(self, image_channels=3, h_dim=256*8*8, z_dim=128, n_components=10):
        super(VAE_GMM, self).__init__()
        self.n_components = n_components
        self.z_dim = z_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),   
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
        )
        
        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(h_dim, z_dim * n_components)    # Mean vectors for all components
        self.fc_logvar = nn.Linear(h_dim, z_dim * n_components)  # Log variance vectors for all components
        self.fc_pi = nn.Linear(h_dim, n_components)  # Mixing coefficients
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

    def reparameterize_gmm(self, mu, logvar, pi):
        # Choose a Gaussian component based on the mixing coefficients
        categorical = torch.distributions.Categorical(pi)
        component = categorical.sample().unsqueeze(-1)  # Shape: (batch_size, 1)

        # Expand component to match the shape of mu and logvar
        component = component.expand(-1, self.z_dim).unsqueeze(1)  # Shape: (batch_size, 1, z_dim)

        # Select the mean and variance of the chosen component
        selected_mu = torch.gather(mu, 1, component).squeeze(1)  # Shape: (batch_size, z_dim)
        selected_logvar = torch.gather(logvar, 1, component).squeeze(1)  # Shape: (batch_size, z_dim)
        
        # Reparameterize
        std = torch.exp(0.5 * selected_logvar)
        eps = torch.randn_like(std)
        return selected_mu + eps * std

        # Select the mean and variance of the chosen component
        selected_mu = torch.gather(mu, 1, component.repeat(1, self.z_dim))
        selected_logvar = torch.gather(logvar, 1, component.repeat(1, self.z_dim))
        
        # Reparameterize
        std = torch.exp(0.5 * selected_logvar)
        eps = torch.randn_like(std)
        return selected_mu + eps * std

    def forward(self, x):
        h = self.encoder(x).view(x.size(0), -1)  # Flatten the output to (batch_size, h_dim)
        mu = self.fc_mu(h).view(-1, self.n_components, self.z_dim)  # Mean vectors
        logvar = self.fc_logvar(h).view(-1, self.n_components, self.z_dim)  # Log variance vectors
        self.pi = F.softmax(self.fc_pi(h), dim=-1)  # Mixing coefficients
        
        z = self.reparameterize_gmm(mu, logvar, self.pi)  # Sample from the GMM latent space
        z = self.fc3(z).view(-1, 256, 8, 8)  # Reshape for decoding
        return self.decoder(z), mu, logvar  # Decode to image

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.mse_loss(recon_x, x, reduction='sum')  # Reconstruction loss
        
        # KL Divergence for Gaussian Mixture Model
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        
        # Mix the KL divergence by the mixing coefficients and sum
        KLD = torch.sum(self.pi * KLD, dim=-1).mean()
        
        return BCE + KLD
