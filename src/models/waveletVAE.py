import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import matplotlib.pyplot as plt

class WavletVAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=256*8*8, z_dim=256):
        super(WavletVAE, self).__init__()
        # Encoder
        # Encoder
        self.encoder_conv1 = nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1)
        self.encoder_relu1 = nn.ReLU()
        self.encoder_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.encoder_relu2 = nn.ReLU()
        self.encoder_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.encoder_relu3 = nn.ReLU()
        self.encoder_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.encoder_relu4 = nn.ReLU()
        # Fully connected layers for mean and log variance
        self.fc1 = nn.Linear(h_dim, z_dim)  # Mean vector
        self.fc2 = nn.Linear(h_dim, z_dim)  # Log variance vector
        self.fc3 = nn.Linear(z_dim, h_dim)  # Map latent vector back to high-dimensional space
        
        # Decoder
        self.decoder_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.decoder_relu1 = nn.ReLU()
        self.decoder_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.decoder_relu2 = nn.ReLU()
        self.decoder_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.decoder_relu3 = nn.ReLU()
        self.decoder_conv4 = nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1)
        self.decoder_sigmoid = nn.Sigmoid()
        self.attention_layer = self.attention_layer = nn.Sequential(
                            nn.Conv2d(128, 64, kernel_size=1),
                            nn.ReLU(),
                            nn.Conv2d(64, 32, kernel_size=1),
                            nn.ReLU(),
                            nn.Conv2d(32, 1, kernel_size=1),
                            nn.Sigmoid()
                        )      
        self.skip_weight1 = nn.Parameter(torch.tensor(0.5))
        self.latent_weight = nn.Parameter(torch.tensor(2.0))
        self.attention_weight = nn.Parameter(torch.tensor(1.0))
        self.skip_dropout = nn.Dropout(p=0.1)

    def wavelet_transform(self, x):
        # Perform 2D wavelet decomposition on input x
        coeffs = pywt.dwt2(x.cpu().detach().numpy(), 'haar')  # Decompose with Haar wavelets
        cA, (cH, cV, cD) = coeffs  # cA: low-frequency, cH/cV/cD: high-frequency
        return torch.tensor(cA, dtype=torch.float32).to(x.device), torch.tensor(cH + cV + cD, dtype=torch.float32).to(x.device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Apply wavelet transform to separate high and low frequency components
        
        low_freq, high_freq = self.wavelet_transform(x)
       # print(x.shape)
        # Encoder processes the low-frequency component
        h1 = self.encoder_relu1(self.encoder_conv1(x))  # (batch_size, 32, 32, 32)
        h2 = self.encoder_relu2(self.encoder_conv2(h1))  # (batch_size, 64, 16, 16)
        h3 = self.encoder_relu3(self.encoder_conv3(h2))  # (batch_size, 128, 8, 8)
        h4 = self.encoder_relu4(self.encoder_conv4(h3))  # (batch_size, 256, 4, 4)
        h = h4.view(x.size(0), -1)  # Flatten for fully connected layers
        mu, logvar = self.fc1(h), self.fc2(h)  # Mean and log variance
        z = self.reparameterize(mu, logvar)  # Sample from latent space
        z = self.fc3(z).view(-1, 256, 8, 8)  # Reshape for decoding

        # Decoder processes the latent representation
        d1 = self.decoder_relu1(self.decoder_conv1(z))
       # d1 = d1 + high_freq 
        #attention_weight = self.attention_layer(d1)
        #d1 = attention_weight * d1  # Apply attention to decoder layer
        
        # Skip connection passes high-frequency component directly
        d2 = self.decoder_relu2(self.decoder_conv2(d1))
        d3 = self.decoder_relu3(self.decoder_conv3(d2))
        recon_x = self.decoder_sigmoid(self.decoder_conv4(d3))
        recon_x = recon_x +  F.interpolate(high_freq, size=recon_x.shape[2:])#self.skip_dropout(F.interpolate(high_freq, size=recon_x.shape[2:]))
        return recon_x, mu, logvar

    def loss_function(self, epoch, recon_x, x, mu, logvar):
        beta_start = 0.0
        beta_end = 1.0
        anneal_steps = 100.0
        BCE = F.mse_loss(recon_x, x, reduction='sum')  # Reconstruction loss
       # BCE = F.binary_cross_entropy(recon_x, x, reduction='mean')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL Divergence
       # beta = min(beta_start + (beta_end - beta_start) * (epoch / anneal_steps), beta_end)
        #print('BCE, KLD, latent', BCE ,  KLD , latent_loss)
        return BCE + KLD #* beta+ self.info_nce_loss(x, recon_x, 0.1, 1e-9)  #* 1 + latent_loss * 10000
    