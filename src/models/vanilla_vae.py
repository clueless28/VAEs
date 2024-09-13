import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import models
from torchvision.models import vgg19, VGG19_Weights
import torch.nn as nn

class HighFrequencyExtractor(nn.Module):
    def __init__(self):
        super(HighFrequencyExtractor, self).__init__()
        # Use convolution to mimic high-pass filter (like an edge detector)
        self.conv_high_freq = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    
    def forward(self, x):
        high_freq = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return self.conv_high_freq(high_freq)

class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=256*8*8, z_dim=256):
        super(VAE, self).__init__()
        self.encoder_conv1 = nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1)
        self.encoder_relu1 = nn.ReLU()
        self.encoder_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.encoder_relu2 = nn.ReLU()
        self.encoder_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.encoder_relu3 = nn.ReLU()
        self.encoder_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.encoder_relu4 = nn.ReLU()   
        
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
        self.skip_weight1 = nn.Parameter(torch.tensor(0.0000001))
        self.latent_weight = nn.Parameter(torch.tensor(0.001))
        self.attention_weight = nn.Parameter(torch.tensor(1.0))
        self.skip_dropout = nn.Dropout(p=0.1)
        self.skip_dropout = nn.Dropout(p=0.1)
        

        # High-Frequency Extractor for skip connections
        self.high_freq_extractor = HighFrequencyExtractor()

        # Decoder (similar to previous)
        self.decoder_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.decoder_relu1 = nn.ReLU()
        # ... (other layers)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)   
        return mu + eps * std

    def forward(self, x):
        # Encoder steps (similar to previous)
        h1 = self.encoder_relu1(self.encoder_conv1(x))  # (batch_size, 32, 32, 32)
        h2 = self.encoder_relu2(self.encoder_conv2(h1))  # (batch_size, 64, 16, 16)
        h3 = self.encoder_relu3(self.encoder_conv3(h2))  # (batch_size, 128, 8, 8)
        h4 = self.encoder_relu4(self.encoder_conv4(h3))  # (batch_size, 256, 4, 4)
        h = h4.view(x.size(0), -1)  # Flatten for fully connected layers
        mu, logvar = self.fc1(h), self.fc2(h)  # Mean and log variance
        

        # Decoder step
        high_freq = self.high_freq_extractor(h3)
        z = self.reparameterize(mu, logvar)  # Sample from latent space
        z = self.fc3(z).view(-1, 256, 8, 8)  # Reshape for decoding
        
        d1 = self.decoder_relu1(self.decoder_conv1(z))  # (batch_size, 128, 8, 8)
        d1 = self.latent_weight * d1 + self.skip_weight1 *  F.interpolate(high_freq, size=d1.shape[2:]) #self.skip_dropout(F.interpolate(high_freq, size=d1.shape[2:]))
        d2 = self.decoder_relu2(self.decoder_conv2(d1))
        d3 = self.decoder_relu3(self.decoder_conv3(d2))
        recon_x = self.decoder_sigmoid(self.decoder_conv4(d3))
       # recon_x = recon_x +  F.interpolate(high_freq, size=recon_x.shape[2:])#self.skip_dropout(F.interpolate(high_freq, size=recon_x.shape[2:]))
        return recon_x, mu, logvar
    
    def loss_function(self, epoch, recon_x, x, mu, logvar):
        beta_start = 0.0
        beta_end = 1.0
        anneal_steps = 10000.0
        BCE = F.mse_loss(recon_x, x, reduction='sum')  # Reconstruction loss
       # BCE = F.binary_cross_entropy(recon_x, x, reduction='mean')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL Divergence
        beta = min(beta_start + (beta_end - beta_start) * (epoch / anneal_steps), beta_end)
        #print('BCE, KLD, latent', BCE ,  KLD , latent_loss)
        return BCE + KLD * beta #+ self.info_nce_loss(x, recon_x, 0.1, 1e-9)  #* 1 + latent_loss * 10000