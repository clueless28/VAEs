import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample to 64x64
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample to 32x32
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample to 16x16
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        #x -> (32, 3, 128, 128)
        enc1 = self.encoder1(x)   #enc1 -> (32, 64, 128, 128)
        enc2 = self.encoder2(enc1)  #enc2 -> (32, 128, 64, 64)
        enc3 = self.encoder3(enc2) #enc3 -> (32, 256, 32, 32)
        enc4 = self.encoder4(enc3)#enc4 -> (32, 512, 16, 16)
        return enc1, enc2, enc3, enc4 

class Decoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Decoder, self).__init__()
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, enc1, enc2, enc3, enc4):
        ##################
        dec4 = self.upconv4(enc4)  # (32, 256, 32, 32)

        # Conditional concatenation
        if enc3 is not None:
            dec4 = torch.cat((dec4, enc3), dim=1)  # (32, 512, 32, 32)
        else:
            dec4 = torch.cat((dec4, dec4), dim=1)
        dec4 = self.decoder4(dec4)  # If no concatenation, just go to the decoder (32, 256, 32, 32)
        
        dec3 = self.upconv3(dec4)  # (32, 128, 64, 64)
        if enc2 is not None:
            dec3 = torch.cat((dec3, enc2), dim=1)  # (32, 256, 64, 64)
        else:
            dec3 = torch.cat((dec3, dec3), dim=1)
        dec3 = self.decoder3(dec3)  # (32, 128, 64, 64)
        
        dec2 = self.upconv2(dec3)  # (32, 64, 128, 128)
        if enc1 is not None:
            dec2 = torch.cat((dec2, enc1), dim=1)  # (32, 128, 128, 128)
        else:
            dec2 = torch.cat((dec2, dec2), dim=1)
        dec2 = self.decoder2(dec2)  # (32, 64, 128, 128)

        dec1 = self.decoder1(dec2)  # (32, 3, 128, 128)
        return dec1
    
class VAutoEncoder(nn.Module):
    def __init__(self, image_channels=3, z_dim=256):
        super(VAE, self).__init__()
        
        # Encoder-Decoder (Autoencoder Style)
        self.encoder = Encoder(image_channels)
        self.fc_mu = nn.Linear(512*16*16, z_dim)  # Fully connected layer for mean
        self.fc_logvar = nn.Linear(512*16*16, z_dim)  # Fully connected layer for log variance
        self.fc_dec = nn.Linear(z_dim, 512*16*16)  # Map latent space back to high-dimensional space
        self.decoder = Decoder(image_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        enc1, enc2, enc3, enc4 = self.encoder(x)  #x -> (batch_sz(32), channels(3), 128, 128)
        h = enc4.view(x.size(0), -1)  # Flatten for fully connected layers
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)  # Mean and log variance
        z = self.reparameterize(mu, logvar)  # Latent sampling
        z = self.fc_dec(z).view(-1, 512, 16, 16)  # Reshape for decoding

        # Decode
        recon_x = self.decoder(enc1, enc2, enc3, z)
        return recon_x, mu, logvar

    def loss_function(self, epoch, recon_x, x, mu, logvar):
        BCE = F.mse_loss(recon_x, x, reduction='sum')  # Reconstruction loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL Divergence
        m = 4
        r = 0.7
        beta = 0.5
        tau = ((epoch)) 
        factor = 0
        if tau > r:
            factor = 1.0
        else:
            factor =  beta * tau
        return BCE + KLD * factor
