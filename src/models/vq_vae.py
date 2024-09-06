import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        z_flattened = z.view(-1, self.embedding_dim)
        distances = (torch.sum(z_flattened ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(z_flattened, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        z_q = self.embedding(encoding_indices).view(z.shape)
        loss = F.mse_loss(z_q.detach(), z) + self.commitment_cost * F.mse_loss(z_q, z.detach())
        z_q = z + (z_q - z).detach()

        return z_q, loss, encoding_indices


class VQVAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=256*8*8, z_dim=8192, num_embeddings=512, embedding_dim=128):
        super(VQVAE, self).__init__()
        self.z_dim = z_dim
        self.embedding_dim = embedding_dim

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
        
        # Fully connected layer to map encoder output to 8192 elements (for reshaping to (128, 8, 8))
        self.fc1 = nn.Linear(h_dim, z_dim)  # Map from 16384 to 8192
        self.fc2 = nn.Linear(z_dim, h_dim)  # Map back from 8192 to 16384 (for decoding)

        # Vector quantizer for latent space
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encoder to obtain latent representation
        h = self.encoder(x).view(x.size(0), -1)  # Flatten the output to (batch_size, 16384)
        z = self.fc1(h).view(-1, self.embedding_dim, 8, 8)  # Reshape to (batch_size, 128, 8, 8)

        # Apply vector quantization
        z_q, vq_loss, encoding_indices = self.vq_layer(z)

        # Decoder to reconstruct the input
        z_q = self.fc2(z_q.view(z_q.size(0), -1)).view(-1, 256, 8, 8)  # Reshape to (batch_size, 256, 8, 8)
        recon_x = self.decoder(z_q)
        tmp = 0

        return recon_x, vq_loss, tmp

    def loss_function(self, recon_x, x, vq_loss, tmp):
        # Reconstruction loss (MSE or BCE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # Total loss is the sum of reconstruction loss and VQ loss
        return recon_loss + vq_loss
