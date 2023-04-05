import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
##### VAE
class VAE(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        self.encoder_fc1 = nn.Linear(input_dim, 256)
        self.encoder_fc2 = nn.Linear(256, 128)
        self.encoder_fc3 = nn.Linear(128, 64)

        self.z_mean = nn.Linear(64, latent_dim)
        self.z_logvar = nn.Linear(64, latent_dim)

        self.decoder_fc1 = nn.Linear(latent_dim, 64)
        self.decoder_fc2 = nn.Linear(64, 128)
        self.decoder_fc3 = nn.Linear(128, 256)
        self.decoder_fc4 = nn.Linear(256, output_dim)

    def encode(self, x):
        h = F.relu(self.encoder_fc1(x))
        h = F.relu(self.encoder_fc2(h))
        h = F.relu(self.encoder_fc3(h))
        return self.z_mean(h), self.z_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.decoder_fc1(z))
        h = F.relu(self.decoder_fc2(h))
        h = F.relu(self.decoder_fc3(h))
        return self.decoder_fc4(h)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        return output

class GCNAutoencoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super(GCNAutoencoder, self).__init__()
        self.encoder1 = GraphConvolution(input_dim, hidden_dim)
        self.encoder2 = GraphConvolution(hidden_dim, output_dim)
        self.decoder1 = GraphConvolution(output_dim, hidden_dim)
        self.decoder2 = GraphConvolution(hidden_dim, input_dim)

    def forward(self, x, adj):
        encoded1 = self.encoder1(x, adj)
        encoded2 = self.encoder2(encoded1, adj)
        decoded1 = self.decoder1(encoded2, adj)
        decoded2 = self.decoder2(decoded1, adj)
        return decoded2, encoded2
