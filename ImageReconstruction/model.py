import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MyVAE(nn.Module):
    """
    Args:
        hidden_dims (List[int]): Hidden dimensions of the encoder and decoder.
        init_size (int): Size of feature maps of the input image. Must be divisible by 2**len(hidden_dims).
        code_dim (int): Dimension of the latent code.
    """     
        
    def __init__(self, init_size: int, code_dim: int, hidden_dims: List[int]):       
        super().__init__()

        self.hidden_dims = hidden_dims
        self.final_size = init_size >> len(hidden_dims)
        fc_dim = hidden_dims[-1]*(self.final_size**2)
        channels = [3] + hidden_dims

        # Encoder
        encode_layers = []
        for i in range(len(channels)-1):
            encode_layers.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i+1],
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU())
            )
        self.encode_block = nn.Sequential(*encode_layers)
        self.fc_mean = nn.Linear(fc_dim, code_dim)
        self.fc_logvar = nn.Linear(fc_dim, code_dim)

        # Decoder
        decode_layers = []
        self.fc_decode = nn.Linear(code_dim, fc_dim)
        channels.reverse()
        for i in range(len(channels)-2):
            decode_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(channels[i], channels[i+1],
                                       kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU())
            )
        decode_layers.append(nn.ConvTranspose2d(channels[-2], channels[-1],
                                kernel_size=4, stride=2, padding=1))
        self.decode_block = nn.Sequential(*decode_layers)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        out = self.encode_block(x)
        out = out.view(out.shape[0], -1)
        mu = self.fc_mean(out)
        log_var = self.fc_logvar(out)
        return mu, log_var

    def decode(self, z):
        out = self.fc_decode(z)
        out = out.view(-1, self.hidden_dims[-1], self.final_size, self.final_size)
        out = self.decode_block(out)
        out = self.sigmoid(out)
        return out

    def _reparameterize(self, mean, log_var):
        """Samples from N(`mean`, exp(`log_var`))."""        
        eps = torch.randn_like(mean)
        return mean + eps * torch.exp(log_var/2)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self._reparameterize(mean, log_var)
        out = self.decode(z)
        return out, mean, log_var
    
    def loss_fn(self, x, out, mean, log_var):
        # Reconstruction loss
        recons_loss = F.mse_loss(out, x, reduction='sum')
        # KL divergence loss
        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return (recons_loss + kl_div) / x.shape[0]
