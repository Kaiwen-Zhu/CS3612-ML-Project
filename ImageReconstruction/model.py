import torch
import torch.nn as nn
import torch.nn.functional as F


class MyVAE(nn.Module):
        
    def __init__(self):
        super().__init__()
        hidden_dims = [128, 64]
        encoded_dim = 32
        self.fc1 = nn.Linear(3*32*32, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc_mu = nn.Linear(hidden_dims[1], encoded_dim)
        self.fc_logvar = nn.Linear(hidden_dims[1], encoded_dim)
        
        self.fc_3 = nn.Linear(encoded_dim, hidden_dims[1])
        self.fc_4 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.fc_output = nn.Linear(hidden_dims[0], 3*32*32)
        
        self.dropout = 0.5 

    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.fc2(x))
        mean, log_var = self.fc_mu(x), self.fc_logvar(x)
        return mean, log_var

    def decode(self, z):
        x = F.relu(self.fc_3(z))
        x = F.relu(self.fc_4(x))
        output = self.fc_output(x)
        return torch.sigmoid(output).reshape(output.shape[0], 3, 32, 32)

    def _reparameterize(self, mean, log_var):
        """Samples from N(`mean`, exp(`log_var`))."""        
        eps = torch.randn_like(mean)
        return mean + eps * torch.exp(log_var/2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
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