import torch
import torch.nn as nn
import numpy as np


class DiagonalCovarianceStochasticMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, shared_hidden_layers, output_dim, device):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.shared_hidden_layers = shared_hidden_layers
        self.output_dim = output_dim
        self.device = device

        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Linear(hidden_dim, hidden_dim),nn.ReLU()]*shared_hidden_layers,
            nn.ReLU(),
        )
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.log_std_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.to(self.device)
    def forward(self, x):
        shared = self.shared_net(x)
        mean = self.mean_head(shared)
        log_standard_deviation = self.log_std_head(shared).clamp(-5, 2)
        standard_deviation = torch.exp(log_standard_deviation)
        return mean, standard_deviation

class Encoder(DiagonalCovarianceStochasticMLP):
    def __init__(self, input_dim, hidden_dim, shared_hidden_layers, latent_dim, device):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            shared_hidden_layers=shared_hidden_layers,
            output_dim=latent_dim,
            device=device
        )
    
class Dynamics(DiagonalCovarianceStochasticMLP):
    def __init__(self, latent_dim, action_dim, hidden_dim, shared_hidden_layers, device):
        super().__init__(
            input_dim=latent_dim+action_dim,
            hidden_dim=hidden_dim,
            shared_hidden_layers=shared_hidden_layers,
            output_dim=latent_dim,
            device=device
        )
        
class Reward(DiagonalCovarianceStochasticMLP):
    def __init__(self, latent_dim, hidden_dim, shared_hidden_layers, device):
        super().__init__(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            shared_hidden_layers=shared_hidden_layers,
            output_dim=1,
            device=device
        )
 
class Student(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(
            input_dim=cfg.teacher_latent_dim,
            hidden_dim=cfg.encoder_hidden_dim,
            shared_hidden_layers=cfg.encoder_shared_hidden_layers,
            latent_dim=cfg.student_latent_dim,
            device=cfg.device
        )
        self.dynamics = Dynamics(
            latent_dim=cfg.student_latent_dim,
            action_dim=cfg.action_dim,
            hidden_dim=cfg.dynamics_hidden_dim,
            shared_hidden_layers=cfg.dynamics_shared_hidden_layers,
            device=cfg.device
        )
        self.reward = Reward(
            latent_dim=cfg.student_latent_dim+cfg.action_dim,
            hidden_dim=cfg.dynamics_hidden_dim,
            shared_hidden_layers=cfg.dynamics_shared_hidden_layers,
            device=cfg.device
        )

    def stochastic_encode(self, z):
        zbar_mean, zbar_std = self.encoder(z)
        eps = torch.randn_like(zbar_mean)
        zbar = zbar_mean + zbar_std * eps
        return zbar
