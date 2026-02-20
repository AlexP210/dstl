import torch
import torch.nn as nn
import numpy as np

def diag_gaussian_log_prob(x, mean, std):
    var = std**2
    return -0.5 * (((x-mean)**2 / var) + torch.log(var) + np.log(2*torch.pi)).sum(-1)


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
    def __init__(self, teacher, cfg):
        super().__init__()
        self.teacher = teacher
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

    def compute_loss(self, batch):
        # Get the student latent
        z = self.teacher.model.encode(batch["observation"])
        zbar_mean, zbar_std = self.encoder(z)
        eps = torch.randn_like(zbar_mean)
        zbar = zbar_mean + zbar_std * eps

        # Concatenate the latent and action for the prediction
        a = batch["action"]
        zbar_and_a = torch.cat([zbar, a], dim=-1)

        # Mean and StDev for predicted next latents & reward
        predicted_zbar_prime_mean, predicted_zbar_prime_std = self.dynamics(zbar_and_a)
        predicted_r_mean, predicted_r_std = self.reward(zbar_and_a)

        # Reward LL
        true_r = batch["reward"]
        reward_ll = diag_gaussian_log_prob(true_r, predicted_r_mean, predicted_r_std)
        
        # Next latent LL
        z_prime = self.teacher.model.encode(batch["next_observation"])
        true_zbar_prime_mean, true_zbar_prime_std = self.encoder(z_prime)
        eps = torch.randn_like(true_zbar_prime_mean)
        true_zbar_prime = true_zbar_prime_mean + true_zbar_prime_std * eps
        zbar_prime_ll = diag_gaussian_log_prob(true_zbar_prime, predicted_zbar_prime_mean, predicted_zbar_prime_std)
        
        # KL Divergence
        zbar_var = zbar_std**2
        kl = 0.5*(
            torch.sum(zbar_mean*zbar_mean, dim=-1)
            + torch.sum(zbar_var, dim=-1) 
            - torch.sum(torch.log(zbar_var), dim=-1)
            - self.cfg.student_latent_dim
        )
        # Loss
        mean_reward_ll = torch.mean(reward_ll)
        mean_next_student_latent_ll = torch.mean(zbar_prime_ll)
        mean_kl = torch.mean(kl)
        loss = (-(mean_reward_ll + mean_next_student_latent_ll) + self.cfg.beta*mean_kl)
        return loss, mean_reward_ll, mean_next_student_latent_ll, mean_kl
