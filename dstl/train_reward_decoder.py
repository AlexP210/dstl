from dstl.agent import DSTL
import argparse
from pathlib import Path
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, BatchSampler, RandomSampler, SequentialSampler
from torch.distributions import Normal, Independent
from tensordict import TensorDict, stack
from sklearn.datasets import load_iris
import pandas as pd
import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf
from dstl.common.parser import parse_cfg
from tqdm import tqdm
import numpy as np
import wandb

def diag_gaussian_log_prob(x, mean, std):
    var = std**2
    return -0.5 * (((x-mean)**2 / var) + torch.log(var) + np.log(2*torch.pi)).sum(-1)

# Dataset class
class TransitionDataset(Dataset):
    def __init__(self, path, batch_size, device="cpu", dtype=torch.float32):
        self.path = path
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.file = None

        # Open once to get length
        with h5py.File(self.path, "r") as f:
            self.length = f["state"].shape[0]

    def _ensure_open(self):
        if self.file is None:
            self.file = h5py.File(self.path, "r")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._ensure_open()
        f = self.file
        td = TensorDict(
            {
                "observation": torch.as_tensor(f["o"][idx:idx+self.batch_size], dtype=self.dtype),
                "action": torch.as_tensor(f["a"][idx:idx+self.batch_size], dtype=self.dtype),
                "reward": torch.as_tensor(f["r"][idx:idx+self.batch_size], dtype=self.dtype),
                "next_observation": torch.as_tensor(f["oprime"][idx:idx+self.batch_size], dtype=self.dtype),
                "terminated": torch.as_tensor(f["terminated"][idx:idx+self.batch_size], dtype=self.dtype),
                "truncated": torch.as_tensor(f["truncated"][idx:idx+self.batch_size], dtype=self.dtype),
                "state": torch.as_tensor(f["state"][idx:idx+self.batch_size], dtype=self.dtype),
                "next_state": torch.as_tensor(f["state_prime"][idx:idx+self.batch_size], dtype=self.dtype)
            },
            device=self.device,
        )
        return td

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
    def __init__(self, latent_dim, hidden_dim, shared_hidden_layers, device, teacher):
        super().__init__(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            shared_hidden_layers=shared_hidden_layers,
            output_dim=1,
            device=device
        )
        self.teacher = teacher

    def compute_loss(self, batch):
        # Get the student latent
        z = self.teacher.model.encode(batch["observation"])

        # Concatenate the latent and action for the prediction
        a = batch["action"]
        z_and_a = torch.cat([z, a], dim=-1)

        # Mean and StDev for predicted next latents & reward
        predicted_r_mean, predicted_r_std = self.forward(z_and_a)

        # Reward LL
        true_r = batch["reward"]
        reward_ll = diag_gaussian_log_prob(true_r, predicted_r_mean, predicted_r_std)
        
        mean_reward_ll = torch.mean(reward_ll)
        loss = -mean_reward_ll
        return loss
 
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
    
@hydra.main(config_name='config', config_path='.')
def main(cfg):
    student, training_loss, validation_loss = train(cfg)

def train(cfg):

    logdir = Path(cfg.logdir)
    if not logdir.exists():
        raise FileNotFoundError(f"Log directory does not exist: {logdir}")
    if not logdir.is_dir():
        raise NotADirectoryError(f"Not a directory: {logdir}")
    dataset_path = logdir / "data" / "training.h5"

    transition_dataset = TransitionDataset(dataset_path, batch_size=cfg.batch_size)

    n = len(transition_dataset)
    batch_starts = torch.arange(0, n - cfg.batch_size + 1, cfg.batch_size)
    np.random.shuffle(batch_starts)
    train_len = int(len(batch_starts) * 0.7)
    train_batch_starts = batch_starts[:train_len]
    validation_len = int(len(batch_starts) * 0.15)
    validation_batch_starts = batch_starts[train_len:train_len+validation_len]

    # train_dataset, validation_dataset, test_dataset = random_split(transition_dataset, [train_len, validation_len, test_len])
    train_batch_start_sampler = RandomSampler(data_source=train_batch_starts)
    validation_batch_start_sampler = RandomSampler(data_source=validation_batch_starts)
    train_loader = DataLoader(dataset=transition_dataset, collate_fn=stack, sampler=train_batch_start_sampler, batch_size=1)
    validation_loader = DataLoader(dataset=transition_dataset, collate_fn=stack, sampler=validation_batch_start_sampler, batch_size=1)

    teacher_model_path = Path("/data/AlexPleava/outputs/dstl_pretrain_rl/BoxPlace-Direct-v0/1/2026-02-10_00-51-50/")
    teacher_cfg = OmegaConf.load(teacher_model_path / "params" / "agent.yaml")
    teacher_cfg.obs_shape = {"rgb": (9, 64, 64)}
    teacher_cfg.action_dim = 7
    teacher_cfg.task_dim = 0
    teacher_cfg.episode_length = 300
    teacher_cfg.seed_steps = 1000
    teacher_cfg.num_envs = 1
    teacher_cfg.device = cfg.device
    teacher = DSTL(teacher_cfg)
    teacher.load(teacher_model_path / "models" / "final.pt")
    teacher.to(device=cfg.device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    reward_decoder = Reward(
        latent_dim=512+7,
        hidden_dim=cfg.reward_hidden_dim,
        shared_hidden_layers=cfg.reward_shared_hidden_layers,
        device=cfg.device,
        teacher=teacher
    )

    optimizer = torch.optim.Adam(
        reward_decoder.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0
    )

    wandb.init(
        project="Distilling-Planners",
        name=f"Reward Decoder",  # optional unique run name
    )


    training_losses = []
    validation_losses = []
    epoch_bar = tqdm(range(cfg.epochs), desc="Epochs", position=0)
    for epoch in epoch_bar:
        iterations_since_last_eval = float("inf")
        iteration_bar = tqdm(train_loader, desc="Iterations", position=1, leave=False)
        for train_batch in iteration_bar:
            global_step = epoch * len(train_loader) + iteration_bar.n
            optimizer.zero_grad()
            train_batch = train_batch.to(cfg.device)
            train_loss = reward_decoder.compute_loss(batch=train_batch)
            training_losses.append(train_loss)
            train_loss.backward()
            optimizer.step()
            metrics = {
                "loss": train_loss.item(),
            }
            wandb.log(data={f"train/{key}":val for key, val in metrics.items()}, step=global_step)

            iteration_bar.set_postfix({
                "train_loss":f"{train_loss.item():.2f}",
            })
            if iterations_since_last_eval > cfg.validation_interval*len(train_loader):
                reward_decoder.eval()
                total_validation_loss = 0
                for validation_batch in validation_loader:
                    with torch.no_grad():
                        validation_batch = validation_batch.to(cfg.device)
                        validation_loss = reward_decoder.compute_loss(validation_batch)
                        total_validation_loss += validation_loss
                total_validation_loss /= len(validation_loader)
                wandb.log(
                    data = {
                        "validation/loss": total_validation_loss,
                    },
                    step=global_step
                )
                for name, param in reward_decoder.named_parameters():
                    if param.grad is not None:
                        wandb.log({f"grad/{name}": wandb.Histogram(param.grad.cpu())}, step=global_step)
                        wandb.log({f"weights/{name}": wandb.Histogram(param.data.cpu())}, step=global_step)
                iteration_bar.set_postfix(val_loss=f"{total_validation_loss:.2f}")
                iterations_since_last_eval = 0
                validation_losses.append(validation_loss)
                reward_decoder.train()
            iterations_since_last_eval += 1
    wandb.finish()
    torch.save(reward_decoder.state_dict(), "reward.pt")
    return reward_decoder, training_losses, validation_loss
                

if __name__ == "__main__":
    main()