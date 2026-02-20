from dstl.agent import DSTL
from pathlib import Path
import torch
from torch.utils.data import DataLoader, RandomSampler
from tensordict import stack
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import wandb

from transition_data import TransitionDataset
from student import Student

# Dataset class    
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

    # teacher_model_path = Path("/data/AlexPleava/outputs/dstl_task/BoxPlace-Direct-v0/1/2026-02-15_19-49-17/")
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
    teacher.load(teacher_model_path / "models" / "900864.pt")
    teacher.to(device=cfg.device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    student = Student(teacher=teacher, cfg=cfg)

    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0
    )

    wandb.init(
        project="Distilling-Planners",
        name=f"Distillation Test",  # optional unique run name
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
            train_loss, mean_reward_ll, mean_next_student_latent_ll, mean_kl = student.compute_loss(batch=train_batch)
            training_losses.append(train_loss)
            train_loss.backward()
            optimizer.step()
            metrics = {
                "loss": train_loss.item(),
                "reward_ll": mean_reward_ll.item(),
                "next_latent_ll": mean_next_student_latent_ll.item(),
                "mean_kl": mean_kl.item(),
            }
            wandb.log(data={f"train/{key}":val for key, val in metrics.items()}, step=global_step)

            iteration_bar.set_postfix({
                "train_loss":f"{train_loss.item():.2f}",
            })
            if iterations_since_last_eval > cfg.validation_interval*len(train_loader):
                student.eval()
                total_validation_loss = 0
                for validation_batch in validation_loader:
                    with torch.no_grad():
                        validation_batch = validation_batch.to(cfg.device)
                        validation_loss, _, _, _ = student.compute_loss(validation_batch)
                        total_validation_loss += validation_loss
                total_validation_loss /= len(validation_loader)
                wandb.log(
                    data = {
                        "validation/loss": total_validation_loss,
                    },
                    step=global_step
                )
                for name, param in student.named_parameters():
                    if param.grad is not None:
                        wandb.log({f"grad/{name}": wandb.Histogram(param.grad.cpu())}, step=global_step)
                        wandb.log({f"weights/{name}": wandb.Histogram(param.data.cpu())}, step=global_step)
                iteration_bar.set_postfix(val_loss=f"{total_validation_loss:.2f}")
                iterations_since_last_eval = 0
                validation_losses.append(validation_loss)
                student.train()
            iterations_since_last_eval += 1
    wandb.finish()
    torch.save(student.state_dict(), "student.pt")
    return student, training_losses, validation_loss
                

if __name__ == "__main__":
    main()