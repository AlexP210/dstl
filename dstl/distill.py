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

@hydra.main(config_name='config', config_path='.')
def main(cfg):

    # Prepare dataset
    dataset_path = Path(cfg.transition_data)
    # if not dataset_path.exists():
    #     raise FileNotFoundError(f"Log directory does not exist: {dataset_path}")
    # if not dataset_path.is_dir():
    #     raise NotADirectoryError(f"Not a directory: {dataset_path}")
    transition_dataset = TransitionDataset(dataset_path, batch_size=cfg.batch_size)
    n = len(transition_dataset)
    batch_starts = torch.arange(0, n - cfg.batch_size + 1, cfg.batch_size)
    np.random.shuffle(batch_starts)
    train_len = int(len(batch_starts) * 0.7)
    train_batch_starts = batch_starts[:train_len]
    validation_len = int(len(batch_starts) * 0.15)
    validation_batch_starts = batch_starts[train_len:train_len+validation_len]
    train_batch_start_sampler = RandomSampler(data_source=train_batch_starts)
    validation_batch_start_sampler = RandomSampler(data_source=validation_batch_starts)
    train_loader = DataLoader(dataset=transition_dataset, collate_fn=stack, sampler=train_batch_start_sampler, batch_size=1)
    validation_loader = DataLoader(dataset=transition_dataset, collate_fn=stack, sampler=validation_batch_start_sampler, batch_size=1)

    # Prepare the teacher
    teacher_cfg_path = Path(cfg.teacher_config)
    teacher_cfg = OmegaConf.load(teacher_cfg_path)
    teacher_cfg.obs_shape = {"rgb": (9, 64, 64)}
    teacher_cfg.action_dim = 7
    teacher_cfg.task_dim = 0
    teacher_cfg.episode_length = 300
    teacher_cfg.seed_steps = 1000
    teacher_cfg.num_envs = 1
    teacher_cfg.device = cfg.device
    teacher = DSTL(teacher_cfg)
    teacher.load(cfg.teacher_model)
    teacher.to(device=cfg.device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    # Prepare the student
    student = Student(teacher=teacher, cfg=cfg)
    if cfg.student_encoder_checkpoint is not None:
        state_dict = torch.load(cfg.student_encoder_checkpoint)
        student.encoder.load_state_dict(state_dict)
    if cfg.student_dynamics_checkpoint is not None:
        state_dict = torch.load(cfg.student_dynamics_checkpoint)
        student.dynamics.load_state_dict(state_dict)
    if cfg.student_reward_checkpoint is not None:
        state_dict = torch.load(cfg.student_reward_checkpoint)
        student.reward.load_state_dict(state_dict)

    # Prepare the optimizer
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg.learning_rate,
    )

    # Initialize W&B
    wandb.init(
        project="Distilling-Planners",
        name=f"Distillation Test",  # optional unique run name
    )

    # Train
    iterations_since_last_eval = float("inf")
    iterations = 0
    for epoch in (epoch_bar:=tqdm(range(cfg.epochs), desc="Epochs", position=0)):
        for train_batch in (iteration_bar:=tqdm(train_loader, desc="Iterations", position=1, leave=False)):

            # Performing a training iteration
            optimizer.zero_grad()
            train_batch = train_batch.to(cfg.device)
            train_loss, mean_reward_ll, mean_next_student_latent_ll, mean_kl = student.compute_loss(batch=train_batch)
            train_loss.backward()
            optimizer.step()
            wandb.log(data={
                "train/loss": train_loss.item(),
                "train/reward_ll": mean_reward_ll.item(),
                "train/next_latent_ll": mean_next_student_latent_ll.item(),
                "train/mean_kl": mean_kl.item(),
            }, step=iterations)
            iteration_bar.set_postfix({
                "train_loss":f"{train_loss.item():.2f}",
            })

            # Evaluation if we need it
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
                    step=iterations
                )
                for name, param in student.named_parameters():
                    if param.grad is not None:
                        wandb.log({f"grad/{name}": wandb.Histogram(param.grad.cpu())}, step=iterations)
                        wandb.log({f"weights/{name}": wandb.Histogram(param.data.cpu())}, step=iterations)
                iteration_bar.set_postfix(val_loss=f"{total_validation_loss:.2f}")
                iterations_since_last_eval = 0
                student.train()
            iterations += 1
            iterations_since_last_eval += 1
        torch.save(student.encoder.state_dict(), Path(cfg.save_path) / f"student_encoder_{epoch}.pt")
        torch.save(student.dynamics.state_dict(), Path(cfg.save_path) / f"student_dynamics_{epoch}.pt")
        torch.save(student.reward.state_dict(), Path(cfg.save_path) / f"student_reward_{epoch}.pt")
    OmegaConf.save(cfg, Path(cfg.save_path) / "student_config.yaml")
    wandb.finish()                

if __name__ == "__main__":
    main()