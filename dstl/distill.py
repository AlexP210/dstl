from dstl.agent import DSTL
from pathlib import Path
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset
from tensordict import stack
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import wandb
import pandas as pd
from latentmi import lmi
from torch_mist.estimators.discriminative.factories import mine
from torch_mist.utils.estimation import infer_dims
from torch_mist.estimators.multi import MultiMIEstimator
from torch_mist import estimate_mi

import warnings
from pydantic._internal._generate_schema import UnsupportedFieldAttributeWarning

warnings.filterwarnings(
    "ignore",
    message="The 'repr' attribute with value False was provided to the `Field()` function"
)
warnings.filterwarnings(
    "ignore",
    message="The 'frozen' attribute with value True was provided to the `Field()` function"
)
from transition_data import TransitionDataset
from student import Student

def diag_gaussian_log_prob(x, mean, std):
    var = std**2
    return -0.5 * (((x-mean)**2 / var) + torch.log(var) + np.log(2*torch.pi)).sum(-1)

def squeeze_collate(batch):
    return batch[0]

class DistillationTrainer:

    def __init__(self, cfg):
        self.cfg = cfg

        # Prepare the teacher
        self.initialize_teacher()

        # prepare the student
        self.initialize_student()

        # Create the train, validation, and test dataloaders
        self.split_datasets()

        # Precompute the teacher latents to save time
        self.precompute_latents()


    def train(self):

        # Prepare the optimizer
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.cfg.learning_rate,
        )
        
        # Initialize W&B
        if self.cfg.use_wandb:
            wandb.init(
                project="Distilling-Planners",
                name=self.cfg.run_name,
                config=OmegaConf.to_container(self.cfg)
            )

        # Train
        iterations_since_last_eval = float("inf")
        iterations = 0
        for epoch in (epoch_bar:=tqdm(range(self.cfg.epochs), desc="Epochs", position=0)):
            for i, train_batch in enumerate(iteration_bar:=tqdm(self.train_loader, desc="Iterations", position=1, leave=False)):
                
                # Performing a training iteration
                optimizer.zero_grad()
                # train_batch = train_batch.to(self.cfg.device)
                loss, info = self.compute_loss(
                    batch=train_batch,
                    teacher_latent=self.train_teacher_latents[i],
                    next_teacher_latent=self.train_next_teacher_latents[i]
                )
                loss.backward()
                optimizer.step()
                if self.cfg.use_wandb:
                    logged_metrics = {f"train/{key}": val for key, val in info.items()}
                    wandb.log(data=logged_metrics, step=iterations)
                iteration_bar.set_postfix({
                    "train_loss":f"{loss.item():.2f}",
                })

                # Evaluation if we need it
                if iterations_since_last_eval > self.cfg.validation_interval*len(self.train_loader):
                    info = self.evaluate()
                    if self.cfg.use_wandb:
                        logged_metrics = {f"validation/{key}": val for key, val in info.items()}
                        wandb.log(data=logged_metrics, step=iterations)
                    iterations_since_last_eval = 0
                else:
                    iterations_since_last_eval += 1
                iterations += 1
            torch.save(self.student.state_dict(), Path(self.cfg.save_path) / f"student_{epoch}.pt")
            torch.save(optimizer.state_dict(), Path(self.cfg.save_path) / f"student_optimizer_{epoch}.pt")
        OmegaConf.save(self.cfg, Path(self.cfg.save_path) / "student_config.yaml")
        if self.cfg.use_wandb: wandb.finish()                

    def split_datasets(self):
        # Provided different datasets for training and evaluation
        # Evaluation is split into validation and test based on cfg.validation_fraction
        if self.cfg.training_data != self.cfg.evaluation_data:
            # Create the training dataset (only use `training_data_use_fraction` of it)
            training_dataset = TransitionDataset(Path(self.cfg.training_data), batch_size=self.cfg.batch_size, device=self.cfg.device)
            training_idxs = np.random.choice(
                a=len(training_dataset), size=int(self.cfg.training_data_use_fraction*len(training_dataset)), replace=False
            )
            self.training_dataset = Subset(dataset=training_dataset, indices=training_idxs)

            # Create the evaluation dataset (only use `evaluation_data_use_fraction` of it)
            evaluation_dataset = TransitionDataset(Path(self.cfg.evaluation_data), batch_size=self.cfg.batch_size, device=self.cfg.device)
            evaluation_idxs = np.random.choice(
                a=len(evaluation_dataset), size=int(self.cfg.evaluation_data_use_fraction*len(evaluation_dataset)), replace=False
            )
            evaluation_dataset = Subset(dataset=evaluation_dataset, indices=evaluation_idxs)

            # Create the validation & test datasets from the evaluation dataset
            # By randomly subsetting based on validation_fraction
            evaluation_idxs = np.arange(len(evaluation_dataset))
            np.random.shuffle(evaluation_idxs)
            validation_idxs = evaluation_idxs[:int(len(evaluation_idxs)*self.cfg.validation_fraction)]
            test_idxs = evaluation_idxs[int(len(evaluation_idxs)*self.cfg.validation_fraction):]
            self.validation_dataset = Subset(evaluation_dataset, validation_idxs)
            self.test_dataset = Subset(evaluation_dataset, test_idxs)
        
        # Provided the same datasets for training and evaluation
        # Evaluation is split into training, validation and test based on cfg.training_fraction, cfg.validation_fraction
        else:
            assert type(self.cfg.training_fraction) == float, "When the same dataset is used for training and validation,"
            "`training_fraction` must be provided."
            # Create the dataset (only use `training_data_use_fraction` of it)
            dataset = TransitionDataset(Path(self.cfg.training_data), batch_size=self.cfg.batch_size)
            idxs = np.random.choice(
                a=len(dataset), size=int(self.cfg.training_data_use_fraction*len(dataset)), replace=False
            )
            dataset = Subset(dataset=dataset, indices=idxs)
            
            # Split the dataset
            n = len(dataset)
            dataset_idxs = torch.arange(0, n).numpy()
            np.random.shuffle(dataset_idxs)
            
            training_idxs = dataset_idxs[:int((self.cfg.training_fraction)*n)]
            validation_idxs = dataset_idxs[int((self.cfg.training_fraction)*n):int((self.cfg.training_fraction+self.cfg.validation_fraction)*n)]
            test_idxs = dataset_idxs[int((self.cfg.training_fraction+self.cfg.validation_fraction)*n):]
        
            self.training_dataset = Subset(dataset=dataset, indices=training_idxs)
            self.validation_dataset = Subset(dataset=dataset, indices=validation_idxs)
            self.test_dataset = Subset(dataset=dataset, indices=test_idxs)

        # Create the training dataloader
        train_sampler = RandomSampler(data_source=self.training_dataset)
        self.train_loader = DataLoader(dataset=self.training_dataset, collate_fn=squeeze_collate, sampler=train_sampler, batch_size=1)

        # Create the validation dataloader
        validation_sampler = RandomSampler(data_source=self.validation_dataset)
        self.validation_loader = DataLoader(dataset=self.validation_dataset, collate_fn=squeeze_collate, sampler=validation_sampler, batch_size=1)

        # Create the test dataloader
        test_sampler = RandomSampler(data_source=self.test_dataset)
        self.test_loader = DataLoader(dataset=self.test_dataset, collate_fn=squeeze_collate, sampler=test_sampler, batch_size=1)

    def precompute_latents(self):
        self.train_teacher_latents = torch.empty(
            size=(len(self.training_dataset), self.cfg.batch_size, self.cfg.teacher_latent_dim), device=self.cfg.device
        )
        self.train_next_teacher_latents = torch.empty(
            size=(len(self.training_dataset), self.cfg.batch_size, self.cfg.teacher_latent_dim), device=self.cfg.device
        )
        for i, batch in enumerate(tqdm(self.train_loader, desc="Precomputing Training Teacher Latents")):
            self.train_teacher_latents[i] = self.teacher.model.encode(batch["observation"])
            self.train_next_teacher_latents[i] = self.teacher.model.encode(batch["next_observation"])
        
        self.validation_teacher_latents = torch.empty(
            size=(len(self.validation_dataset), self.cfg.batch_size, self.cfg.teacher_latent_dim), device=self.cfg.device
        )
        self.validation_next_teacher_latents = torch.empty(
            size=(len(self.validation_dataset), self.cfg.batch_size, self.cfg.teacher_latent_dim), device=self.cfg.device
        )
        for i, batch in enumerate(tqdm(self.validation_loader, desc="Precomputing Validation Teacher Latents")):
            self.validation_teacher_latents[i] = self.teacher.model.encode(batch["observation"])
            self.validation_next_teacher_latents[i] = self.teacher.model.encode(batch["next_observation"])

    def initialize_teacher(self):
        # Prepare the teacher
        teacher_cfg_path = Path(self.cfg.teacher_config)
        teacher_cfg = OmegaConf.load(teacher_cfg_path)
        teacher_cfg.obs_shape = {"rgb": (9, 64, 64)}
        teacher_cfg.action_dim = 7
        teacher_cfg.task_dim = 0
        teacher_cfg.episode_length = 300
        teacher_cfg.seed_steps = 1000
        teacher_cfg.num_envs = 1
        teacher_cfg.device = self.cfg.device
        teacher = DSTL(teacher_cfg)
        teacher.load(self.cfg.teacher_model)
        teacher.to(device=self.cfg.device)
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()
        self.teacher = teacher

    def initialize_student(self):
        # Prepare the student
        student = Student(cfg=self.cfg)
        if self.cfg.student_encoder_checkpoint is not None:
            state_dict = torch.load(self.cfg.student_encoder_checkpoint)
            student.encoder.load_state_dict(state_dict)
        if self.cfg.student_dynamics_checkpoint is not None:
            state_dict = torch.load(self.cfg.student_dynamics_checkpoint)
            student.dynamics.load_state_dict(state_dict)
        if self.cfg.student_reward_checkpoint is not None:
            state_dict = torch.load(self.cfg.student_reward_checkpoint)
            student.reward.load_state_dict(state_dict)
        self.student = student

    def compute_loss(self, batch, teacher_latent=None, next_teacher_latent=None):
        # Get the student latent
        if teacher_latent is None: z = self.teacher.model.encode(batch["observation"])
        else: z = teacher_latent
        zbar_mean, zbar_std = self.student.encoder(z)
        eps = torch.randn_like(zbar_mean)
        zbar = zbar_mean + zbar_std * eps

        # Concatenate the latent and action for the prediction
        a = batch["action"]
        zbar_and_a = torch.cat([zbar, a], dim=-1)

        # Mean and StDev for predicted next latents & reward
        predicted_zbar_prime_mean, predicted_zbar_prime_std = self.student.dynamics(zbar_and_a)
        predicted_r_mean, predicted_r_std = self.student.reward(zbar_and_a)

        # Reward LL
        true_r = batch["reward"]
        reward_ll = diag_gaussian_log_prob(true_r, predicted_r_mean, predicted_r_std)
        
        # Next latent LL
        if next_teacher_latent is None: z_prime = self.teacher.model.encode(batch["next_observation"])
        else: z_prime = next_teacher_latent
        true_zbar_prime_mean, true_zbar_prime_std = self.student.encoder(z_prime)
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
        loss = (-(self.cfg.reward_ll_weight*mean_reward_ll + self.cfg.dynamics_ll_weight*mean_next_student_latent_ll) + self.cfg.beta*mean_kl)
        return loss, {
            "loss": loss,
            "reward_ll": self.cfg.reward_ll_weight*mean_reward_ll,
            "dynamics_ll": self.cfg.dynamics_ll_weight*mean_next_student_latent_ll,
            "kl": self.cfg.beta*mean_kl
        }
    
    def estimate_mutual_informations(self):
        # Split the validation dataset further into an MI-train and MI-val dataset
        validation_idxs = np.arange(start=0, stop=len(self.validation_dataset))
        np.random.shuffle(validation_idxs)
        mi_train_idxs = validation_idxs[:int(len(validation_idxs)*(1-self.cfg.lmi_validate_fraction))]
        mi_val_idxs = validation_idxs[int(len(validation_idxs)*(1-self.cfg.lmi_validate_fraction)):]
        mi_train_dataset = Subset(self.validation_dataset, mi_train_idxs)
        mi_val_dataset = Subset(self.validation_dataset, mi_val_idxs)
        mi_train_sampler = RandomSampler(data_source=mi_train_dataset)
        mi_val_sampler = RandomSampler(data_source=mi_val_dataset)
        mi_train_dataloader = DataLoader(dataset=mi_train_dataset, collate_fn=squeeze_collate, sampler=mi_train_sampler, batch_size=1)
        mi_val_dataloader = DataLoader(dataset=mi_val_dataset, collate_fn=squeeze_collate, sampler=mi_val_sampler, batch_size=1)
        
        info = {}
        for i, target_variable_key in enumerate(("state", "reward")):

            # Prepare the LMI embedding model
            example_datum = mi_val_dataset[0][target_variable_key][0]
            lmi_encoder = lmi.models.AECross(
                x_dim=self.cfg.student_latent_dim,
                y_dim=example_datum.shape[0],
                latent_size=self.cfg.lmi_latent_size,
            ).to(self.cfg.device)

            # Prepare the stuff to train LMI embedding model
            optimizer = torch.optim.Adam(lmi_encoder.parameters(), lr=self.cfg.lmi_lr, eps=1e-07) 
            val_losses = []
            early_stopper = lmi.EarlyStopper(patience=self.cfg.lmi_patience)

            # Train the LMI embedding model
            lmi_training_bar = tqdm(
                iterable=range(self.cfg.lmi_epochs*len(mi_train_dataloader)), 
                desc=f"{target_variable_key} LMI", position=3+i, leave=False
            )
            val_loss = None
            for epoch in range(self.cfg.lmi_epochs):

                # Train 1 epoch
                for batch in mi_train_dataloader:
                    teacher_latent = self.teacher.model.encode(batch["observation"])
                    student_latent = self.student.stochastic_encode(teacher_latent)
                    train_loss = lmi_encoder.learning_loss(student_latent, batch[target_variable_key])
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()
                    lmi_training_bar.set_postfix(train_loss=train_loss.item(), val_loss=val_loss)
                    lmi_training_bar.update(1)

                # Validate 
                with torch.no_grad():
                    epoch_validate_loss = []
                    for batch in mi_val_dataloader:
                        teacher_latent = self.teacher.model.encode(batch["observation"])
                        student_latent = self.student.stochastic_encode(teacher_latent)
                        epoch_validate_loss.append(lmi_encoder.learning_loss(student_latent, batch[target_variable_key]).item())
                    val_loss = np.mean(epoch_validate_loss).item()
                    val_losses.append(val_loss)
                    lmi_training_bar.set_postfix(train_loss=train_loss.item(), val_loss=val_loss)

                # Whether to stop early
                es = early_stopper.early_stop(val_losses[-1], lmi_encoder)
                if es:
                    lmi_encoder.load_state_dict(es)
                    break

            # Encode the data
            Z_Xs = []
            Z_Ys = []
            for batch in self.validation_loader:
                teacher_latent = self.teacher.model.encode(batch["observation"])
                student_latent = self.student.stochastic_encode(teacher_latent)
                Z_X, Z_Y = lmi_encoder.encode(student_latent, batch[target_variable_key])
                Z_Xs.extend(Z_X.tolist())
                Z_Ys.extend(Z_Y.tolist())
            
            # Estimate MI with KSG
            mi_estimate = lmi.ksg.mi(Z_Xs, Z_Ys)
            info[f"mi_{target_variable_key}"] = mi_estimate

        return info

    def estimate_validation_loss(self):
        total_validation_loss = 0
        loss_validation_bar = tqdm(self.validation_loader, desc="Val Loss", position=2, leave=False)
        for i, validation_batch in enumerate(loss_validation_bar):
            with torch.no_grad():
                validation_batch = validation_batch.to(self.cfg.device)
                validation_loss, info = self.compute_loss(
                    validation_batch, 
                    self.validation_teacher_latents[i], 
                    self.validation_next_teacher_latents[i]
                )
                total_validation_loss += validation_loss
        info = {f"loss": total_validation_loss/len(self.validation_loader)}
        return info

    def evaluate(self):
        # Turn the student to eval mode
        self.student.eval()
        self.student.requires_grad_(False)

        info = {}
        # # Validation loss
        # info.update(self.estimate_validation_loss())
        # # Estimate MI
        # info.update(self.estimate_mutual_informations())

        # Turn the student back to train mode
        self.student.train()
        self.student.requires_grad_(True)
        return info

@hydra.main(config_name='config', config_path='.', version_base="1.2")
def main(cfg):
    trainer = DistillationTrainer(cfg=cfg)
    trainer.train()

if __name__ == "__main__":
    main()