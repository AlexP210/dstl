from torch_mist import estimate_mi
import torch
from sklearn.datasets import load_iris
from pathlib import Path
from dstl.agent import DSTL
from student import Student
from transition_data import TransitionDataset
from omegaconf import OmegaConf
from latentmi import lmi
from torch_mist.estimators.discriminative.factories import mine
from torch_mist.utils.estimation import infer_dims
from torch_mist.estimators.multi import MultiMIEstimator
from torch.utils.data import DataLoader, RandomSampler
from tensordict import stack
import pandas as pd
import numpy as np
import hydra

@hydra.main(config_name='config', config_path='.')
def main(cfg):

    # Initialize the configs
    dataset_path = Path(cfg.transition_data)

    teacher_config_path = Path(cfg.teacher_config)
    teacher_config = OmegaConf.load(teacher_config_path)
    teacher_path = Path(cfg.teacher_model)

    student_config = cfg
    student_path = Path(cfg.save_path)


    teacher_config.obs_shape = {"rgb": (9, 64, 64)}
    teacher_config.action_dim = 7
    teacher_config.task_dim = 0
    teacher_config.episode_length = 300
    teacher_config.seed_steps = 1000
    teacher_config.num_envs = 1
    teacher_config.device = "cuda:0"

    # Create the models
    teacher = DSTL(teacher_config)
    teacher.load(teacher_path)
    teacher.to(cfg.device)
    for param in teacher.parameters():
        param.requires_grad = False
    
    student = Student(teacher=teacher, cfg=student_config)
    student.encoder.load_state_dict(torch.load(student_path / "student_encoder_40.pt"))
    student.dynamics.load_state_dict(torch.load(student_path / "student_dynamics_40.pt"))
    student.reward.load_state_dict(torch.load(student_path / "student_reward_40.pt"))
    student.to(cfg.device)
    for param in student.parameters():
        param.requires_grad = False    
    random_students = []
    for i in range(5):
        random_student = Student(teacher=teacher, cfg=student_config)
        for param in random_student.parameters():
            param.requires_grad = False
        random_student.to(cfg.device)
        random_students.append(random_student)

    # Prepare the dataset
    transition_dataset = TransitionDataset(dataset_path, 10_000, device=cfg.device)
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

    N_latent_dims = 8
    # Estimate how much information the petal length and its width have in common
    for train_batch in train_loader:
        z = teacher.model.encode(train_batch["observation"])
        z_prime = teacher.model.encode(train_batch["next_observation"])
        zbar = student.encoder(z)[0]
        zbar_prime = student.encoder(z_prime)[0]
        random_zbars = []
        random_zbar_primes = []
        for random_student in random_students:
            random_zbars.append(random_student.encoder(z)[0].cpu().numpy()[0])
            random_zbar_primes.append(random_student.encoder(z_prime)[0].cpu().numpy()[0])

        z = z.cpu().numpy()[0]
        zbar = zbar.cpu().numpy()[0]
        zbar_prime = zbar_prime.cpu().numpy()[0]
        r = train_batch["reward"].cpu().numpy()[0]
        s = train_batch["state"].cpu().numpy()[0]

        # print(f"Computing {N_latent_dims} embedding of teacher latents")
        # _, teacher_embedding, teacher_embedder = lmi.estimate(
        #     Xs=z, 
        #     Ys=r, 
        #     N_dims=N_latent_dims, 
        #     epochs=10
        # )

        # print(f"Computing {N_latent_dims} embedding of student latents")
        # _, student_embedding, student_embedder = lmi.estimate(
        #     Xs=zbar, 
        #     Ys=r, 
        #     N_dims=N_latent_dims, 
        #     epochs=10,
        #     device=cfg.device
        # )

        print("Preparing dataset for MINE estimation")
        dataset = pd.DataFrame(index=range(10_000))
        dataset["teacher_latent"] = list(z)
        dataset["student_latent"] = list(zbar)
        dataset["reward"] = list(r)
        dataset["state"] = list(s)
        for i, random_student in enumerate(random_students):
            dataset[f"random_student_latent_{i}"] = list(random_zbars[i])
        dims = infer_dims(dataset)

        print(f"Computing MINE estimate of teacher latents")
        teacher_mine_estimator = mine(
            x_dim=dims["teacher_latent"],
            y_dim=dims["state"],
            hidden_dims=[128,]
        )
        teacher_mine_estimator.lower_bound = True
        teacher_estimator = MultiMIEstimator({("teacher_latent", "state"): teacher_mine_estimator})
        teacher_estimated_mis, train_log = estimate_mi(
            data=dataset,          # The dataset (as a pandas.DataFrame, many other formats are supported)
            x_key='teacher_latent',
            y_key='state',
            valid_percentage=0.1,
            test_percentage=0.1,
            patience=1,
            estimator=teacher_estimator,           # Use the MINE mutual information estimator
            max_epochs=30,        # Number of maximum train iterations 
            device="cuda:0",
            verbose=True,
            batch_size=512,
        )

        print(f"Computing MINE estimate of student latents")
        student_mine_estimator = mine(
            x_dim=dims["student_latent"],
            y_dim=dims["state"],
            hidden_dims=[128,]
        )
        student_mine_estimator.lower_bound = True
        student_estimator = MultiMIEstimator({("student_latent", "state"): student_mine_estimator})
        student_estimated_mis, train_log = estimate_mi(
            data=dataset,          # The dataset (as a pandas.DataFrame, many other formats are supported)
            x_key='student_latent',
            y_key='state',
            valid_percentage=0.1,
            test_percentage=0.1,
            patience=1,
            estimator=student_estimator,           # Use the MINE mutual information estimator
            max_epochs=30,        # Number of maximum train iterations 
            device="cuda:0",
            verbose=True,
            batch_size=512,
        )
        teacher_mi = teacher_estimated_mis["I(teacher_latent;state)"]
        student_mi = student_estimated_mis["I(student_latent;state)"]

        print(f"Computing MINE estimate of random student latents")
        random_student_estimated_mis = []
        for i in range(len(random_students)):
            random_student_mine_estimator = mine(
                x_dim=dims[f"random_student_latent_{i}"],
                y_dim=dims["state"],
                hidden_dims=[128,]
            )
            random_student_mine_estimator.lower_bound = True
            random_student_estimator = MultiMIEstimator({(f"random_student_latent_{i}", "state"): random_student_mine_estimator})
            random_student_estimated_mi, train_log = estimate_mi(
                data=dataset,          # The dataset (as a pandas.DataFrame, many other formats are supported)
                x_key=f'random_student_latent_{i}',
                y_key='state',
                valid_percentage=0.1,
                test_percentage=0.1,
                patience=1,
                estimator=random_student_estimator,           # Use the MINE mutual information estimator
                max_epochs=30,        # Number of maximum train iterations 
                device="cuda:0",
                verbose=True,
                batch_size=512,
            )
            random_student_estimated_mis.append(random_student_estimated_mi[f"I(random_student_latent_{i};state)"])

        mean = np.mean(random_student_estimated_mis)
        std = np.std(random_student_estimated_mis)
        print(f"ESTIMATED MI: Student = {student_mi} | Teacher = {teacher_mi} | Random Students = {mean} +- {std}")

if __name__ == "__main__":
    main()
