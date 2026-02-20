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
import pandas as pd

dataset_path = Path("/data/AlexPleava/outputs/dstl_pretrain_rl/BoxPlace-Direct-v0/1/2026-02-10_00-51-50/data/transitions.h5")
teacher_path = Path("/data/AlexPleava/outputs/dstl_pretrain_rl/BoxPlace-Direct-v0/1/2026-02-10_00-51-50/models/900864.pt")
student_path = Path("/home/AlexPleava/projects/distill-plan/dstl/dstl/outputs/2026-02-19/02-24-03/student.pt")
teacher_config_path = Path("/data/AlexPleava/outputs/dstl_pretrain_rl/BoxPlace-Direct-v0/1/2026-02-10_00-51-50/params/agent.yaml")
student_config_path = Path("/home/AlexPleava/projects/distill-plan/dstl/dstl/config.yaml")


teacher_config = OmegaConf.load(teacher_config_path)
teacher_config.obs_shape = {"rgb": (9, 64, 64)}
teacher_config.action_dim = 7
teacher_config.task_dim = 0
teacher_config.episode_length = 300
teacher_config.seed_steps = 1000
teacher_config.num_envs = 1
teacher_config.device = "cuda:0"
teacher = DSTL(teacher_config)
teacher.load(teacher_path)

student_config = OmegaConf.load(student_config_path)
student = Student(teacher=teacher, cfg=student_config)
student_state_dict = torch.load(student_path)
print(student_state_dict.keys())
student.load_state_dict(student_state_dict)

transition_dataset = TransitionDataset(dataset_path, 10_000)
train_loader, validation_loader, test_loader = transition_dataset.split(0.7, 0.15)

N_latent_dims = 8

# Estimate how much information the petal length and its width have in common
for train_batch in train_loader:
    z = teacher.model.encode(train_batch["observation"])
    zbar = student.encoder(z)
    train_batch["teacher_latent"] = z
    train_batch["student_latent"] = zbar
    print(f"Computing {N_latent_dims} embedding of teacher latents")
    teacher_embedding, teacher_embedder = lmi.estimate(
        Xs=train_batch["teacher_latent"], 
        Ys=train_batch["reward"], 
        N_dims=N_latent_dims, 
        epochs=10
    )
    print(f"Computing {N_latent_dims} embedding of student latents")
    student_embedding, student_embedder = lmi.estimate(
        Xs=train_batch["teacher_latent"], 
        Ys=train_batch["reward"], 
        N_dims=N_latent_dims, 
        epochs=10
    )

    print("Preparing dataset for MINE estimation")
    dataset = pd.DataFrame()
    dataset["teacher_latent"] = train_batch["teacher_latent"].item()
    dataset["student_latent"] = train_batch["student_latent"].item()
    dataset["reward"] = train_batch["reward"].item()
    dims = infer_dims(train_batch)

    print(f"Computing MINE estimate of teacher latents")
    teacher_mine_estimator = mine(
		x_dim=dims["teacher_latent"],
		y_dim=dims["reward"],
		hidden_dims=[128,]
	)
    teacher_mine_estimator.lower_bound = True
    teacher_estimator = MultiMIEstimator({("teacher_latent", "reward"): teacher_mine_estimator})
    teacher_estimated_mis, train_log = estimate_mi(
    	data=dataset,          # The dataset (as a pandas.DataFrame, many other formats are supported)
		x_key='teacher_latent',
		y_key='reward',
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
		y_dim=dims["reward"],
		hidden_dims=[128,]
	)
    student_mine_estimator.lower_bound = True
    student_estimator = MultiMIEstimator({("student_latent", "reward"): student_mine_estimator})
    student_estimated_mis, train_log = estimate_mi(
    	data=dataset,          # The dataset (as a pandas.DataFrame, many other formats are supported)
		x_key='student_latent',
		y_key='reward',
		valid_percentage=0.1,
		test_percentage=0.1,
		patience=1,
		estimator=student_estimator,           # Use the MINE mutual information estimator
		max_epochs=30,        # Number of maximum train iterations 
		device="cuda:0",
		verbose=True,
		batch_size=512,
	)

    teacher_mi = teacher_estimated_mis["I(teacher_latent;reward)"]
    student_mi = student_estimated_mis["I(student_latent;reward)"]

    print(f"ESTIMATED MI: Student = {student_mi} | Teacher = {teacher_mi}")

