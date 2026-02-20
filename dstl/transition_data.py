import torch
import h5py
from tensordict import TensorDict
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from tensordict import stack

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

	def split(self, train_fraction, validation_fraction):
		batch_starts = torch.arange(0, self.length - self.batch_size + 1, self.batch_size)
		np.random.shuffle(batch_starts)
		train_len = int(len(batch_starts) * train_fraction)
		train_batch_starts = batch_starts[:train_len]
		validation_len = int(len(batch_starts) * validation_fraction)
		validation_batch_starts = batch_starts[train_len:train_len+validation_len]
		test_batch_starts = batch_starts[train_len+validation_len:]

		# train_dataset, validation_dataset, test_dataset = random_split(transition_dataset, [train_len, validation_len, test_len])
		train_batch_start_sampler = RandomSampler(data_source=train_batch_starts)
		validation_batch_start_sampler = RandomSampler(data_source=validation_batch_starts)
		test_batch_start_sampler = RandomSampler(data_source=test_batch_starts)
		train_loader = DataLoader(dataset=self, collate_fn=stack, sampler=train_batch_start_sampler, batch_size=1)
		validation_loader = DataLoader(dataset=self, collate_fn=stack, sampler=validation_batch_start_sampler, batch_size=1)
		test_loader = DataLoader(dataset=self, collate_fn=stack, sampler=test_batch_start_sampler, batch_size=1)
		return train_loader, validation_loader, test_loader