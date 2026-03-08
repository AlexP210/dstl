import torch
import h5py
from tensordict import TensorDict
from torch.utils.data import Dataset


class TransitionDataset(Dataset):
    def __init__(self, path, batch_size, load_device="cpu", batch_device="cpu", dtype=torch.float32):
        self.path = path
        self.load_device = load_device
        self.batch_device = batch_device
        self.dtype = dtype
        self.batch_size = batch_size

        self._loaded = False

        with h5py.File(self.path, "r") as f:
            n = f["o"].shape[0]

        self.length = n // self.batch_size

        # placeholders for RAM mode
        self.observation = None
        self.action = None
        self.reward = None
        self.next_observation = None
        self.terminated = None
        self.truncated = None
        self.state = None
        self.next_state = None

    def load(self):
        if self._loaded:
            return

        with h5py.File(self.path, "r") as f:
            self.observation = torch.as_tensor(f["o"][:], dtype=self.dtype, device=self.load_device)
            self.action = torch.as_tensor(f["a"][:], dtype=self.dtype, device=self.load_device)
            self.reward = torch.as_tensor(f["r"][:], dtype=self.dtype, device=self.load_device)
            self.next_observation = torch.as_tensor(f["oprime"][:], dtype=self.dtype, device=self.load_device)
            self.terminated = torch.as_tensor(f["terminated"][:], dtype=self.dtype, device=self.load_device)
            self.truncated = torch.as_tensor(f["truncated"][:], dtype=self.dtype, device=self.load_device)
            self.state = torch.as_tensor(f["state"][:], dtype=self.dtype, device=self.load_device)
            self.next_state = torch.as_tensor(f["state_prime"][:], dtype=self.dtype, device=self.load_device)

        self._loaded = True

    def unload(self):
        if not self._loaded:
            return

        self.observation = None
        self.action = None
        self.reward = None
        self.next_observation = None
        self.terminated = None
        self.truncated = None
        self.state = None
        self.next_state = None

        self._loaded = False

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        stop_idx = (idx + 1) * self.batch_size

        if self._loaded:
            obs = self.observation[start_idx:stop_idx]
            act = self.action[start_idx:stop_idx]
            rew = self.reward[start_idx:stop_idx]
            next_obs = self.next_observation[start_idx:stop_idx]
            term = self.terminated[start_idx:stop_idx]
            trunc = self.truncated[start_idx:stop_idx]
            state = self.state[start_idx:stop_idx]
            next_state = self.next_state[start_idx:stop_idx]
        else:
            # Read slice directly from disk
            with h5py.File(self.path, "r") as f:
                obs = torch.as_tensor(f["o"][start_idx:stop_idx], dtype=self.dtype, device=self.load_device)
                act = torch.as_tensor(f["a"][start_idx:stop_idx], dtype=self.dtype, device=self.load_device)
                rew = torch.as_tensor(f["r"][start_idx:stop_idx], dtype=self.dtype, device=self.load_device)
                next_obs = torch.as_tensor(f["oprime"][start_idx:stop_idx], dtype=self.dtype, device=self.load_device)
                term = torch.as_tensor(f["terminated"][start_idx:stop_idx], dtype=self.dtype, device=self.load_device)
                trunc = torch.as_tensor(f["truncated"][start_idx:stop_idx], dtype=self.dtype, device=self.load_device)
                state = torch.as_tensor(f["state"][start_idx:stop_idx], dtype=self.dtype, device=self.load_device)
                next_state = torch.as_tensor(f["state_prime"][start_idx:stop_idx], dtype=self.dtype, device=self.load_device)

        td = TensorDict(
            {
                "observation": obs,
                "action": act,
                "reward": rew,
                "next_observation": next_obs,
                "terminated": term,
                "truncated": trunc,
                "state": state,
                "next_state": next_state,
            },
            device=self.batch_device,
            batch_size=self.batch_size,
        )

        return td