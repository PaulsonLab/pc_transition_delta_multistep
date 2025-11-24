import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np


class MultiStepTransitionDataset(Dataset):
    def __init__(self, states_seqs, actions_seqs, rollout_horizon=5):
        """
        states_seqs: list of numpy arrays, each [T_seq, material_nodes, 3]
        actions_seqs: list of numpy arrays, each [T_seq, actions_dim]
        """
        self.states_seqs = states_seqs
        self.actions_seqs = actions_seqs
        self.rollout_horizon = rollout_horizon
        self.samples = []  # list of (seq_idx, t)

        for seq_idx, (s_seq, a_seq) in enumerate(zip(states_seqs, actions_seqs)):
            T = len(a_seq)
            for t in range(T - rollout_horizon):
                self.samples.append((seq_idx, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_idx, t = self.samples[idx]
        s_seq = self.states_seqs[seq_idx]
        a_seq = self.actions_seqs[seq_idx]

        actions = a_seq[t : t+self.rollout_horizon]
        states_tpH = s_seq[t+1 : t+1+self.rollout_horizon]


        state_t = torch.from_numpy(s_seq[t]).float()  # [material_points, 3]
        actions = torch.from_numpy(a_seq[t : t+self.rollout_horizon]).float()  # [rollout_horizons, actions_dim]
        states_tpH = torch.from_numpy(s_seq[t+1 : t+1+self.rollout_horizon]).float()  # [rollout_horizons, material_points, 3]
        delta_tpH = states_tpH - state_t.unsqueeze(0)

        return state_t, actions, delta_tpH, states_tpH


def RandomSplit(datasets, train_set_percentage):
    lengths = [int(len(datasets)*train_set_percentage), len(datasets)-int(len(datasets)*train_set_percentage)]
    return random_split(datasets, lengths)


def GetMultiStepDataLoaders(states_list, actions_list, batch_size, train_set_percentage=0.9,
                            shuffle=True, num_workers=0, pin_memory=True, rollout_horizon=5):

    dataset = MultiStepTransitionDataset(states_list, actions_list, rollout_horizon)

    train_set, test_set = RandomSplit(dataset, train_set_percentage)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, test_loader
