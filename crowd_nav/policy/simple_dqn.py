import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL
from crowd_sim.envs.utils.action import ActionRot, ActionXY

class QNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                 cell_size, cell_num):
        super().__init__()
        action_size = 5*16 + 1
        dropout = 0.2
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]

        self.hidden1 = nn.Linear(in_features=self.self_state_dim, out_features=64)
        self.hidden2 = nn.Linear(in_features=64, out_features=64)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(in_features=64, out_features=action_size)

    def forward(self, state):
        pass
