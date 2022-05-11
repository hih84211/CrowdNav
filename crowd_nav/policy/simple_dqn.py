import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL
from crowd_sim.envs.utils.action import ActionRot, ActionXY
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, cell_size, cell_num):
        super().__init__()
        action_size = 5*16 + 1
        dropout = 0.2
        self.state_dim = 8 * 13

        self.hidden1 = nn.Linear(in_features=self.state_dim, out_features=64)
        self.hidden2 = nn.Linear(in_features=64, out_features=64)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(in_features=64, out_features=action_size)
        self.cell_size = cell_size
        self.cell_num = cell_num

    def forward(self, state):
        # state: Tensor(1, 8, 13)
        state = state.view(1, -1)  # shape: (1, 104)

class QLearning(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'QLearning'

    def build_occupancy_maps(self, state):
        """

        :param human_states:
        :return: tensor of shape (# human - 1, self.cell_num ** 2)
        """
        human_states = state.human_states
        self_state = state.self_state
        occupancy_maps = []
        humans = np.concatenate([np.array([(human.px, human.py, human.vx, human.vy)])
                                 for human in human_states], axis=0)
        self_px = humans[:, 0] - human.px
        self_py = humans[:, 1] - human.py
        # new x-axis is in the direction of human's velocity
        human_velocity_angle = np.arctan2(human.vy, human.vx)
        other_human_orientation = np.arctan2(self_py, self_px)
        rotation = other_human_orientation - human_velocity_angle
        distance = np.linalg.norm([self_px, self_py], axis=0)
        self_px = np.cos(rotation) * distance
        self_py = np.sin(rotation) * distance

        # compute indices of humans in the grid
        other_x_index = np.floor(self_px / self.cell_size + self.cell_num / 2)
        other_y_index = np.floor(self_py / self.cell_size + self.cell_num / 2)
        other_x_index[other_x_index < 0] = float('-inf')
        other_x_index[other_x_index >= self.cell_num] = float('-inf')
        other_y_index[other_y_index < 0] = float('-inf')
        other_y_index[other_y_index >= self.cell_num] = float('-inf')
        grid_indices = self.cell_num * other_y_index + other_x_index
        occupancy_map = np.isin(range(self.cell_num ** 2), grid_indices)
        if self.om_channel_size == 1:
            occupancy_maps.append([occupancy_map.astype(int)])
        else:
            # calculate relative velocity for other agents
            other_human_velocity_angles = np.arctan2(humans[:, 3], humans[:, 2])
            rotation = other_human_velocity_angles - human_velocity_angle
            speed = np.linalg.norm(humans[:, 2:4], axis=1)
            other_vx = np.cos(rotation) * speed
            other_vy = np.sin(rotation) * speed
            dm = [list() for _ in range(self.cell_num ** 2 * self.om_channel_size)]
            for i, index in np.ndenumerate(grid_indices):
                if index in range(self.cell_num ** 2):
                    if self.om_channel_size == 2:
                        dm[2 * int(index)].append(other_vx[i])
                        dm[2 * int(index) + 1].append(other_vy[i])
                    elif self.om_channel_size == 3:
                        dm[3 * int(index)].append(1)
                        dm[3 * int(index) + 1].append(other_vx[i])
                        dm[3 * int(index) + 2].append(other_vy[i])
                    else:
                        raise NotImplementedError
            for i, cell in enumerate(dm):
                dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
            occupancy_maps.append([dm])

        return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()

