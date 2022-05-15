import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.multi_human_rl import MultiHumanRL
from crowd_sim.envs.utils.action import ActionRot, ActionXY
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_action_size):
        super().__init__()
        action_size = output_action_size
        drop = 0.2
        self.state_dim = input_dim

        self.hidden1 = nn.Linear(in_features=self.state_dim, out_features=64)
        self.hidden2 = nn.Linear(in_features=64, out_features=64)
        self.dropout = nn.Dropout(drop)
        self.out = nn.Linear(in_features=64, out_features=action_size)

    def forward(self, state):
        output = self.hidden1(state)
        output = self.hidden2(output)
        output = self.dropout(output)
        output = self.out(output)
        return output


class QLearning(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'QLearning'

    def configure(self, config):
        # set up self parameters: gamma, kinematics, sampling, speed_samples,
        # rotation_samples, query_env, cell_size, om_channel_size
        self.set_common_parameters(config)
        self.model = QNetwork(self.input_dim(), self.self_state_dim, )
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')

    def build_occupancy_map(self, state):
        """
        :param joint_state:
        :return: tensor of shape (self.cell_num ** 2)
        """
        human_states = state.human_states
        self_state = state.self_state
        humans = np.concatenate([np.array([(human.px, human.py, human.vx, human.vy)])
                                 for human in human_states], axis=0)
        self_px = humans[:, 0] - self_state.px
        self_py = humans[:, 1] - self_state.py

        # new x-axis is in the direction of human's velocity
        human_velocity_angle = np.arctan2(self_state.vy, self_state.vx)
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
            return torch.from_numpy(occupancy_map.astype(int))
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
            return torch.from_numpy(dm).float()

    def predict(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to Q-value network.
        The input to the Q-value network is always of shape (batch_size, # full_state + occupancy_map)

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            max_value = float('-inf')
            max_action = None
            occupancy_map = self.build_occupancy_map(state)
            self_state = torch.tensor(state.self_state.to_list())
            self.action_values = self.model(torch.cat([self_state, occupancy_map.view(1, -1).squeeze()]))
            with torch.no_grad():
                # some actions might have the same value (?
                max_value = torch.argmax(self.action_values)
                max_action = self.action_space[max_value]

            if max_action is None:
                raise ValueError('Value network is not well trained. ')

        # if self.phase == 'train':
        #     self.last_state = self.transform(state)
        return max_action



