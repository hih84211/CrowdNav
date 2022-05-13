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

    def configure(self, config):
        # set up self parameters: gamma, kinematics, sampling, speed_samples,
        # rotation_samples, query_env, cell_size, om_channel_size
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl', 'with_om')
        with_global_state = config.getboolean('sarl', 'with_global_state')
        self.model = QNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-SARL'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

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

    def transform(self, state):
        """
        Take the state passed from agent and transform it to the input of value network

        :param state:
        :return: tensor of shape (# of humans, len(state))
        """
        # self_position: (px, py), self_velocity: (vx, vy), self_radius: r,
        # goal_position: (gx, gy), theta: heading direction
        # human_state: [px, py, vx, vy, r]
        # robot_state: [px, py, vx, vy, r, gx, gy, theta, v_pref]
        state_tensor = torch.cat([torch.Tensor([state.self_state + human_state]).to(self.device)
                                  for human_state in state.human_states], dim=0)
        if self.with_om:
            occupancy_maps = self.build_occupancy_maps(state.human_states)
            state_tensor = torch.cat([self.rotate(state_tensor), occupancy_maps.to(self.device)], dim=1)
        else:
            state_tensor = self.rotate(state_tensor)
            # self.rotate(state) will transform the state from global coord to robot-centric coord
            # state_tensor for human i: [dg, v_pref, vx, vy, r,
            #                            pix, piy, vix, viy, ri, di, ri+r]
        return state_tensor

    def rotate(self, state):
        # from cadrl.py
        # can be removed
        """
        Transform the coordinate to agent-centric.
        Input state tensor is of size (batch_size, state_length)

        """
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))
        if self.kinematics == 'unicycle':
            theta = (state[:, 8] - rot).reshape((batch, -1))
        else:
            # set theta to be zero since it's not used
            theta = torch.zeros_like(v_pref)
        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                  reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
        new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
        return new_state

