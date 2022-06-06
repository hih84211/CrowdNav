import torch
import torch.nn as nn
import numpy as np
import itertools
import logging
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import ObservableState, FullState


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, mlp_dims):
        super().__init__()
        layers = []
        mlp_dims = [input_dim] + mlp_dims
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i != len(mlp_dims) - 2:
                layers.append(nn.ReLU())
        self.value_network = nn.Sequential(*layers)

    def forward(self, state):
        value = self.value_network(state)
        return value


class CARL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'CARL'
        self.trainable = True
        self.epsilon = None
        self.gamma = None
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.action_space = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.self_state_dim = 6
        self.human_state_dim = 7
        self.joint_state_dim = self.self_state_dim + self.human_state_dim

    def configure(self, config):
        self.gamma = config.getfloat('rl', 'gamma')
        # self.sampling = config.get('action_space', 'sampling')
        self.speed_samples = config.getint('action_space', 'speed_samples')
        self.rotation_samples = config.getint('action_space', 'rotation_samples')
        mlp_dims = [int(x) for x in config.get('rl', 'mlp_dims').split(', ')]
        human_num = 5  # Temp hard coding...
        self.model = ValueNetwork(self.joint_state_dim * human_num, mlp_dims)

    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def build_action_space(self, v_pref):
        """
        Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
        """
        speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * v_pref for i in range(self.speed_samples)]

        rotations = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)

        action_space = [ActionXY(0, 0)]
        for rotation, speed in itertools.product(rotations, speeds):
            action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))

        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space

    def propagate(self, state, action):
        if isinstance(state, ObservableState):
            # propagate state of humans
            next_px = state.px + action.vx * self.time_step
            next_py = state.py + action.vy * self.time_step
            next_state = ObservableState(next_px, next_py, action.vx, action.vy, state.radius)
        elif isinstance(state, FullState):
            # propagate state of current agent
            # perform action without rotation
            next_px = state.px + action.vx * self.time_step
            next_py = state.py + action.vy * self.time_step
            next_state = FullState(next_px, next_py, action.vx, action.vy, state.radius,
                                   state.gx, state.gy, state.v_pref, state.theta)
        else:
            raise ValueError('Type error')

        return next_state

    def predict(self, state):
        """
        Input state is the joint state of robot concatenated by the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            self.action_values = list()
            max_min_value = float('-inf')
            max_action = None
            for action in self.action_space:
                next_self_state = self.propagate(state.self_state, action)
                ob, reward, done, info = self.env.onestep_lookahead(action)
                batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(self.device)
                                              for next_human_state in ob], dim=0)
                # Value update
                rotated_state = self.rotate(batch_next_states)
                rotated_state = torch.reshape(rotated_state, (-1,))
                outputs = self.model(rotated_state)
                min_output, min_index = torch.min(outputs, 0)
                min_value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * min_output.data.item()
                self.action_values.append(min_value)
                if min_value > max_min_value:
                    max_min_value = min_value
                    max_action = action

        if self.phase == 'train':
            state = self.transform(state)
            state = torch.reshape(state, (-1,))
            self.last_state = state

        return max_action

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

        state_tensor = self.rotate(state_tensor)
        return state_tensor

    def rotate(self, state):
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
