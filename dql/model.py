import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """
        Initialize parameters and build model.
        :param state_size: Dimension of the states
        :param action_size: Dimension of the actions
        :param seed: seed for RNG
        :param fc1_units: First hidden layer node count
        :param fc2_units: Second hidden layer node count
        """
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """
        Forward function for the network.
        :param state: input state
        :return: Tensor output of the last hidden layer
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN2(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=512,
                 fc2_units=512, fc3_units=256, fc4_units=128, fc5_units=64):
        """
        Initialize parameters and build model.
        :param state_size: Dimension of the states
        :param action_size: Dimension of the actions
        :param seed: seed for RNG
        :param fc1_units: First hidden layer node count
        :param fc2_units: Second hidden layer node count
        """
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, fc4_units)
        self.fc5 = nn.Linear(fc4_units, fc5_units)
        self.fc6 = nn.Linear(fc5_units, action_size)

    def forward(self, state):
        """
        Forward function for the network.
        :param state: input state
        :return: Tensor output of the last hidden layer
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        return x
