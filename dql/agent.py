"""
Implementation follows heavily that of presented in the Udacity Nanodegree
'Deep Reinforcement Learning' (see: https://eu.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)
"""

import numpy as np
import random
from collections import namedtuple, deque

from dql.model import DQN

import torch
import torch.nn.functional as F
import torch.optim as optim

# hyperparameters
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

class Agent:
    """
    Agent class utilizing Deep Q-Learning to interact with the environment.
    """

    def __init__(self, state_size, action_size, seed, use_gpu=False):
        self.device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        # Q-Network
        self.dqn_local = DQN(state_size, action_size, seed).to(self.device)
        self.dqn_target = DQN(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.dqn_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, self.device)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """
        Do one step in environment and learn if time for update.
        :param state: current state
        :param action: taken action
        :param reward: gained immediate reward
        :param next_state: next state after doing action
        :param done: is the episode complete
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.0):
        """
        Returns actions for given state as per current policy.
        :param state: a state array
        :param eps: epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.dqn_local.eval()
        with torch.no_grad():
            action_values = self.dqn_local(state)
        self.dqn_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.int32(np.argmax(action_values.cpu().data.numpy()))
        else:
            return np.int32(random.choice(np.arange(self.action_size)))

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.
        :param experiences: type (Tuple[torch.Variable]). Tuple of (s, a, r, s', done) tuples
        :param gamma: discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.dqn_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.dqn_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.dqn_local, self.dqn_target, TAU)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            :param local_model: (PyTorch model). weights will be copied from
            :param target_model: (PyTorch model)-weights will be copied to
            :param tau: interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """ Buffer that holds the experience tuples. """

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """
        Initialize the replay buffer.
        :param action_size: dimension of the actions
        :param buffer_size: buffer max size
        :param batch_size: size of batch used in training
        :param seed: seed for RNG
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack(
            [e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack(
            [e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
