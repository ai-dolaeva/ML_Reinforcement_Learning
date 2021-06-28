import random
import numpy as np
from collections import namedtuple
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

Sample = namedtuple('Sample',
                    ('state', 'action', 'reward',
                     'next_state', 'non_terminal'))


def create_agent(state_size, num_actions, device, fc1_dim=64, fc2_dim=64):
    def network_builder():
        net = nn.Sequential(
            nn.Linear(state_size, fc1_dim),
            nn.Sigmoid(),
            # nn.ReLU(),
            # nn.LeakyReLU(0.1),

            # nn.Linear(fc1_dim, fc2_dim), # дополнительный слой
            # nn.ReLU(),

            nn.Linear(fc2_dim, num_actions)
        )
        return net

    lr = 1e-03
    momentum = 0.9
    rho = 0.95

    optimizer_builder = lambda parameters: optim.SGD(parameters,
                                                     lr=lr, momentum=momentum)
    # optimizer_builder = lambda parameters: optim.Adam(parameters, lr=lr)
    # optimizer_builder = lambda parameters: optim.Adadelta(parameters, lr=lr,
    #                                                       rho=rho)
    gamma = 0.99
    memory_size = 100000
    batch_size = 512
    update_frequency = 100
    agent = DQNAgent(network_builder, optimizer_builder,
                     device, num_actions,
                     gamma=gamma,
                     memory_size=memory_size,
                     batch_size=batch_size,
                     update_frequency=update_frequency
                     )
    return agent


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        # Saves a training sample
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Sample(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, min(len(self.memory), batch_size))

    def __len__(self):
        return len(self.memory)


class DQNAgent(object):

    def __init__(self,
                 network_builder,
                 optimizer_builder,
                 device,
                 num_actions,
                 memory_size,
                 batch_size,
                 gamma,
                 update_frequency
                 ):

        self.device = device
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 10000
        self.update_frequency = update_frequency
        self.Q_net = network_builder().to(device)
        self.Q_net_target = network_builder().to(device)
        self.Q_net_target.load_state_dict(self.Q_net.state_dict())
        self.Q_net_target.eval()
        self.optimizer = optimizer_builder(self.Q_net.parameters())
        self.memory = ReplayMemory(memory_size)
        self.steps_done = 0
        self.eps_threshold = self.eps_start

    def save_weights(self, path):
        torch.save(self.Q_net.state_dict(), path)

    def load_weights(self, path):
        state_dict = torch.load(path)
        self.Q_net.load_state_dict(state_dict)
        self.Q_net_target.load_state_dict(state_dict)

    def act(self, state):
        self.Q_net.eval()
        #
        # self.eps_decay = 10000
        # self.eps_start = 0.9
        # self.eps_end = 0.05
        # self.eps_threshold = self.eps_start - (
        #         self.eps_start - self.eps_end) * min(
        #     1, self.steps_done / self.eps_decay)

        self.eps_decay = 5e-4
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_threshold = self.eps_end + (
                self.eps_start - self.eps_end) * np.exp(
            -self.eps_decay * self.steps_done)

        # if self.eps_threshold > self.eps_end:
        #     self.eps_threshold *= (1 - self.eps_decay)


        self.steps_done += 1

        if random.random() > self.eps_threshold:
            with torch.no_grad():
                state = torch.from_numpy(state).unsqueeze(0).type(
                    torch.get_default_dtype()).to(self.device)
                action_scores = self.Q_net(state)
                action = action_scores.max(1)[1]
                return np.asscalar(action.cpu().numpy())
        else:
            return random.randrange(self.num_actions)

    def learn(self):
        self.Q_net.train()

        if len(self.memory) < max(self.batch_size, 1):
            return

        samples = self.memory.sample(self.batch_size)
        batch = Sample(*zip(*samples))
        state_batch = torch.stack(batch.state).type(
            torch.get_default_dtype()).to(self.device)
        action_batch = torch.stack(batch.action).to(self.device)
        reward_batch = torch.stack(batch.reward).to(self.device)
        next_state_batch = torch.stack(batch.next_state).type(
            torch.get_default_dtype()).to(self.device)
        non_terminal_mask = torch.stack(batch.non_terminal).to(self.device)

        all_Q_values = self.Q_net(state_batch)
        Q_values = all_Q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        # не вычисляются значения градиента для Q_net_target
        with torch.no_grad():
            next_Q_values = self.Q_net_target(next_state_batch).max(1)[0]
            expected_Q_values = (next_Q_values * non_terminal_mask *
                                 self.gamma) + reward_batch

        loss = F.mse_loss(Q_values, expected_Q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.update_frequency == 0:
            self.Q_net_target.load_state_dict(self.Q_net.state_dict())

    def observe_and_learn(self, state, action, reward, next_state, terminal):

        self.memory.push(
            torch.tensor(state),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.get_default_dtype()),
            torch.tensor(next_state),
            torch.tensor(0 if terminal else 1,
                         dtype=torch.get_default_dtype())
        )
        self.learn()