import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.relu = nn.ReLU()
        self.convblock = nn.Sequential(nn.Conv2d(2, 128, 3, padding=1),
                                       nn.BatchNorm2d(128))

        self.resblocks = []
        self.resblocks2 = torch.nn.ModuleList()
        for _ in range(5):
            self.resblocks2.append(nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128))
            )
        self.policy_head = nn.Sequential(nn.Conv2d(128, 2, 1),
                                         nn.BatchNorm2d(2),
                                         nn.ReLU(),
                                         nn.Flatten(),
                                         nn.Dropout(0.3),
                                         nn.Linear(6*7*2, 7),
                                         nn.Softmax(dim=-1))
        self.value_head = nn.Sequential(nn.Conv2d(128, 1, 1),
                                        nn.BatchNorm2d(1),
                                        nn.ReLU(),
                                        nn.Flatten(),
                                        nn.Linear(6*7, 128),
                                        nn.ReLU(),
                                        nn.Dropout(0.3),
                                        nn.Linear(128, 1),
                                        nn.Tanh())
    def predict(self, board):
        own = np.zeros((6,7))
        opponent = np.zeros((6,7))
        own[board == 1] = 0
        opponent[board == 2] = 1
        board = np.array([own, opponent])
        x = torch.as_tensor(board, dtype=torch.float32)
        x = torch.unsqueeze(x, 0)
        p, v = self.forward(x)
        return p[0].detach().numpy(), v[0].detach().numpy()

    def forward(self, x):
        x = self.convblock(x)
        res_input = self.relu(x)
        for i in range(len(self.resblocks2)):
            x = self.resblocks2[i](res_input)
            x = x + res_input
            res_input = self.relu(x)
        policy = self.policy_head(res_input)
        value = self.value_head(res_input)

        return policy, value

class FCModel(nn.Module):
    """Fully Connected Model"""
    def __init__(self):
        super(FCModel, self).__init__()
        self.layer1 = nn.Linear(6*7*3, 128)
        self.layer2 = nn.Linear(128, 7) 
        self.flatten = nn.Flatten()
        self.tryhard = False

    def forward(self, x, actions):
        actions = actions[:,:-1]
        x = self.flatten(x)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        actions = torch.where(actions > 0.5, 0, -10000) # Make illegal actions super unlikely
        x = x + actions
        x = F.softmax(x, dim=-1)
        return x

    def get_policy(self, obs, actions):
        probs = self.forward(obs, actions)
        return Categorical(probs)

    def act(self, obs):
        """ Samples act from distribution of actions """
        actions = np.expand_dims(obs['action_mask'], axis=0)
        actions = torch.as_tensor(actions)

        if (actions[:,-1] == 1).any(): # Pass
            return 7

        obs = np.expand_dims(onehot(obs['board']), axis=0)
        obs = torch.as_tensor(obs, dtype=torch.float32)
        if self.training:
            return self.get_policy(obs, actions).sample().item()
        else:
            return int(torch.argmax(self.forward(obs, actions)))

    def get_action_probs(self, obs):
        actions = np.expand_dims(obs['action_mask'], axis=0)
        actions = torch.as_tensor(actions)
        obs = np.expand_dims(onehot(obs['board']), axis=0)
        obs = torch.as_tensor(obs, dtype=torch.float32)
        return self.forward(obs, actions)

class ConvModel(nn.Module):
    """Fully Connected Model"""
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(128, 7)
        self.flatten = nn.Flatten()

    def forward(self, x, actions):
        x = torch.as_tensor(x, dtype=torch.float32).reshape(-1, 3, 6, 7)
        actions = actions[:,:-1]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        actions = torch.where(actions > 0.5, 0, -10000) # Make illegal actions super unlikely
        x = x + actions
        x = F.softmax(x, dim=-1)
        return x

    def get_policy(self, obs, actions):
        probs = self.forward(obs, actions)
        return Categorical(probs)

    def act(self, obs):
        """ Samples act from distribution of actions """
        actions = np.expand_dims(obs['action_mask'], axis=0)
        actions = torch.as_tensor(actions)

        if (actions[:,-1] == 1).any(): # Pass
            return 7

        obs = np.expand_dims(onehot(obs['board']), axis=0)
        if self.training:
            return self.get_policy(obs, actions).sample().item()
        else:
            return torch.argmax(self.forward(obs, actions))

    def get_action_probs(self, obs):
        actions = np.expand_dims(obs['action_mask'], axis=0)
        actions = torch.as_tensor(actions)

        obs = np.expand_dims(onehot(obs['board']), axis=0)
        obs = torch.as_tensor(obs, dtype=torch.float32)
        return self.forward(obs, actions)