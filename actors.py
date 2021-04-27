import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from neural_networks import FCModel, ConvModel, DummyModel
from mcts import MCTS

class Actor():
    # To make all actors compatible
    def opponent_act(self, action):
        return
    def reset(self):
        return

class AZActor(Actor):
    def __init__(self, env, nnet):
        self.env = env
        self.nnet = nnet
        self.mcts = MCTS(self.nnet)

    def reset(self):
        self.mcts = MCTS(self.nnet)

    def act(self, obs):
        if obs['action_mask'][-1] == 1:
            return 7
        probs = self.mcts.get_probs(self.env.game, 0)
        return np.argmax(probs)

class RandomActor(Actor):
    """Random actor"""
    def act(self, obs):
        actions = np.argwhere(obs['action_mask']).reshape(-1)
        return np.random.choice(actions)

class GreedyRandomActor(Actor):
    def __init__(self, env):
        self.env = env

    def find_winning_moves(self):
        winning_moves = []
        for move in self.env.game.get_moves():
            game = self.env.game.clone()
            game.move(move)
            if game.is_game_over():
                winning_moves.append(move)
        return winning_moves

    def find_saving_moves(self):
        saving_moves = []
        original_game = self.env.game.clone()
        for move in self.env.game.get_moves():
            game = original_game.clone()
            game.player ^= 1
            game.move(move)
            if game.is_game_over():
                saving_moves.append(move)
        return saving_moves

    def act(self, obs):
        if obs['action_mask'][-1] == 1:
            return 7

        winning_moves = self.find_winning_moves()
        if winning_moves:
            return np.random.choice(winning_moves)
        saving_moves = self.find_saving_moves()
        if saving_moves:
            return np.random.choice(saving_moves)

        actions = np.argwhere(obs['action_mask']).reshape(-1)
        return np.random.choice(actions)
        


class ManualActor(Actor):
    """Manual Actor, leave render off in workflow"""
    def __init__(self, env):
        self.env = env

    def act(self, obs):
        actions = np.argwhere(obs['action_mask']).reshape(-1)
        if actions[0] == 7:
            return 7
        else:
            self.env.render()
            action = int(input("Choose column... ")) - 1
        return action

class FCActor(Actor):
    def __init__(self):
        self.model = FCModel()

    def act(self, obs):
        return self.model.act(obs)
    
    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

class ConvActor(Actor):
    def __init__(self):
        self.model = ConvModel()

    def act(self, obs):
        return self.model.act(obs)
    
    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

if __name__ == "__main__":


    testboard = torch.rand(6,7,3)
    model = ConvActor().model
    out = model(testboard)
    print(out)
