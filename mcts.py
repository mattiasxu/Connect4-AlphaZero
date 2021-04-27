import torch
from tools import *
import gym
import gym_connect4
from gym_connect4.envs.connect4_env import Connect4
import math
import numpy as np
from neural_networks import DummyModel


class MCTS():
    def __init__(self, nnet, rollouts=25):
        self.nnet = nnet
        self.rollouts = rollouts

        self.Ns = {}
        self.Nsa = {}
        self.Vs = {}
        self.Qsa = {}
        self.Ps = {}

    def get_probs(self, game, temp=1):
        for i in range(self.rollouts):
            self.search(game)
        s = to_string(game)

        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(7)]
        
        if temp == 0:
            best_as = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_a = np.random.choice(best_as)
            probs = [0] * 7
            probs[best_a] = 1
            return probs
        
        counts = [i ** (1. / temp) for i in counts]
        sum_count = float(sum(counts))
        probs = [i / sum_count for i in counts]
        return probs

    def search(self, node):
        current = node
        s = to_string(current)

        if current.is_game_over():
            return 1

        if s not in self.Ns:
            self.Ps[s], v = self.nnet.predict(game_to_board(current))
            self.Ps[s] = self.Ps[s] * current.get_action_mask()
            if np.sum(self.Ps[s]) > 0:
                self.Ps[s] = self.Ps[s] / np.sum(self.Ps[s]) # Renorm
            else:
                print("Something bugged")
        
            self.Vs[s] = current.get_action_mask()
            self.Ns[s] = 0
            return -v
        
        valid_moves = self.Vs[s]
        current_best = -float('inf')
        best_action = -1

        for a in range(7):
            if valid_moves[a] == 1:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + 1 * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.Ps[s][a] * math.sqrt(self.Ns[s] + 0.000001)

                if u > current_best:
                    best_action = a
                    current_best = u
        
        a = best_action
        next_game = current.clone()
        next_game.move(a)

        v = self.search(next_game)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        
        self.Ns[s] += 1
        return -v




if __name__ == "__main__":
    env = gym.make("Connect4Env-v0")
    nnet = DummyModel()
    mcts = MCTS(nnet)

    while not env.game.is_game_over():
        probs = mcts.get_probs(env.game)
        action = np.random.choice(np.arange(7), p=probs)
        print(probs, action)
        env.game.move(action)