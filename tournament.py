import torch
import gym
import gym_connect4
import itertools
import actors
from mcts import MCTS
import neural_networks

class Tournament():
    def __init__(self, players, games, env):
        self.players = players
        self.games = games
        self.rewards = [0 for _ in range(len(players))]
        self.env = env

    def run(self):
        player_indeces = list(itertools.combinations([i for i in range(len(self.players))], 2))
        for player1, player2 in player_indeces:
            actors = [self.players[player1], self.players[player2]]

            # Start the games
            for i in range(int(self.games//2)):
                rewards = self.play(actors)
                self.rewards[player1] += rewards[0]
                self.rewards[player2] += rewards[1]

            # Swap starting player
            actors[0], actors[1] = actors[1], actors[0]

            for i in range(int(self.games//2)):
                rewards = self.play(actors)
                self.rewards[player1] += rewards[1]
                self.rewards[player2] += rewards[0]

    def play(self, actors):
        obses = self.env.reset()
        game_over = False
        for actor in actors:
            actor.reset()

        while not game_over:
            action_dict = {}
            for actor_id, actor in enumerate(actors):
                action = actor.act(obses[actor_id])
                action_dict[actor_id] = action
                actors[actor_id ^ 1].opponent_act(action)

            obses, rewards, game_over, info = self.env.step(action_dict)
        return rewards

if __name__ == "__main__":
    env = gym.make('Connect4Env-v0')
    network = neural_networks.DummyModel()
    network.load_state_dict(torch.load("./AZTEST.pt"))
    AZ = actors.AZActor(50, env, network)

    random = actors.RandomActor()
    conv = actors.ConvActor()
    conv.model.load_state_dict(torch.load("./saved_models/convmodel.pt"))
    
    contestants = [MCTS(100, env), AZ]
    test = Tournament(contestants, 100, env)
    test.run()
    print(test.rewards)
