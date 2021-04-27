import gym
import gym_connect4
import numpy as np
from tools import *
from mcts import MCTS
from neural_networks import DummyModel
import tournament
import actors

class Dojo():
    def __init__(self, nnet, episodes=100, rollouts=25, remember_iterations=15):
        self.nnet = nnet
        self.nnet.eval()
        self.episodes = episodes
        self.rollouts = rollouts
        self.remember_iterations = remember_iterations
    
    def play_one_game(self):
        env = gym.make("Connect4Env-v0")
        mcts = MCTS(self.nnet, self.rollouts)

        boards = []
        probs = []
        values = []

        game_len = 0
        while not env.game.is_game_over():
            if game_len < 15:
                temp = 1
            else:
                temp = 0
            P = mcts.get_probs(env.game, temp)
            action = np.random.choice(np.arange(7), p=P)
            
            boards.append(game_to_board(env.game))
            probs.append(P)

            env.game.move(action)
            game_len += 1

        values = [[env.game.get_reward(0) * (-1)**i] for i in range(len(boards))]        

        return boards, probs, values
    
    def train(self, iterations=100000000):
        data_from_iterations = []
        optimizer = torch.optim.Adam(self.nnet.parameters(), lr=0.001, weight_decay=0.0001)
        loss = AZLoss()

        cpu = torch.device("cpu")
        gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for i in range(iterations):
            print(f"ITERATION {i}")
            boards = []
            probs = []
            values = []
            for _ in range(self.episodes):
                print(".", end="")
                b, p, v = self.play_one_game()
                boards.extend(b)
                probs.extend(p)
                values.extend(v)
            data_from_iterations.append([boards, probs, values])
            
            if len(data_from_iterations) > self.remember_iterations:
                print("Removing old samples...")
                data_from_iterations.pop(0)

            boards = []
            probs = []
            values = []
            for data in data_from_iterations:
                boards.extend(data[0])
                probs.extend(data[1])
                values.extend(data[2])

            dataset = C4Dataset(boards, probs, values)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

            torch.save(self.nnet.state_dict(), f"./saved/models/temp.pt")

            self.nnet.train()
            self.nnet.to(gpu)
            print("Training")
            for epochs in range(20):
                epoch_loss = 0
                for idx, data in enumerate(dataloader):
                    state = data['state'].to(gpu)
                    target_probs = data['probs'].to(gpu)
                    target_value = data['value'].to(gpu)
                    optimizer.zero_grad()

                    pred_probs, pred_vals = self.nnet.forward(state)
                    batch_loss = loss(pred_probs, pred_vals, target_probs, target_value)
                    batch_loss.backward()
                    epoch_loss += batch_loss.item()
                    optimizer.step()
                print(f"Dataset length: {len(dataset)} Epoch {epochs}: {epoch_loss}")
            self.nnet.to(cpu)
            self.nnet.eval()

            test_env = gym.make('Connect4Env-v0')

            new_actor = actors.AZActor(test_env, self.nnet)
            oldnet = DummyModel()
            oldnet.load_state_dict(torch.load(f"./saved/models/temp.pt"))
            old_actor = actors.AZActor(test_env, oldnet)
            print("Running tournament")
            tourny = tournament.Tournament([new_actor, old_actor], 40, test_env)
            tourny.run()
            if tourny.rewards[0] >= 3:
                print(f"New model accepted: {tourny.rewards[0]}")
                torch.save(self.nnet.state_dict(), f"./saved/models/temp_best{i}.pt")
            else:
                print(f"New model rejected: {tourny.rewards[0]}")
                self.nnet.load_state_dict(torch.load(f"./saved/models/temp.pt"))

            torch.save(dataset, f"./saved/datasets/dataset_{i}.pt")
            torch.save(self.nnet.state_dict(), f"./saved/models/state_dict_{i}.pt")


if __name__ == "__main__":
    nnet = DummyModel()
    dojo = Dojo(nnet, episodes=100)
    dojo.train(100)