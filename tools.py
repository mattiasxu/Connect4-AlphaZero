import numpy as np
import torch

def to_string(game):
    board = game_to_board(game)
    return np.array2string(board)

def game_to_board(game):
    id = game.player
    player_board = bb_to_array(game.bitboard[id ^ 1])
    opp_board = bb_to_array(game.bitboard[id])
    board = player_board + opp_board * 2
    return board

def bb_to_array(bb):
    s = bin(bb)[2:].zfill(49)
    array = np.zeros((6, 7))
    indices = np.flip(np.arange(49).reshape((7,7)).T, 0)
    for i in range(1,7):
        for j in range(7):
            array[i-1,j] = s[-(indices[i,j] + 1)]

    return array

class AZLoss(torch.nn.modules.loss._Loss):

    def __init__(self, reduction: str = 'mean') -> None:
        super(AZLoss, self).__init__()

    def forward(self, probs, value, target_probs, target_value):
        mse = torch.nn.functional.mse_loss(value, target_value)
        ce = torch.nn.functional.binary_cross_entropy(probs, target_probs)
        return mse + ce

class C4Dataset(torch.utils.data.Dataset):
    def __init__(self, states, probs, values):
        encoded = []
        for state in states:
            own = np.zeros((6,7))
            opponent = np.zeros((6,7))
            own[state == 1] = 1
            opponent[state == 2] = 1
            encoded.append(np.array([own, opponent]))

        states = torch.as_tensor(encoded, dtype=torch.float32)
        probs = torch.as_tensor(probs, dtype=torch.float32)
        values = torch.as_tensor(values, dtype=torch.float32)

        flipped_states = torch.flip(states, [-1])
        flipped_probs = torch.flip(probs, [-1])

        self.states = torch.cat((states, flipped_states))
        self.probs = torch.cat((probs, flipped_probs))
        self.values = torch.cat((values, values))

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        sample = {'state': self.states[idx], 'probs': self.probs[idx], 'value': self.values[idx]}
        return sample

    def save_data(self, path="./datasets/dataset"):
        torch.save(self, f"{path}{len(self)}kuk.pt")