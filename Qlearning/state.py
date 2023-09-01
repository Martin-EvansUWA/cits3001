
import numpy as np
import torch

from torchvision import transforms
#convert observation to current state
def info_to_state(info):
    state = []

    # convert to list
    state.append(info['world'])
    state.append(info['stage'])
    state.append(info['y_pos'])
    state.append(info['x_pos'])
    
    return state

# convert state to q_table entry
def q_state(q_table, state):
    return q_table[state[0]][state[1]][state[2]][state[3]]


#convert obs to state
def obs_to_state(obs: np.ndarray):
    transform = transforms.Grayscale()
    return transform(torch.tensor(np.transpose(obs, (2, 0, 1)).copy(), dtype=torch.float))
