from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

import numpy as np

from state import obs_to_state

def get_learning_action(q_table, state, epsilon, env):
    random_int = np.random.uniform(0,1)
    if random_int > epsilon:
        action = np.argmax(q_table[state[0]][state[1]][state[2]][state[3]])
    else:
        action = env.action_space.sample()
    return action

def q_state(q_table, state):
    return q_table[state[0]][state[1]][state[2]][state[3]]

# train the q table
def train_table(n_episodes, min_epsilon, max_epsilon, decay_rate, gamma, learning_rate, env: JoypadSpace, max_steps, q_table):
    # amount of episodes to begin
    for episode in range(len(n_episodes)):
        #initial state
        state = [1,1,79,40]
        done = False
        for step in range(max_steps ):
            action = get_learning_action(q_table, )
            obs, reward, terminated, truncated, info = env.step(action)
            new_state = obs_to_state(info)

            q_state(state)[action] =  q_state(state)[action] + learning_rate * (reward + gamma * np.max(q_state(new_state)) - q_state(state)[action])


            done = terminated or truncated
            if done:
                break

            state = new_state
    return q_table