from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

import numpy as np

import random
from state import info_to_state

def get_learning_action(q_table, state, epsilon, env):
    random_int = np.random.uniform(0,1)
    if random_int > epsilon:
        action = np.argmax(q_table[state[0]][state[1]][state[2]][state[3]])
    else:
        x = random.choices(SIMPLE_MOVEMENT, (0,10,20,10,20,20,5))
        action = SIMPLE_MOVEMENT.index(x[0])
    return action

def q_state(q_table, state):
    return q_table[state[0]][state[1]][state[2]][state[3]]

# train the q table
def train_table(n_episodes, min_epsilon, max_epsilon, decay_rate, gamma, learning_rate, env: JoypadSpace, max_steps, q_table):
    # amount of episodes to begin
    print(q_table)

    for episode in range(n_episodes):
        #initial state
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        env.reset()
        state = [1,1,79,40]
        done = False
        action = get_learning_action(q_table, state, epsilon, env)
        for step in range(max_steps):
            if step % 6 == 0:
                action = get_learning_action(q_table, state, epsilon, env)
            obs, reward, terminated, truncated, info = env.step(action)
            new_state = info_to_state(info)

            q_table[state[0]][state[1]][state[2]][state[3]][action] =  q_table[state[0]][state[1]][state[2]][state[3]][action] + learning_rate * (reward + gamma * np.max(q_table[new_state[0]][new_state[1]][new_state[2]][new_state[3]]) - q_table[state[0]][state[1]][state[2]][state[3]][action])


            done = terminated or truncated
            if done:
                env.reset()
                break

            state = new_state
    return q_table