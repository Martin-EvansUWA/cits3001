from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

import numpy as np

from state import info_to_state
from train import train_table

import torch

from constants import min_epsilon, max_epsilon, decay_rate, gamma, learning_rate, max_steps

import random


if __name__ == "__main__":
    env_name = "SuperMarioBros-v0"
    random.seed(0)
    env = gym.make(env_name, apply_api_compatibility=True, render_mode="human")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    done = True
    env.reset()


    # amount of simulations to improve the q-table
        # run a simulation
    q_table = np.zeros((5,5,10000,10000, env.action_space.n))
    train_table(1000, min_epsilon, max_epsilon, decay_rate, gamma, learning_rate, env, max_steps, q_table )
    state = [1,1,79,40]

    for step in range(5000):
        if step % 5 == 0:
            action = np.argmax(q_table[state[0]][state[1]][state[2]][state[3]])
        print(f"Action Space: {env.action_space.sample()}")
        print(f"Current action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        
        state = info_to_state(info)
        done = terminated or truncated
        print(info)
        if done:
            state = env.reset()
            # update q-table with Bellman's Algorithm
    env.close()