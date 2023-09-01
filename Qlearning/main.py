from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

import numpy as np

from state import obs_to_state
from train import train_table

from constants import min_epsilon, max_epsilon, decay_rate, gamma, learning_rate, max_steps


if __name__ == "__main__":
    env_name = "SuperMarioBros-v0"
    env = gym.make(env_name, apply_api_compatibility=True, render_mode="human")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    done = True
    env.reset()


    # amount of simulations to improve the q-table
        # run a simulation
    q_table = np.zeros(10,10,10000,10000)
    train_table(q_table, 0.05, 1, 0.0005, 0.9 )
    for step in range(1):
        if step % 5 == 0:
            action = 1
        else:
            action = 5
        print(f"Action Space: {env.action_space.sample()}")
        print(f"Current action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(info)
        if done:
            state = env.reset()
            # update q-table with Bellman's Algorithm
    env.close()