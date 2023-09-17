from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from gym.wrappers.frame_stack import FrameStack

import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from preprocess import SkipFrame, GrayScaleObservation, ResizeObservation
from agent import MarioAgent

if __name__ == "__main__":
    env_name = "SuperMarioBros-v0"
    env = gym.make(env_name, apply_api_compatibility=True, render_mode="human")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    done = True

    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)
    
    env.reset()

    # Setup Agent
    os.mkdir("./tmp")
    mario = MarioAgent(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir="./network_log", scratch_dir="./tmp")
    num_episodes = 1000

    mario.save()

    # amount of simulations to improve the q-network
    for episode in range(num_episodes):
        
        state = env.reset()
        print(f"Episode: {episode}, Step: {mario.curr_step}")
        while True:
            action = mario.act(state)


            next_state, reward, terminated, truncated, info = env.step(action)
            """plt.imshow(next_state)
            plt.show()"""


            mario.cache(state, next_state, action, reward, done)
            mario.learn()
            state = next_state



            if terminated or info["flag_get"]:
                break
        
    env.close()
    os.rmdir("./tmp")
