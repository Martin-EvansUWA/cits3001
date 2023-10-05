from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from gym.wrappers.frame_stack import FrameStack

import os

import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt

from preprocess import SkipFrame, GrayScaleObservation, ResizeObservation
from agent import MarioAgent
from logger import MarioLogger

if __name__ == "__main__":
    env_name = "SuperMarioBros-v0"
    env = gym.make(env_name, apply_api_compatibility=True, render_mode="rgb")
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
    mario = MarioAgent(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir="./network_log", scratch_dir="./tmp")
    print(f"Device: {mario.device}...")
    num_episodes = 40000 if mario.device == "cuda" else 50
    mario_logger = MarioLogger()

    mario.save()
    train = True
    # amount of simulations to improve the q-network\
    if train == True:
        for episode in range(num_episodes):
            if episode % 10 == 0:
                mario_logger.save_logger()
            mario_logger.init_episode()
            state = env.reset()
            while True:
                action = mario.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)

                mario.cache(state, next_state, action, reward, done)
                loss = mario.learn()
                state = next_state

                mario_logger.log_step(reward,loss)
                if terminated or info["flag_get"]:
                    break
            mario_logger.log_episode(episode)
            print(f"Episode: {episode}, Step: {mario.curr_step}, Eps: {mario.exploration_rate}, Avg_Loss: {mario_logger.avg_loss}")\
            
    else:
        for i in range(5):
            current_state = env.reset()

            checkpoint = torch.load("./checkpoints/2023-09-28T14-32-30/mario_net_15.chkpt")
            mario.policy_net.load_state_dict(checkpoint["model"])
            mario.policy_net.eval()
            while True:
                    
                    action = mario.act(current_state, eval=True)


                    state, reward, terminated, truncated, info = env.step(action)

                    if terminated or info["flag_get"]:
                        break
                    current_state = state
                    print(state)
                    
        
    env.close()
    mario.save()