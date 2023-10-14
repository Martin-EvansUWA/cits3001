from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from gym.wrappers.frame_stack import FrameStack

import os
import sys

import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt

from preprocess import SkipFrame, GrayScaleObservation, ResizeObservation
from agent import MarioAgent
from logger import MarioLogger

import psutil, os
import time



if __name__ == "__main__":
    env_name = "SuperMarioBros-1-1-v0"
    env = gym.make(env_name, apply_api_compatibility=True, render_mode="rgb")
    env = JoypadSpace(env, [["right"],["right","A"]])

    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)

    env.reset()
   

    current_log = sys.argv[1]
    mario = MarioAgent(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir="./network_log", scratch_dir="./tmp")
    checkpoint = torch.load("mario_net_22.chkpt", map_location=torch.device("cpu"))
    mario.policy_net.load_state_dict(checkpoint["model"])
    mario.policy_net.online.eval()
    mario.policy_net.target.eval()
    print(f"Device: {mario.device}...")
    distance_index = 200

    
    print(f"Testing: {current_log}")
    if current_log == "time":
        time_dict = {}
        start_time = time.time()

        # Setup Agent
        mario = MarioAgent(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir="./network_log", scratch_dir="./tmp")
        print(f"Device: {mario.device}...")
        while time.time() < start_time + 15*(60):
            print(f"Current Time: {(time.time() - start_time) / 60}")
            current_state = env.reset() 
            while True:
                    action = mario.act(current_state)

                    state, reward, terminated, truncated, info = env.step(action)

                    if((distance_index not in time_dict.keys()) and info["x_pos"] >= distance_index):
                        time_dict[distance_index] = time.time() - start_time
                        distance_index += 200
                    
                    if terminated or info["flag_get"]:
                        if info["flag_get"]:
                            print("COMPLETED!!!")
                        break
                    
                    current_state = state
        # plot matplotlib

        print(time_dict)
        values = sorted(time_dict.items())
        x, y = zip(*values)

        print(x)

        print()

        print(y)
        x = np.array(x)
        y = np.array(y)

        plt.scatter(x,y)
        plt.ylim(0,None)

        plt.title(" Time to reach certain distances.")
        plt.xlabel("Distance (x position)")
        plt.ylabel("Time taken (seconds)")

        plt.show()
        plt.savefig("time_to_reach_certain_distances2.png")



    if current_log == "steps":
        steps_dict = {}
        n_episodes = 1000
        step_count = 0
        for episode in range(n_episodes):
            print(f"Episode: {episode}, {steps_dict}")
            current_state = env.reset()
            while True:
                    action = mario.act(current_state)

                    state, reward, terminated, truncated, info = env.step(action)
                    step_count += 1

                    if(distance_index not in steps_dict.keys() and info["x_pos"] >= distance_index):
                        steps_dict[distance_index] = step_count
                        distance_index += 200
                    
                    if terminated or info["flag_get"]:
                        episode = n_episodes
                        break
                    
                    current_state = state
        # plot matplotlib

        print(steps_dict)
        values = sorted(steps_dict.items())
        x, y = zip(*values)

        x = np.array(x)
        y = np.array(y)

        plt.scatter(x,y)
        plt.ylim(0,None)

        plt.scatter(x, y)
        plt.title(" Numbers of moves taken to reach certain distances.")
        plt.xlabel("Distance (x position)")
        plt.ylabel("Number of moves taken (step)")
        plt.show()
        plt.savefig("steps_to_reach_certain_distances.png")

    if current_log == "deaths":
        # plot matplotlib
        deaths_dict = {}
        death_count = 0
        n_episodes = 1000
        for episode in range(n_episodes):
            if episode % 50 == 0:
                print(f"Episode: {episode}, {death_count}")
            current_state = env.reset()
            while True:
                    action = mario.act(current_state)

                    state, reward, terminated, truncated, info = env.step(action)

                    if(info["x_pos"] >= distance_index):
                        deaths_dict[distance_index] = death_count
                        distance_index += 200
                    
                    if terminated or info["flag_get"]:
                        death_count += 1
                        break
                    
                    current_state = state
        # plot matplotlib
        print(deaths_dict)
        values = sorted(deaths_dict.items())
        x, y = zip(*values)

        x = np.array(x)
        y = np.array(y)

        plt.scatter(x,y)
        plt.ylim(0,None)

        plt.title(" Deaths required to reach certain distances.")
        plt.xlabel("Distance (x position)")
        plt.ylabel("Times died (deaths)")
        
        plt.savefig("deaths_to_reach_certain_distances.png")

    if current_log == "score":
        # reset env
        score_dict = {}
        value_dict = {}
        n_episodes = 5000
        for episode in range(n_episodes):
            distance_index = 200
            if episode % 50 == 0:
                 print(f"Ep: {episode}")
            current_state = env.reset()
            while True:
                    action = mario.act(current_state)

                    state, reward, terminated, truncated, info = env.step(action)

                    if(info["x_pos"] >= distance_index):
                        value_dict[distance_index] = []
                        value_dict[distance_index].append(info["score"])
                        score_dict[distance_index] = sum(value_dict[distance_index]) / len(value_dict[distance_index])
                        distance_index += 200

                    
                    if terminated or info["flag_get"] or info["life"] < 2:
                        break
                    
                    current_state = state
        # plot matplotlib
        print(score_dict)
        values = sorted(score_dict.items())
        x, y = zip(*values)

        x = np.array(x)
        y = np.array(y)

        plt.scatter(x,y)
        plt.ylim(0,None)
    

        plt.title("Mean score over distance")
        plt.xlabel("Distance (x position) ")
        plt.ylabel("Times died (deaths) ")
        plt.show()
        plt.savefig("mean_score_at_certain_distances.png")


    if current_log == "hardware":
        times = []
        start_time = time.time()

        # Setup Agent
        print(f"Device: {mario.device}...")
        while time.time() < start_time + 120*(60):
            print(f"Current Time: {(time.time() - start_time) / 60}")
            current_state = env.reset() 
            while True:
                    action = mario.act(current_state)

                    state, reward, terminated, truncated, info = env.step(action)

                    if info["flag_get"]:
                        times.append(time.time() - start_time)
                        print(times)
                        print("Finished!")
                        start_time = time.time()
                        break

                    if terminated:
                        break
                    
                    current_state = state
        # plot matplotlib
        print(sum(times) / len(times))
    
    if current_log == "memory":
        time_taken = 0
        start_time = time.time()

        net_memory = []
        avg = 0

        # Setup Agent
        print(f"Device: {mario.device}...")
        while time.time() < start_time + 15*(60):
            if(len(net_memory) > 0):
                print(f"Current Avg: {sum(net_memory) / len(net_memory)}")
            current_state = env.reset() 
            while True:
                    action = mario.act(current_state)

                    state, reward, terminated, truncated, info = env.step(action)

                    if info["flag_get"]:
                        avg = sum(net_memory) / len(net_memory)
                        break

                    if terminated:
                        break
                    
                    current_state = state

                    net_memory.append(psutil.Process().memory_info().rss / (1024**2))
                    avg = sum(net_memory) / len(net_memory)
        # plot matplotlib
        print(f"Average: {avg}")
        values = sorted(time_dict.items())
        x, y = zip(*values)

        x = np.array(x)
        y = np.array(y)

        x = np.array(x)
        y = np.array(y)

        plt.scatter(x,y)
        plt.ylim(0,None)

        plt.title(" Time to reach certain distances.")
        plt.xlabel("Distance (x position)")
        plt.ylabel("Time taken (seconds)")
        plt.show()
        plt.savefig("time_to_reach_certain_distances.png")

    env.close()



