from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from gym.wrappers.frame_stack import FrameStack

import os
import internet

import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt

import psutil
import time

if __name__ == "__main__":
    
    current_log = "time"

    if current_log == "time":

        level_walkthrough, time_dict = internet.main(1, 1, [0], "rgb")

        print(time_dict)
        # plot matplotlib
        values = sorted(time_dict.items())
        x, y = zip(*values)

        plt.title(" Time to reach certain distances.")
        plt.xlabel("Distance (x position)")
        plt.ylabel("Time taken (seconds)")
        plt.plot(x,y)
        plt.show()
        plt.savefig("time_to_reach_certain_distances.png")

    '''if current_log == "steps":
        steps_dict = {}
        n_episodes = 100
        step_count = 0
        for episode in range(n_episodes):
            current_state = env.reset()
            while True:
                    action = mario.act(current_state)

                    state, reward, terminated, truncated, info = env.step(action)
                    step_count += 1

                    if(info["x_pos"] >= distance_index):
                        steps_dict[distance_index] = step_count
                        distance_index += 200
                    
                    if terminated or info["flag_get"]:
                        break
                    
                    current_state = state
        # plot matplotlib
        values = sorted(steps_dict.items())
        x, y = zip(*values)

        plt.title(" Numbers of moves taken to reach certain distances.")
        plt.xlabel("Distance (x position)")
        plt.yabel("Number of moves taken (step)")
        plt.plot(x,y)
        plt.show()
        plt.savefig("steps_to_reach_certain_distances.png")

    if current_log == "deaths":
        deaths_dict = {}
        n_episodes = 100
        death_count = 0
        for episode in range(n_episodes):
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
        values = sorted(deaths_dict.items())
        x, y = zip(*values)

        plt.title(" Deaths required to reach certain distances.")
        plt.xlabel("Distance (x position)")
        plt.yabel("Times died (deaths)")
        plt.plot(x,y)
        plt.show()
        plt.savefig("deaths_to_reach_certain_distances.png")

    if current_log == "score":
        # reset env
        score_dict = {}
        n_episodes = 100
        for episode in range(n_episodes):
            current_state = env.reset()
            while True:
                    action = mario.act(current_state)

                    state, reward, terminated, truncated, info = env.step(action)

                    if(info["x_pos"] >= distance_index):
                        score_dict[distance_index] = info["score"]
                        distance_index += 200
                    
                    if terminated or info["flag_get"] or info["life"] < 3:
                        break
                    
                    current_state = state
        # plot matplotlib
        values = sorted(score_dict.items())
        x, y = zip(*values)

    

        plt.title("Mean score over distance")
        plt.xlabel("Distance (x position) ")
        plt.yabel("Times died (deaths) ")
        plt.plot(x,y)
        plt.show()
        plt.savefig("mean_score_at_certain_distances.png")


    if current_log == "hardware":
        time_taken = 0
        start_time = time.time()

        # Setup Agent
        print(f"Device: {mario.device}...")
        while time.time() < start_time + 15(60):
            current_state = env.reset() 
            while True:
                    action = mario.act(current_state)

                    state, reward, terminated, truncated, info = env.step(action)

                    if info["flag_get"]:
                        time_taken = time.time() - start_time
                        break

                    if terminated:
                        break
                    
                    current_state = state
        # plot matplotlib
        values = sorted(time_dict.items())
        x, y = zip(*values)

        plt.title(" Time to reach certain distances.")
        plt.xlabel("Distance (x position)")
        plt.yabel("Time taken (seconds)")
        plt.plot(x,y)
        plt.show()
        plt.savefig("time_to_reach_certain_distances.png")
    
    if current_log == "memory":
        time_taken = 0
        start_time = time.time()

        net_memory = 0
        mm_count = 0

        # Setup Agent
        print(f"Device: {mario.device}...")
        while time.time() < start_time + 15(60):
            current_state = env.reset() 
            while True:
                    action = mario.act(current_state)

                    state, reward, terminated, truncated, info = env.step(action)

                    if info["flag_get"]:
                        net_memory = net_memory / mm_count
                        break

                    if terminated:
                        break
                    
                    current_state = state

                    net_memory += psutil.virtual_memory().percent
                    mm_count += 1
        # plot matplotlib
        values = sorted(time_dict.items())
        x, y = zip(*values)

        plt.title(" Time to reach certain distances.")
        plt.xlabel("Distance (x position)")
        plt.yabel("Time taken (seconds)")
        plt.plot(x,y)
        plt.show()
        plt.savefig("time_to_reach_certain_distances.png")

    env.close()'''



