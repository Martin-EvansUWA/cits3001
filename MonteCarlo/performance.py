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

        level_walkthrough, time_dict, steps_dict, deaths_dict, score_dict = internet.main(1, 1, [0], "rgb")

        print("Time dict: ", time_dict)
        print("Steps dict: ", steps_dict)
        print("deaths dict:", deaths_dict)
        print("score dict:", score_dict)


        # plot time
        values = sorted(time_dict.items())
        x, y = zip(*values)

        plt.title("Time to reach certain distances.")
        plt.xlabel("Distance (x position)")
        plt.ylabel("Time taken (seconds)")
        plt.plot(x,y)
        plt.show()
        plt.savefig("time_to_reach_certain_distances.png")

        print(steps_dict)

        # plot steps
        values = sorted(steps_dict.items())
        x, y = zip(*values)

        plt.title(" Numbers of moves taken to reach certain distances.")
        plt.xlabel("Distance (x position)")
        plt.ylabel("Number of moves taken (step)")
        plt.plot(x,y)
        plt.show()
        plt.savefig("steps_to_reach_certain_distances.png")

        # plot deaths
        values = sorted(deaths_dict.items())
        x, y = zip(*values)

        plt.title(" Deaths required to reach certain distances.")
        plt.xlabel("Distance (x position)")
        plt.ylabel("Times died (deaths)")
        plt.plot(x,y)
        plt.show()
        plt.savefig("deaths_to_reach_certain_distances.png")

        # plot score
        values = sorted(score_dict.items())
        x, y = zip(*values)

        plt.title("Mean score over distance")
        plt.xlabel("Distance (x position) ")
        plt.ylabel("Score ")
        plt.plot(x,y)
        plt.show()
        plt.savefig("mean_score_at_certain_distances.png")

'''
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

    env.close()

'''