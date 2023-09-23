import numpy as np
from torch import Tensor

import matplotlib
import matplotlib.pyplot as plt


class MarioLogger:

    def __init__(self):
        self.ep_rewards = []
        self.ep_lengths = []
        self.avg_ep_loss = []
        self.episodes = []

        self.current_reward = 0
        self.current_loss = 0
        self.current_length = 0

    def init_episode(self):
        self.current_reward = 0
        self.current_loss = []
        self.current_length = 0
    
    def log_episode(self, ep):
        self.ep_rewards.append(self.current_reward)
        self.ep_lengths.append(self.current_length)
        self.avg_ep_loss.append(sum(self.current_loss) / len(self.current_loss))
        self.episodes.append(ep)

    def log_step(self, reward, loss):
        self.current_reward += reward
        self.current_loss.append(loss.item())
        self.current_length += 1

    def save_logger(self):
        print(f"Total Rewards Per Ep: {self.ep_rewards}")
        plt.plot( np.array(self.episodes), np.array(self.ep_rewards))
        plt.savefig("avg_rewards.png")

        plt.cla()
        plt.plot( np.array(self.episodes), np.array(self.ep_lengths))
        plt.savefig("avg_lengths.png")

        plt.cla()
        plt.plot( np.array(self.episodes), np.array(self.avg_ep_loss))
        plt.savefig("avg_loss.png")
