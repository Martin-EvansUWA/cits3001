import numpy as np
from torch import Tensor

import matplotlib
import matplotlib.pyplot as plt


class MarioLogger:

    def __init__(self):

        self.current_episode = 0
        self.total_rewards = []
        self.ep_rewards = []
        self.ep_lengths = []
        self.avg_ep_loss = []
        self.avg_qs = []
        self.episodes = []

        self.current_reward = 0
        self.current_loss = 0
        self.current_length = 0

        self.avg_q = 0
        self.avg_loss = 0



    def init_episode(self):
        self.current_reward = 0
        self.current_loss = []
        self.current_qs = []
        self.current_length = 0

        self.avg_q = 0
        self.avg_loss = 0

    def log_episode(self, ep):
        self.avg_q = sum(self.current_qs) / len(self.current_qs) if len(self.current_qs) > 0 else 0
        self.avg_loss = sum(self.current_loss) / len(self.current_loss) if len(self.current_loss) > 0 else 0
        self.ep_rewards.append(self.current_reward)
        self.ep_lengths.append(self.current_length)
        self.avg_qs.append(self.avg_q)
        self.avg_ep_loss.append(self.avg_loss)
        self.episodes.append(ep)
        if(self.current_episode % 20 == 0):
            self.reward_plot()
        self.current_episode += 1
    def log_step(self, reward, loss, q):
        self.current_reward += reward
        if loss != None:
            self.current_loss.append(loss.item())
        if q != None:
            self.current_qs.append(q)
        self.current_length += 1
        
    def reward_plot(self):
        self.total_rewards.append(sum(self.ep_rewards) / len(self.ep_rewards))
        self.ep_rewards = []

    def save_logger(self):
        plt.plot( np.array(self.episodes/20), np.array(self.total_rewards))
        plt.savefig("avg_rewards.png")

        plt.cla()
        plt.plot( np.array(self.episodes), np.array(self.ep_lengths))
        plt.savefig("avg_lengths.png")

        plt.cla()
        plt.plot( np.array(self.episodes), np.array(self.avg_ep_loss))
        plt.savefig("avg_loss.png")
        
        plt.cla()
        plt.plot( np.array(self.episodes), np.array(self.avg_qs))
        plt.savefig("avg_q.png")
