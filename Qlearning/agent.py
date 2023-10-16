import torch
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
import numpy as np

from network import MarioNetwork
from logger import MarioLogger

import copy

class MarioAgent:
    def __init__(self, state_dim, action_dim, save_dir, scratch_dir):

        # Env params
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Save params
        self.save_dir = save_dir
        self.scratch_dir = scratch_dir



        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.policy_net = MarioNetwork(state_dim, action_dim).to(self.device)

        # Discount function
        self.gamma = 0.99   

        # Exploration vs Exploitation constants
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975    
        self.exploration_rate_min = 0.1


        # Experience replay buffers
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(10000, device=torch.device("cpu"),scratch_dir=self.scratch_dir))
        self.batch_size = 32

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.curr_step = 0
        self.save_distance = 40000

        self.first_learn = 10000
        self.learn_every = 3 

        self.sync_value = 1000

        self.TAU = 0.005

    def act(self, state):
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_dim)
        #EXPLOIT
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.policy_net(state,"online")
            action = torch.argmax(action_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min,self.exploration_rate)
        self.curr_step += 1
        return action

    # Save Memory to path
    def save(self):
        save_path = (
            self.save_dir +  f"/mario_v{int(self.curr_step // self.save_distance)}.chkpt"
        )
        # save model, along with current training process
        torch.save(
            dict(model=self.policy_net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"Mario  Network saved to {save_path} at step {self.curr_step}")


    #  
    def save_experience(self, state, next_state, action, reward, done):

        # Save the current experience information to buffer for future learning
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()


        state = torch.tensor(state).to(self.device)
        next_state = torch.tensor(next_state).to(self.device)
        action = torch.tensor([action]).to(self.device)
        reward = torch.tensor([reward]).to(self.device)
        done = torch.tensor([done]).to(self.device)

        # self.memory.append((state, next_state, action, reward, done,))
        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))

    # Returns an experience from memory
    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


    def learn(self):
        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Save network to disk  
        if self.curr_step % self.save_distance == 0:
            self.save()


        # Update target net with current policy net
        if self.curr_step % self.sync_value == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


        # Ensure we have sufficient experiences for learning
        if self.curr_step < self.first_learn:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Update the table

        # Q_current
        current_Q = self.policy_net(state)[
            np.arange(0, self.batch_size), action
        ]  
        
        # Q_target
        with torch.no_grad():
            next_state_Q = self.policy_net(next_state)
            best_action = torch.argmax(next_state_Q, axis=1)
            next_Q = self.target_net(next_state)[
                np.arange(0, self.batch_size), best_action
            ]
        
        next_Q_values = (reward + (1 - done.float()) * self.gamma * next_Q).float()

        loss = self.loss_fn(current_Q, next_Q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, next_Q_values.mean().item()
    

    def train(self,n_episodes: int, logger: MarioLogger, env):
        done = False
        for episode in range(n_episodes):
            if episode % 10 == 0:
                logger.save_logger()
            logger.init_episode()
            state = env.reset()
            while True:
                action = self.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)

                self.save_experience(state, next_state, action, reward, done)
                loss, q = self.learn()
                state = next_state

                logger.log_step(reward,loss, q)
                if terminated or truncated:
                    break
            logger.log_episode(episode)
            print(f"Episode: {episode}, Step: {self.curr_step}, Rewards: {logger.current_reward}, Avg_q: {logger.avg_q}, Avg_loss: {logger.avg_loss}")
    
    
  

