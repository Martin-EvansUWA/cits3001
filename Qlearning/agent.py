import torch
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
import numpy as np

from network import MarioNetwork

class MarioAgent:
    def __init__(self, state_dim, action_dim, save_dir, scratch_dir):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.scratch_dir = scratch_dir

        self.target_net = MarioNetwork(state_dim, action_dim)
        self.policy_net = MarioNetwork(state_dim, action_dim)

        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"

        self.gamma = 0.9

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1

        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(10000, device=torch.device("cpu"),scratch_dir=self.scratch_dir))
        self.batch_size = 32

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.curr_step = 0
        self.save_distance = 40000

        self.TAU = 0.005

    def act(self, state):
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_dim)
        #EXPLOIT
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.target_net(state)
            action = torch.argmax(action_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action

    # Save Memory to path
    def save(self):
        save_path = (
            self.save_dir +  f"/mario_net_{int(self.curr_step // self.save_distance)}.chkpt"
        )
        torch.save(
            dict(model=self.target_net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"Mario  Network saved to {save_path} at step {self.curr_step}")

    def cache(self, state, next_state, action, reward, done):
        # Save the current experience information to buffer for future learning
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()


        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        # self.memory.append((state, next_state, action, reward, done,))
        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))

    # Returns an experience from memory
    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()



    # Update the current q_network, via loss derived from bellman's
    def q_update(self,state, next_state, action, reward, done):
        # get the loss value and propagate the network
        # current Q 
        current_q = self.policy_net(state)[
            np.arange(0, self.batch_size), action
        ]
        td_estimate = current_q

        # Target Q, e.g. 
        next_state_Q = self.policy_net(next_state)
        best_action = torch.argmax(next_state_Q, axis=1)
        target_q = self.target_net(next_state)[
            np.arange(0, self.batch_size), best_action
        ]
        td_target = ((reward + (1 - done.float()) * self.gamma * target_q)).float()


        # Update q_network
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def learn(self):
        # Sample from memory
        state, next_state, action, reward, done = self.recall()


        # Save network to disk
        if self.curr_step % self.save_distance == 0:
            self.save()

        # Update the table
        self.q_update(state,next_state,action,reward,done)

        # Update target policies
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

