from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info



env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = SkipFrame(env, skip=4)
print(SIMPLE_MOVEMENT)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
done = True
env.reset()

arr = [3, 1, 3, 3, 3, 2, 2, 2, 4, 2, 1, 5, 2, 1, 4, 6, 5, 0, 2, 2, 2, 0, 2, 1, 1, 1, 5, 4, 3, 4, 4, 3, 4, 4, 2, 3, 4, 2, 6, 3, 2, 2, 4, 5, 0, 2, 2, 6, 6, 6, 1, 5, 1, 4, 2, 4, 3, 2, 4, 4, 4, 1, 5, 2, 2, 2, 5, 5, 2, 0, 2, 2, 2, 5, 1, 3, 2, 6, 2, 1, 6, 4, 5, 4, 6, 4, 3, 4, 0, 5, 0, 2, 0, 3, 4, 6, 3, 1, 2, 6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(len(arr)):
    action = env.action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(arr[i])
    #print("XPOS: ", info['x_pos'])
    #print("TIME LEFT: ", info['time'])
    #print("OBSLEN: ", len(obs)) 
    #print("OBSlen2: ", len(obs[1]))
    #print("OBSlen2: ", len(obs[1][1]))
    print("reward", reward)
    if terminated:
        print("terminated", terminated)
    if truncated:
        print("truncated", truncated)
    #print("info", info)
    done = terminated or truncated
    if done:
        print("RESETTING" , step, "\n\n\n\n\n\n\n\n\n\n\n\n\n")
        state = env.reset()
env.close()