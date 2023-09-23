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

arr = [2, 2, 2, 3, 1, 1, 0, 5, 3, 1, 5, 0, 3, 1, 3, 0, 5, 4, 1, 2, 5, 5, 3, 3, 4, 5, 2, 3, 4, 1, 2, 6, 2, 6, 3, 3, 4, 5, 4, 5, 4, 5, 3, 5, 1, 4, 2, 5, 6, 5, 3, 6, 0, 3, 2, 4, 4, 6, 4, 6, 1, 5, 6, 6, 0, 6, 6, 5, 5, 6, 6, 2, 1, 4, 4, 2, 6, 0, 0, 0, 5, 1, 6, 1, 6, 1, 6, 2, 5, 1, 6, 5, 6, 6, 2, 3, 4, 5, 2, 0, 6, 1, 6, 4, 4, 6, 0, 1, 4, 6, 2, 0, 0, 6, 6, 1, 1, 6, 3, 0, 3, 6, 1, 6, 5, 0, 0, 1, 4, 6, 5, 2, 1, 6, 4, 5, 6, 0, 3, 2, 3, 2, 0, 6, 3, 0, 6, 4, 1, 6, 5, 1, 6, 4, 4, 6, 2, 6, 1, 0, 3, 0, 4, 2, 5, 2, 5, 3, 1, 3, 1, 4, 1, 3, 0, 2, 0, 4, 1, 2, 1, 3, 0, 0, 4, 5, 5, 1, 3, 3, 3, 1, 3, 2, 1, 2, 0, 3, 5, 4, 2, 4, 1, 5, 5, 1, 5, 0, 4, 5, 3, 5, 5, 4, 1, 0, 4, 4, 0, 4, 4, 2, 3, 3, 2, 2, 4, 5, 4, 3, 4, 0, 0, 4, 6, 2, 1, 1, 2, 4, 0, 2, 2, 1, 4, 0, 2, 0, 2, 5, 3, 5, 0, 2, 0, 6, 3, 2, 4, 6, 3, 4, 0, 2, 0, 5, 0, 3, 3, 6, 2, 0, 2, 2, 5, 2, 3, 1, 5, 4, 4, 5, 5, 1, 1, 2, 2, 1, 3, 5, 5, 4, 2, 5, 6, 2, 4, 2, 0, 5, 0, 2, 2, 5, 4, 0, 1, 1, 4, 1, 4, 4, 4, 0, 6, 2, 4, 1, 5, 6, 1, 6, 5, 2, 5, 4, 4, 5, 4, 3, 1, 3, 1, 0, 2, 5, 5, 3, 5, 0, 1, 3, 2, 0, 1, 1, 4, 4, 3, 1, 1, 1, 0, 0, 5, 3, 4, 5, 3, 4, 2, 0, 4, 1, 5, 5, 3, 6, 1, 3, 2, 0, 1, 5, 5, 1, 5, 4, 1, 2, 0, 3, 6, 2, 0, 0, 1, 1, 1, 4, 4, 2, 6, 1, 3, 2, 4, 3, 4, 4, 3, 4, 0, 1, 3, 0, 5, 4, 3, 2, 2, 6]
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

