from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
print(SIMPLE_MOVEMENT)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
done = True
env.reset()
for step in range(100):
    action = env.action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(6)
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