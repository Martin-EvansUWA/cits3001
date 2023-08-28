from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
print(SIMPLE_MOVEMENT)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
done = True
env.reset()
for step in range(1000):
    if step % 5 == 0:
        action = 0
    else:
        action = 5
    print(f"Action Space: {env.action_space.sample()}")
    print(f"Current action: {action}")
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"Obs: {obs}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"INFO DICTIONARY: {info}")
    if done:
        state = env.reset()
env.close()