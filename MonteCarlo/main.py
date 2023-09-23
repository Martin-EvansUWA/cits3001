import internet

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import math
import random

level_walkthroughs = [[] * 32]



for world in range (1,9):
    for stage in range (1,5):
        level_index = world * stage - 1
        level_walkthroughs[level_index] = internet.main(world,stage, level_index)
        print("Talking from main. Level ", level_index + 1, "complete. The sequence to complete it is:", level_walkthroughs[level_index])




