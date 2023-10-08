import internet

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import math
import random
import sys

#FOR NOW: ARG FORMAT = [mode (rgb, human etc.), startingworld, startingstage]




def main():
    args = sys.argv[1:]
    mode = args[0]
    starting_world = int(args[1]) #1-8
    starting_stage = int(args[2]) #1-4

    starting_sequence =  [0]

    level_walkthroughs = [[[] for stage in range(4)] for world in range(8)]
    print(level_walkthroughs)

    f = open("demofile.txt", "a")
    f.write("\n\n\n\n")


    for world in range (starting_world,9):
        for stage in range (starting_stage,5): 
            
            level_walkthroughs[world-1][stage-1] = internet.main(world, stage, starting_sequence, mode)
            print("Talking from main. world ", world, "stage ", stage, "complete. The sequence to complete it is:", level_walkthroughs[world-1][stage-1])
            f.write("WORLD: ")
            f.write(str(world))
            f.write(" STAGE: ")
            f.write(str(stage))
            f.write( " SOLUTION ")
            f.write(str(level_walkthroughs[world-1][stage-1]))
            f.write("\n\n")
    
        f.close()



if __name__ == '__main__':
    main()

