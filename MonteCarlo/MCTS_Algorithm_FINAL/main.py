'''
Robert Beashel 23489302
Martin Evans 23621647
'''

import agent

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import sys

#INPUT FORMAT = python3 main.py [render_mode ('rgb', 'human', ...)]

def main():
    args = sys.argv[1:]
    mode = args[0]

    level_walkthroughs = [[[] for stage in range(4)] for world in range(8)]

    f = open("levelwalkthroughs.txt", "a")
    #At the end, the walkthroughs (action sequences) to all levels will be saved in this file
    f.write("\n\n\n\n")

    for world in range (1,9):

        for stage in range (1,5): 
            
            level_walkthroughs[world-1][stage-1], useless_time_dict, useless_steps_dict, useless_deaths_dict, useless_score_dict, useless1, useless2 = agent.main(world, stage, [0], mode)
            
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


