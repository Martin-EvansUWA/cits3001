from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import math
import random
import time
import psutil
import os

#MOVE SEQUENCE
#0: ['NOOP']
#1: ['right'
#2: ['right', 'A']
#3: ['right', 'B']
#4: ['right', 'A', 'B']
#5: ['A']
#6: ['left']]

from gym.spaces import Box

EXPLORATIONCONSTANT = 0.5
DEPTHLIMIT = 15
NUMBEROFSIMULATIONS = 10
INITIALRECALCULATIONS = 5
GOALREWARDBONUS = 200
DEATHREWARDPENALTY = -30



class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            
            
            if done:
                print("DONE DURING 4peat after step", i)
                break
        return obs, total_reward, done, trunk, info



class Node:
    
    '''
    The Node class represents a node of the MCTS tree. 
    It contains the information needed for the algorithm to run its search.
    '''

    

    def __init__(self, move_sequence, terminated, parent, action_index, info):
          
        # child nodes is a dictionary of child nodes which, for every possible action from that state, will tell us what the next state of the game is taking that action
        self.childDict = None
        
        # total rewards from MCTS exploration (sum of value of rollouts) - maybe use reward
        self.total_reward = 0
        
        self.visitcount = 0        
 
        #the sequence of moves to get to this point
        self.move_sequence = move_sequence
        
        # if game is won/loss/draw
        self.terminated = terminated

        # link to parent node
        self.parent = parent
        
        # action index that leads to this node
        self.action_index = action_index

        #info index
        self.info = info

    def getUCBscore(self):
        
        '''
        This is the formula that gives a value to the node.
        The MCTS will pick the nodes with the highest value.
        Exploitation vs exploration        
        '''
        
        # Unexplored nodes have maximum values so we favour exploration
        if self.visitcount == 0:
            return float('inf')
        
        # We need the parent node of the current node 
        # If no parent, will be default assigned to itself (top node?)
        parent_node = self
        if parent_node.parent:
            parent_node = parent_node.parent
            
        # We use one of the possible MCTS formula for calculating the node value
        return (self.total_reward / self.visitcount) + EXPLORATIONCONSTANT * math.sqrt(math.log(parent_node.visitcount) / self.visitcount)

    def create_childDict(self):
        
        '''
        We create one children for each possible action of the game, 
        then we apply such action to a copy of the current node enviroment 
        and create such child node with proper information returned from the action executed
        '''
    
        actions = []

        for i in range(len(SIMPLE_MOVEMENT)):
            actions.append(i)          

        childDict = {} 

        for child_action in actions:

            child_move_sequence = self.move_sequence.copy()
            child_move_sequence.append(child_action)

            childDict[child_action] = Node(child_move_sequence, self.terminated, self, child_action, self.info)
            #creates a entry in the childDict for every possible move                
            
        self.childDict = childDict


def limitedSimulation(self, env, steps, deaths):
    #This part might need to be changed from random to not random
    
    #print("Simulating starting from sequence", self.move_sequence)
    #Checks leaf is not goal

    returnValue = 0
    env.reset()
    info = {}
    info["flag_get"] = False
    if len(self.move_sequence) > 0:
        for move in self.move_sequence:

            if self.info["flag_get"]:
                print("goal reached before sim started")
                return GOALREWARDBONUS, steps, deaths
            elif self.terminated == True:
                #for performance
                deaths = deaths + 1
                print("dead before sim started")
                return DEATHREWARDPENALTY, steps, deaths
            else:
                obs, reward, self.terminated, truncated, self.info = env.step(move)
                #performance
                steps = steps + 1
                #getting back to  position
                

    count = 0
    terminated = self.terminated
    while not terminated and count < DEPTHLIMIT:

        move = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(move)
        #performance
        steps = steps + 1
        returnValue = returnValue + reward

        if terminated == True:
            if info["flag_get"]:
                print("Reached goal during sim")
                return reward + GOALREWARDBONUS, steps, deaths

            else:
                print("Dead during simulation (Shouldnt be called for goal)")
                #for performance
                deaths = deaths + 1
                return reward, steps, deaths
                #undecided if i want to incorporate my own penalty for death, becaause it shnould already have one (-15)
            
        count += 1

    return returnValue, steps, deaths

def explore_world(self, env, steps, deaths):
    
    #BEGINNING OF SELECTION STAGE
    node: Node
    node = self
    count = 0
        
    while not (node.childDict == None): #checks the current node has children

        childDict = node.childDict

        #Find the UCB value of most favourable move
        bestUCB = float('-inf')
        best_actions = []
        for childNode in childDict.values():
            if childNode.getUCBscore() >= bestUCB:
                bestUCB = childNode.getUCBscore()

        for childNode in childDict.values():
            if childNode.getUCBscore() == bestUCB:
                best_actions.append(childNode.move_sequence[-1])
                
        if len(best_actions) == 0:
            print("Eror no moves found", bestUCB)   #this check will be removed eventaulyl     

        #Of these moves, a random one is chosen, and the DFS continues
        next_action = random.choice(best_actions)                    
        node = node.childDict[next_action]

    #END OF SELECTION. NODE IS THE SELECTED LEAF NODE
    
    #BEGINNING OF EXPANSION STAGE

    if node.visitcount == 0:
        reward, steps, deaths = limitedSimulation(node, env, steps, deaths)
        node.total_reward = node.total_reward + reward
        
    else:
        #print(node.visitcount, " visits at sequence:", node.action_index)
        node.create_childDict()
        if not (node.childDict == None):
            node = random.choice(node.childDict)
            #select random child node for the simulation to begin at
        else:
            print("\n\n\n\n\nERROR\n\n\n\n\n\n")
        reward, steps, deaths = limitedSimulation(node, env, steps, deaths)
        node.total_reward  = node.total_reward 

    node.visitcount = node.visitcount + 1
    
    #END OF EXPANSION STAGE

    #UPDATE WITH BACKPROPAGATION
    
    parent_node : Node
    parent_node = node

    #print("BACKPROP BEGIN. Reward will be updated by ", parent_node.total_reward)
    while parent_node.parent:
        parent_node = parent_node.parent
        
        parent_node.total_reward = parent_node.total_reward + node.total_reward
        parent_node.visitcount += 1
    
    #BACKPROPAGATION UPDATING FINISHED
    return steps, deaths

def get_next_move(current):
        #i just changed it from reward to visit count
        child : Node
        max = float("-inf")
        best_children = []
        for child in current.childDict.values():
            print("Move  ", child.move_sequence[-1], "   has reward   ", child.total_reward, "  with  ", child.visitcount, "  moves.")
            if (child.total_reward / child.visitcount) >= max:
                max = (child.total_reward / child.visitcount)
                
        
        for child in current.childDict.values():
            if (child.total_reward / child.visitcount) == max:
                best_children.append(child)
        

        if len(best_children) == 0:
            print("ERROR NO BEST MOVE")
        
        chosen_child = random.choice(best_children)
        return chosen_child
    
def stuck(node, escapes):
    ago_node = node
    for i in range(INITIALRECALCULATIONS + escapes):
        ago_node = ago_node.parent
        #if not moved right in x moves, escape backwards
        #the 'and' check ensures its not moving dramatically back (e.g. due to a pipe), which is fine
    print("current x pos: ", node.info["x_pos"])
    print("x position from ", INITIALRECALCULATIONS + escapes, " moves ago: ", ago_node.info["x_pos"])
    return node.info["x_pos"] <= ago_node.info["x_pos"] and node.info["x_pos"] > ago_node.info["x_pos"] - 15
    
def policy(current, env, steps, deaths):
    for i in range (NUMBEROFSIMULATIONS):
        steps, deaths = explore_world(current, env, steps, deaths)
    chosen_child = get_next_move(current)
    return chosen_child, steps, deaths


            

def main(world, stage, starting_sequence, mode):
        #code for performance analysis 'memory'
        avgmemorySum = 0
        memCount = 0

        #code for performance analysis 'time'
        time_dict = {}
        start_time = time.time()
        time_distance_index = 200

        #code for performance analysis 'steps'
        steps_dict = {}
        steps = 0
        print("\n\n\n\n\n\n\n", steps)
        step_distance_index = 200

        #code for performance analysis 'deaths'
        deaths_dict = {}
        deaths = 0
        death_distance_index = 200

        #code for performance analysis 'score'
        score_dict = {}
        score_distance_index = 200
        
        non_simulated_deaths = 0
        escapes = 0

        make_string = "SuperMarioBros-" + str(world) + "-" + str(stage) + "-v0"
        env = gym.make(make_string, apply_api_compatibility=True, render_mode=mode)
        env = SkipFrame(env, skip=4)
        print(SIMPLE_MOVEMENT)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        done = True
        env.reset()

        #The following block of code allows me to run trials of learning with part of the route already cheated, for quicker testing 
        #and not having to wait for mario to get up to the part i want to test every time.

        node = []
        obs, reward, terminated, truncated, info = env.step(0)
        #performance
        steps = steps + 1
        temp = Node([starting_sequence[0]], terminated, None, starting_sequence[0], info)
        node.append(temp)
        for move in range (1, len(starting_sequence)):
            obs, reward, terminated, truncated, info = env.step(starting_sequence[move])
            #performance
            steps = steps + 1
            temp = Node(starting_sequence[:move+1], False, node[move-1], starting_sequence[move], info)
            node.append(temp)
            node[move].create_childDict()
        current = temp


        while True:
            #performance parameters
            chosen_child, steps, deaths = policy(current, env, steps, deaths)
            
            print("CHOSEN MOVE", chosen_child.move_sequence[-1])
            print("NEW MOVE SEQUENCE", chosen_child.move_sequence)
            print("WORLD", world, "STAGE", stage)
            env.reset()
            terminated = False
            for move in chosen_child.move_sequence:
                if not terminated:
                    obs, reward, terminated, truncated, info = env.step(move)
                    #performance
                    steps = steps + 1
                #getting to new position
            
            chosen_child.terminated = terminated
            chosen_child.info = info

            child_level_index = chosen_child.info["stage"] * chosen_child.info["world"] - 1

            print("Flag reached: ", chosen_child.info["flag_get"])
            if (chosen_child.info["flag_get"]):
                #Check for level completion

                print("WORLD", world, "STAGE", stage, "complete")

                print(chosen_child.move_sequence)
                env.close()
                return chosen_child.move_sequence, time_dict, steps_dict, deaths_dict, score_dict, (avgmemorySum / memCount)

            else:
                #Check for canonical death
                if chosen_child.terminated == True:
                    #for performance
                    deaths = deaths + 1
                    print("\n\nRecalculating last ", INITIALRECALCULATIONS + non_simulated_deaths, " moves due to DEATH\n\n")
                    for backstep in range (INITIALRECALCULATIONS + non_simulated_deaths):
                        if current.parent == None:
                                print("Top node reached. No more parents")
                                break
                        current = current.parent
                    non_simulated_deaths = non_simulated_deaths + 1
                else:
                    if len(current.move_sequence) > INITIALRECALCULATIONS + escapes:
                        if stuck(chosen_child, escapes):
                            print("\n\nRecalculating last ", INITIALRECALCULATIONS + escapes, " moves due to STUCK\n\n")
                            for backstep in range (INITIALRECALCULATIONS + escapes):
                                if current.parent == None:
                                    print("Top node reached. No more parents")
                                    break
                                current = current.parent
                            escapes = escapes + 1
                        else:
                            current = chosen_child
                    else:
                        current = chosen_child

                #for time performance analysis
                if(time_distance_index not in time_dict.keys() and current.info["x_pos"] >= time_distance_index):
                    print("saving ", time.time() - start_time, "to index: ", time_distance_index)
                    time_dict[time_distance_index] =  time.time() - start_time
                    time_distance_index += 200   

                #for steps performance analysis
                if(step_distance_index not in steps_dict.keys() and current.info["x_pos"] >= step_distance_index):
                    print("saving ", steps, "to index: ", step_distance_index)
                    steps_dict[step_distance_index] =  steps
                    step_distance_index += 200  
                
                #for death performance analysis
                if(death_distance_index not in deaths_dict.keys() and current.info["x_pos"] >= death_distance_index):
                    print("saving ", deaths, "to index: ", death_distance_index)
                    deaths_dict[death_distance_index] =  deaths
                    death_distance_index += 200   

                #for score performance analysis
                if(score_distance_index not in score_dict.keys() and current.info["x_pos"] >= score_distance_index):
                    print("saving ", info["score"], "to index: ", score_distance_index)
                    score_dict[score_distance_index] = info["score"]
                    score_distance_index += 200  
                
                #memory perf analysis
                memory = psutil.Process().memory_info().rss / (1024**2)
                avgmemorySum = avgmemorySum + memory
                memCount = memCount + 1
                print("Memory usage: ", memory)

                print("Deaths: ", deaths)
                print("Time dict: ", time_dict)
                print("Steps dict: ", steps_dict)
                print("deaths dict:", deaths_dict)
                print("score dict:", score_dict)
    
                env.reset()



#steps, and deaths scope should be fixed. Cant figure out the global scope.