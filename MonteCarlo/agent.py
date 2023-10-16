from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import math
import random
import os

from gym.spaces import Box

EXPLORATIONCONSTANT = 0.5
DEPTHLIMIT = 15
NUMBEROFSIMULATIONS = 10
INITIALRECALCULATIONS = 5
GOALREWARDBONUS = 200
DEATHREWARDPENALTY = -30

#The following class allows for frame skipping to occur
#3 frames are skipped for every 'step' call to allow more natural movement flow
#I did not write this code. It is referenced in our report
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            
            if done:
                break

        return obs, total_reward, done, trunk, info



class Node:
    
    #A 'Node' object contains information on a single entry in the action tree
    #In simple terms, one Node represents one environment state 

    def __init__(self, move_sequence, terminated, parent, action_index, info):
          
        #A dictionary of Nodes containing every possible child node in the tree from the current node
        #One Node for every possible action from the current state
        self.childDict = None
        
        #Number of simulations that have stemmed from this Node
        self.visitcount = 0  

        #Sum of 'reward' values obtained by simulation that stemming from this Node
        self.total_reward = 0     
 
        #The sequence of actions required to get to this state
        self.move_sequence = move_sequence
        
        #Whether or not the game is terminated at current state
        self.terminated = terminated

        #The parent Node
        self.parent = parent
        
        #Most recent action that leads to this node
        self.action_index = action_index

        #Info dictionary of the current state
        self.info = info

    def getUCBscore(self):
        #Formula that assigns value to Node DURING SELECTION PHASE
        
        if self.visitcount == 0:
            #Favouring exploration over exploitation during selection phase
            return float('inf')
        
        # We need the parent node of the current node 
        # If no parent (top node) will be stay assigned to itself
        parent_node = self
        if parent_node.parent:
            parent_node = parent_node.parent
            
        # 'Upper Confidence bound1 for Trees' formula to assign value to node
        return (self.total_reward / self.visitcount) + EXPLORATIONCONSTANT * math.sqrt(math.log(parent_node.visitcount) / self.visitcount)

    def create_childDict(self):
        #Create dictionary of child nodes and assign it to relevant field of Node
        
        childDict = {} 

        actions = []
        for i in range(len(SIMPLE_MOVEMENT)):
            actions.append(i)          

        for child_action in actions:

            child_move_sequence = self.move_sequence.copy()
            child_move_sequence.append(child_action)

            childDict[child_action] = Node(child_move_sequence, self.terminated, self, child_action, self.info)
            # Creates an entry in the childDict for every possible move (from SIMPLE_MOVEMENT)              
            
        self.childDict = childDict
        #Assigns the childDict to the Node


def limitedSimulation(self, env):
    
    # If the Node has a valid move_sequence (its not the top node), reset the environment and apply
    # the sequence of moves to get to the required state 
    
    #BEGINNING OF SIMULATION STAGE
    if len(self.move_sequence) > 0:
        env.reset()
        info = {}
        info["flag_get"] = False
        
        for move in self.move_sequence:

            if self.info["flag_get"]:

                print("Goal reached before sim started")
                return GOALREWARDBONUS
            
            elif self.terminated == True:
            
                print("Dead before sim started")
                return DEATHREWARDPENALTY
            
            else:
                obs, reward, self.terminated, truncated, self.info = env.step(move)               

    total_reward = 0         
    count = 0
    terminated = self.terminated

    while not terminated and count < DEPTHLIMIT:

        # Perform [DEPTHLIMIT] random actions from the state of the Node, 
        move = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(move)

        total_reward = total_reward + reward
        # Accumulate the total reward after each step

        # If the agent dies or gets the flag before the simulation
        # should be over, return accordingly
        if terminated == True:

            if info["flag_get"]:
                print("Reached goal during simulation")
                return reward + GOALREWARDBONUS

            else:
                print("Dead during simulation")
                return reward + DEATHREWARDPENALTY
            
        count += 1

    return total_reward
    #END OF SIMULATION STAGE

def explore_world(self, env):
    #This function is where the main MCTS cycle takes place.
    #The 4 stages are shown pretty clearly

    #BEGINNING OF SELECTION STAGE
    node: Node
    node = self
    count = 0
        
    while not (node.childDict == None): #checks the current Node has children

        childDict = node.childDict

        #Find the UCB value of most favourable move
        bestUCB = float('-inf')
        
        for childNode in childDict.values():
            if childNode.getUCBscore() >= bestUCB:
                bestUCB = childNode.getUCBscore()

        # Find action(s) resulting in the best UCB value
        # This exists as multiple nodes could share the best UCB value
        best_actions = []
        for childNode in childDict.values():
            if childNode.getUCBscore() == bestUCB:
                best_actions.append(childNode.move_sequence[-1])

        #Of these moves, a random one is chosen
        best_action = random.choice(best_actions)                    
        node = node.childDict[best_action] # 'node' represents selected leaf node

    #END OF SELECTION STAGE

    #BEGINNING OF EXPANSION STAGE

    # If the chosen node is unvisited, simulate from it
    if node.visitcount == 0:
        reward = limitedSimulation(node, env)
        node.total_reward = node.total_reward + reward

    #If it has already been visited, select one of its children at random and simulate from there 
    else:
        node.create_childDict()
        node = random.choice(node.childDict)

        reward = limitedSimulation(node, env)
        node.total_reward  = node.total_reward + reward

    node.visitcount = node.visitcount + 1
    
    #END OF EXPANSION STAGE

    #BEGINNING OF BACKPROPAGATION STAGE
    
    parent_node : Node
    parent_node = node

    #Backpropagate reward and visitcount values all the way up through the tree to the root node
    while parent_node.parent:
        parent_node = parent_node.parent
        
        parent_node.total_reward = parent_node.total_reward + node.total_reward
        parent_node.visitcount += 1
    
    #END OF BACKPROPAGATION STAGE

def get_next_move(current):
        child : Node
        max = float("-inf")
        best_children = []

        #Finds the max reward per visit of the potential actions from the state (child nodes)
        for child in current.childDict.values():
            print("Move  ", child.move_sequence[-1], "   has reward   ", child.total_reward, "  with  ", child.visitcount, "  moves.")
            if (child.total_reward / child.visitcount) >= max:
                max = (child.total_reward / child.visitcount)
        
        # This maximum value could be shared between multiple children.
        # If it is, a random child with this maximum value is chosen and returned.
        # This is the chosen move
        for child in current.childDict.values():
            if (child.total_reward / child.visitcount) == max:
                best_children.append(child)
        chosen_child = random.choice(best_children)

        return chosen_child


def stuck(node, escapes):

    # The stuck check ensures the node is not 'stuck' around the same x position it was 
    # a variable (incremented by 'escapes' in main) number of moves ago.

    past_node = node
    for i in range(INITIALRECALCULATIONS + escapes):
        past_node = past_node.parent

    #the 'and' check ensures its not moving dramatically back (e.g. due to a pipe), which is fine
    return node.info["x_pos"] <= past_node.info["x_pos"] and node.info["x_pos"] > past_node.info["x_pos"] - 15


def handler(current, env):
    
    #This function runs the MCTS cycle NUMBEROFSIMULATIONS times, resulting in a fully updated tree
    #The function then calls get_next_move, and returns the result to main()

    for i in range (NUMBEROFSIMULATIONS):
        explore_world(current, env)

    return get_next_move(current)

def main(world, stage, starting_sequence, mode):
        
        #These values used to increment the number of recalculations for a stuck/dead agent
        non_simulated_deaths = 0
        escapes = 0
        
        #Environment setup
        make_string = "SuperMarioBros-" + str(world) + "-" + str(stage) + "-v0"
        env = gym.make(make_string, apply_api_compatibility=True, render_mode=mode)
        env = SkipFrame(env, skip=4)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        done = True
        env.reset()

        
        obs, reward, terminated, truncated, info = env.step(starting_sequence[0])
        current = Node([starting_sequence[0]], terminated, None, starting_sequence[0], info)

        while True:
            # Getting the chosen child move
            chosen_child = handler(current, env)
            
            print("CHOSEN MOVE", chosen_child.move_sequence[-1])
            print("NEW MOVE SEQUENCE", chosen_child.move_sequence)

            env.reset()
            terminated = False

            # Getting to state of chosen child
            for move in chosen_child.move_sequence:
                if not terminated:
                    obs, reward, terminated, truncated, info = env.step(move)
            
            # Update fields of Node object with returned values
            chosen_child.terminated = terminated
            chosen_child.info = info

            #Check for level completion
            if (chosen_child.info["flag_get"]):

                print("WORLD", world, "STAGE", stage, "complete")
                print("Move sequence to complete it: ", chosen_child.move_sequence)

                env.close()
                return chosen_child.move_sequence

            else:
                # If the agent has died as a result of the added move, a variable (incremented by 'non_simulated_deaths')
                # number of the last moves will be recalculated
                if chosen_child.terminated == True:
                    print("\n\nRecalculating last ", INITIALRECALCULATIONS + non_simulated_deaths, " moves due to DEATH\n\n")

                    for backstep in range (INITIALRECALCULATIONS + non_simulated_deaths):
                        
                        if current.parent == None:
                            print("Top node reached. No more parents")
                            break
                        else:
                            current = current.parent

                    non_simulated_deaths = non_simulated_deaths + 1

                
                else: #The new move has NOT caused death

                    if len(current.move_sequence) > INITIALRECALCULATIONS + escapes:

                        # If the agent is 'stuck' (see stuck function), recalculate a (variable)
                        # number of the last moves
                        if stuck(chosen_child, escapes):

                            print("\n\nRecalculating last ", INITIALRECALCULATIONS + escapes, " moves due to being STUCK\n\n")

                            for backstep in range (INITIALRECALCULATIONS + escapes):

                                if current.parent == None:
                                    print("Top node reached. No more parents")
                                    break
                                else:
                                    current = current.parent

                            escapes = escapes + 1
                        else:
                            current = chosen_child
                    else:
                        current = chosen_child
                    #If the agent is not stuck or dead, assign the current state to the returned child state
                env.reset()
