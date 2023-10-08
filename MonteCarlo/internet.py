from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import math
import random

#MOVE SEQUENCE
#0: ['NOOP']
#1: ['right'
#2: ['right', 'A']
#3: ['right', 'B']
#4: ['right', 'A', 'B']
#5: ['A']
#6: ['left']]

from gym.spaces import Box

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



EXPLORATIONCONSTANT = 0.5
DEPTHLIMIT = 15
NUMBEROFSIMULATIONS = 10
INITIALRECALCULATIONS = 5
GOALREWARDBONUS = 200
DEATHREWARDPENALTY = -30
#Was working with 5, 20, but for long paths the 20 sims took ages

class Node:
    
    '''
    The Node class represents a node of the MCTS tree. 
    It contains the information needed for the algorithm to run its search.
    '''

    #def __init__(self, move_sequence, env, terminated, parent, obs, action_index):
    def __init__(self, move_sequence, terminated, parent, obs, action_index, info):
          
        # child nodes is a dictionary of child nodes which, for every possible action from that state, will tell us what the next state of the game is taking that action
        self.childDict = None
        
        # total rewards from MCTS exploration (sum of value of rollouts) - maybe use reward
        self.total_reward = 0
        
        self.visitcount = 0        
 
        #the sequence of moves to get to this point
        self.move_sequence = move_sequence
        
        # observation of the environment
        self.obs = obs
        
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

            childDict[child_action] = Node(child_move_sequence, self.terminated, self, None, child_action, self.info)
            #creates a entry in the childDict for every possible move                
            
        self.childDict = childDict


def limitedSimulation(self, env):
    #This part might need to be changed from random to not random
    
    #print("Simulating starting from sequence", self.move_sequence)
    #Checks leaf is not goal

    returnValue = 0
    env.reset()
    info = {}
    info["flag_get"] = False
    #print("MOVE SEQ", self.move_sequence)
    if len(self.move_sequence) > 0:
        for move in self.move_sequence:
            #print("TERMINATED: ", terminated)

            if self.info["flag_get"]:
                print("goal reached before sim started")
                return GOALREWARDBONUS
            elif self.terminated == True:
                print("dead before sim started")
                return DEATHREWARDPENALTY
            else:
                obs, reward, self.terminated, truncated, self.info = env.step(move)
                #getting back to  position
                #print("POS NOW", info["x_pos"])

    count = 0
    terminated = self.terminated
    while not terminated and count < DEPTHLIMIT:
        #print("MOVE #", count)
        move = env.action_space.sample()
        #print("Chosen rand MOVE ", move)
        obs, reward, terminated, truncated, info = env.step(move)
        #print(terminated)
        returnValue = returnValue + reward

        if terminated == True:
            if info["flag_get"]:
                print("Reached goal during sim")
                return reward + GOALREWARDBONUS

            else:
                print("Dead during simulation (Shouldnt be called for goal)")
                return reward
                #undecided if i want to incorporate my own penalty for death, becaause it shnould already have one (-15)
            
        count += 1

    return returnValue

def explore_world(self, env):
    
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
        #print("No visits at sequence:", node.action_index)
        #print("EXPANSION PHASE OVER. SIMULATION BEING RAN FROM PARENT = ", node.action_index)
        node.total_reward = node.total_reward + limitedSimulation(node, env)
        
    else:
        #print(node.visitcount, " visits at sequence:", node.action_index)
        node.create_childDict()
        if not (node.childDict == None):
            node = random.choice(node.childDict)
            #select random child node for the simulation to begin at
        else:
            print("\n\n\n\n\nERROR\n\n\n\n\n\n")
        node.total_reward = node.total_reward + limitedSimulation(node, env)
        #print("EXPANSION PHASE OVER. SIMULATION BEING RAN FROM CHILD = ", node.action_index)
    
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
        #if moved less than 5 units right in x moves, time to escape backwards
    return node.info["x_pos"] <= ago_node.info["x_pos"] + 5
    
def policy(current, env):
    for i in range (NUMBEROFSIMULATIONS):
        explore_world(current, env)
    chosen_child = get_next_move(current)
    return chosen_child

def main(world, stage, starting_sequence, mode):
    
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
    temp = Node([starting_sequence[0]], terminated, None, None, starting_sequence[0], info)
    node.append(temp)
    for move in range (1, len(starting_sequence)):
        obs, reward, terminated, truncated, info = env.step(starting_sequence[move])
        temp = Node(starting_sequence[:move+1], False, node[move-1], None, starting_sequence[move], info)
        node.append(temp)
        node[move].create_childDict()
    current = temp


    while True: #FIX
        
        chosen_child = policy(current, env)
        
        print("CHOSEN MOVE", chosen_child.move_sequence[-1])
        print("NEW MOVE SEQUENCE", chosen_child.move_sequence)
        print("WORLD", world, "STAGE", stage)
        env.reset()
        terminated = False
        for move in chosen_child.move_sequence:
            if not terminated:
                obs, reward, terminated, truncated, info = env.step(move)
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
            return chosen_child.move_sequence
        
            #level_walkthroughs[parent_level_index] = current.move_sequence
            #print(level_walkthroughs[parent_level_index])
            #current = chosen_child

        else:
            #Check for canonical death
            if chosen_child.terminated == True:
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
                
                
            #current = check_child(chosen_child)
            env.reset()
            

        

        


'''
FUNCTIONALITY TO BE ADDED
- 
- What is truncation????
- Something to do with obs
'''

