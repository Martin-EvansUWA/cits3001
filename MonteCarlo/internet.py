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

level_walkthroughs = [[] * 32]

EXPLORATIONCONSTANT = 0.5
DEPTHLIMIT = 5
NUMBEROFSIMULATIONS = 10
RECALCULATIONS = 3
#Was working with 5, 20, but for long paths the 20 sims took ages

non_simulated_deaths = 0
#used to incrementally increase number of recalculations if keep dying on same thing

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
        
        # visit count
        self.visitcount = 0        
                
        # the environment in current state
       # self.env = env

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
        
        if self.terminated:
            return
    
        actions = []
        envs = []

        for i in range(len(SIMPLE_MOVEMENT)):
            actions.append(i)          

        childDict = {} 

        for child_action in actions:

            child_move_sequence = self.move_sequence.copy()
            child_move_sequence.append(child_action)

            childDict[child_action] = Node(child_move_sequence, False, self, None, child_action, None)
            #creates a entry in the childDict for every possible move                
            
        self.childDict = childDict


def limitedSimulation(self, env):
    #This part might need to be changed from random to not random
    
    #print("Simulating starting from sequence", self.move_sequence)
    #Checks leaf is not goal
    if self.terminated == True:
        print("Dead")
        terminated = True
        return 0
    else: 
        terminated = False

    returnValue = 0
    env.reset()
    #print("MOVE SEQ", self.move_sequence)
    if len(self.move_sequence) > 0:
        for move in self.move_sequence:
            obs, reward, terminated, truncated, info = env.step(move)
            #getting back to  position
            #print("POS NOW", info["x_pos"])
        
        self.info = info

    count = 0
    while not terminated and count < DEPTHLIMIT:
        #print("MOVE #", count)
        move = env.action_space.sample()
        #print("Chosen rand MOVE ", move)
        obs, reward, terminated, truncated, info = env.step(move)
        #print(terminated)
        returnValue = returnValue + reward
        if terminated == True:
            print("Dead during simulation (Check its not at goal lol)")
        count += 1

    return returnValue

def explore_world(self):
    
    #BEGINNING OF SELECTION STAGE
    node = self
    count = 0
        
    while not (node.childDict == None): #checks the current node has children
        #print("Node has children after sequence:", node.action_index)
        #print("CHILDDICT:", node.childDict)
        node: Node

        childDict = node.childDict

        #Find the UCB value of most favourable move
        bestUCB = float('-inf')
        best_actions = []
        for childNode in childDict.values():
            if childNode.getUCBscore() >= bestUCB:
                bestUCB = childNode.getUCBscore()
                
        #print("BEST UCB:", bestUCB)

        for childNode in childDict.values():
            if childNode.getUCBscore() == bestUCB:
                best_actions.append(childNode.move_sequence[-1])
                #print("Added child", childNode.move_sequence," with UCB:", childNode.getUCBscore())
        

        #A list should be constructed with all moves that result in the MAX ucb score
        #This is because multiple moves could share the same UCB result
        '''
        for child_move in childDict.keys():
            if childDict[child_move].getUCBscore == bestUCB:
                best_actions.append(child_move)'''
        
        if len(best_actions) == 0:
            print("Eror no moves found", bestUCB)   #this check will be removed eventaulyl     

        #Of these moves, a random one is chosen, and the DFS continues
        next_action = random.choice(best_actions)                    
        node = node.childDict[next_action]
    
    #print("SELECTION PHASE OVER. SELECTED NODE = ", node.action_index)
    #END OF SELECTION. NODE IS THE SELECTED LEAF NODE. READY FOR EXPANSION
    
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
            
        while parent_node.parent:
            parent_node = parent_node.parent

            parent_node.total_reward = parent_node.total_reward + node.total_reward
            parent_node.visitcount += 1
    
    #BACKPROPAGATION UPDATING FINISHED

def get_next_move(current):

        max_reward = float('-inf')
        best_children = []
        print("CHILDDICT VALUES", current.childDict.values())
        for child in current.childDict.values():
            if child.total_reward >= max_reward:
                print("CHILD with move ", child.move_sequence[-1], " has reward ", child.total_reward)
                
                max_reward = child.total_reward
                
        
        for child in current.childDict.values():
            if child.total_reward == max_reward:
                best_children.append(child)
        

        if len(best_children) == 0:
            print("ERROR NO BEST MOVE")
        
        chosen_child = random.choice(best_children)
        return chosen_child
    

def policy(current):
    for i in range (NUMBEROFSIMULATIONS):
        explore_world(current)
    chosen_child = get_next_move(current)
    return chosen_child

def check_child(potential_child):
    try:
        print("PARENT X POS:", potential_child.parent.info["x_pos"])
        print("Child X POS:", potential_child.info["x_pos"])

        if (potential_child.info["x_pos"] + 30) < potential_child.parent.info["x_pos"]:
            #check that the character has not moved dramatically backwards (for now only implying death)

            print("This move has pushed us to the left a lot. Recalculating the last ", RECALCULATIONS + non_simulated_deaths, "moves")
            current = potential_child
            for backstep in range (RECALCULATIONS + non_simulated_deaths):
                current = current.parent
            return current

        else:
            print("Child is fine")
            return potential_child 
    except:
        print("top node. no parent")
        return potential_child

def main():
    obs, reward, terminated, truncated, info = env.step(0)
    topNode = Node([2, 2, 2, 3, 1, 1, 0, 5, 3, 1, 5, 0, 3, 1, 3, 0, 5, 4, 1, 2, 5, 5, 3, 3, 4, 5, 2, 3, 4, 1, 2, 6, 2, 6, 3, 3, 4, 5, 4, 5, 4, 5, 3, 5, 1, 4, 2, 5, 6, 5, 3, 6, 0, 3, 2, 4, 4, 6, 4, 6, 1, 5, 6, 6, 0, 6, 6, 5, 5, 6, 6, 2, 1, 4, 4, 2, 6, 0, 0, 0, 5, 1, 6, 1, 6, 1, 6, 2, 5, 1, 6, 5, 6, 6, 2, 3, 4, 5, 2, 0, 6, 1, 6, 4, 4, 6, 0, 1, 4, 6, 2, 0, 0, 6, 6, 1, 1, 6, 3, 0, 3, 6, 1, 6, 5, 0, 0, 1, 4, 6, 5, 2, 1, 6, 4, 5, 6, 0, 3, 2, 3, 2, 0, 6, 3, 0, 6, 4, 1, 6, 5, 1, 6, 4, 4, 6, 2, 6, 1, 0, 3, 0, 4, 2, 5, 2, 5, 3, 1, 3, 1, 4, 1, 3, 0, 2, 0, 4, 1, 2, 1, 3, 0, 0, 4, 5, 5, 1, 3, 3, 3, 1, 3, 2, 1, 2, 0, 3, 5, 4, 2, 4, 1, 5, 5, 1, 5, 0, 4, 5, 3, 5, 5, 4, 1, 0, 4, 4, 0, 4, 4, 2, 3, 3, 2, 2, 4, 5, 4, 3, 4, 0, 0, 4, 6, 2, 1, 1, 2, 4, 0, 2, 2, 1, 4, 0, 2, 0, 2, 5, 3, 5, 0, 2, 0, 6, 3, 2, 4, 6, 3, 4, 0, 2, 0, 5, 0, 3, 3, 6, 2, 0, 2, 2, 5, 2, 3, 1, 5, 4, 4, 5, 5, 1, 1, 2, 2, 1, 3, 5, 5, 4, 2, 5, 6, 2, 4, 2, 0, 5, 0, 2, 2, 5, 4, 0, 1, 1, 4, 1, 4, 4, 4, 0, 6, 2, 4, 1, 5, 6, 1, 6, 5, 2, 5, 4, 4, 5, 4, 3, 1, 3, 1, 0, 2, 5, 5, 3, 5, 0, 1, 3, 2, 0, 1, 1, 4, 4, 3, 1, 1, 1, 0, 0, 5, 3, 4, 5, 3, 4, 2, 0, 4, 1, 5, 5, 3, 6, 1, 3, 2, 0, 1, 5, 5, 1, 5, 4, 1, 2, 0, 3, 6, 2, 0, 0, 1, 1, 1, 4, 4, 2, 6, 1, 3, 2, 4, 3, 4, 4, 3, 4, 0, 1, 3, 0,5,4,3,2], terminated, None, None, 0, info)
    
    current = topNode
    while True: #FIX
        chosen_child = policy(current)
        
        print("CHOSEN MOVE", chosen_child.move_sequence[-1])
        print("NEW MOVE SEQUENCE", chosen_child.move_sequence)
        env.reset()
        for move in chosen_child.move_sequence:
            obs, reward, terminated, truncated, info = env.step(move)
            #getting to new position
        
        chosen_child.info = info

        child_level_index = chosen_child.info["stage"] * chosen_child.info["world"] - 1
        parent_level_index = current.info["stage"] * current.info["world"] - 1

        if (not child_level_index == parent_level_index):

            print("LEVEL", parent_level_index + 1 , "complete")

            print(current.move_sequence)
            level_walkthroughs[parent_level_index] = current.move_sequence
            print(level_walkthroughs[parent_level_index])
            current = chosen_child
            #some code to restart whole process 
        else:
            current = check_child(chosen_child)
            env.reset()
            

        

        



main()

'''
FUNCTIONALITY TO BE ADDED
- If death happens -> reversal of move(s)? Go backwards and pop from list? DONE
- Saving a move array NEXT
- Dealing with completion of level NEXT
- Escaping maxima of shit - not death
- What is truncation????
- Something to do with obs
'''

