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


EXPLORATIONCONSTANT = 0.5
DEPTHLIMIT = 15
NUMBEROFSIMULATIONS = 10
#Was working with 5, 20, but for long paths the 20 sims took ages

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
    def __init__(self, move_sequence, terminated, parent, obs, action_index):
          
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
            
            '''env.reset()
            #print("PARENT MOVE SEQ", self.move_sequence)
            for move in self.move_sequence:
                obs, reward, terminated, truncated, info = env.step(move)
                #getting back to parent position
            obs, reward, terminated, truncated, info = env.step(child_action)
            #applying next move
            '''

            child_move_sequence = self.move_sequence.copy()
            child_move_sequence.append(child_action)
            #print("CHILD MOVE SEQ", child_move_sequence)
            

            #print("X_POS", info["x_pos"], " ACTION: ", child_action)
            #print("Y_POS", info["y_pos"], " ACTION: ", child_action)
            #childDict[child_action] = Node(child_move_sequence, terminated, self, obs, child_action)
            childDict[child_action] = Node(child_move_sequence, False, self, None, child_action)
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
    for move in self.move_sequence:
        obs, reward, terminated, truncated, info = env.step(move)
        #getting back to  position
        #print("POS NOW", info["x_pos"])

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



def test():
    obs, reward, terminated, truncated, info = env.step(0)
    topNode = Node([], terminated, None, obs, 0)
    
    current = topNode
    while True: #FIX
        chosen_child = policy(current)
        
        print("CHOSEN MOVE", chosen_child.move_sequence[-1])
        print("NEW MOVE SEQUENCE", chosen_child.move_sequence)
        env.reset()
        for move in chosen_child.move_sequence:
            obs, reward, terminated, truncated, info = env.step(move)
            #getting to new position
        

        current = chosen_child


        

        


    '''
    topNode.create_childDict()
    for child in topNode.childDict.values():
        child.create_childDict()
    print(topNode.childDict[3].action_index)

    for i in range (10):
        explore_world(topNode)
    '''

test()

