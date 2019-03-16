import numpy as np
import random
from random import shuffle
from time import time, sleep
from collections import deque
import os.path as op

from settings import s
from settings import e

def setup(self):
    np.random.seed() 
    # Q matrix
    if (op.isfile("\agent_code\my_agent\Q.txt") == False):
        Q = np.zeros((176,5),dtype = float)
        np.savetxt("agent_code\my_agent\Q.txt", Q)
        
    self.coordinate_history = deque([], 20)
   
    self.logger.info('Initialize')
        
        
def act(self):    
    Q = np.loadtxt("agent_code\my_agent\Q.txt")
    arena = self.game_state['arena']
    x, y, _, bombs_left, score = self.game_state['self']
    self.logger.debug(f'(x,y): {(x,y)}')
    self.coordinate_history.append((x,y))

    

    
    epsilon = 0.1    
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT' , 'WAIT']
    shuffle(action_ideas)        
    if np.random.rand(1) <= epsilon:
        self.next_action = action_ideas.pop()
    else:
        accessible = []
        for a in range(16):
            for b in range(16):
                if (arena[a][b] != -1):
                    accessible.append((a,b))
        self.logger.debug(f'accessible: {accessible}')
        index = accessible.index((x,y))
        self.logger.debug(f'index: {index}')
                    
        q_state = Q[index]
        act = np.argmax(q_state)
        if act == 0: action_ideas.append('UP')
        if act == 1: action_ideas.append('DOWN')
        if act == 2: action_ideas.append('LEFT')
        if act == 3: action_ideas.append('RIGHT')
        if act == 4: action_ideas.append('WAIT')        
        self.next_action = action_ideas.pop()
    self.logger.info('Pick action')

def reward_update(self): 
    
    Q = np.loadtxt("agent_code\my_agent\Q.txt")
    alpha = 1
    
    
    arena = self.game_state['arena']
    x, y, _, bombs_left, score = self.game_state['self']
    accessible = []
    for a in range(16):
        for b in range(16):
            if (arena[a][b] != -1):
                accessible.append((a,b))
        
    state = accessible.index((x,y))
            
    reward = np.zeros((176,5),dtype = float)
    if self.events == e.MOVED_LEFT: reward[state][0] =  reward[state][0] - 1
    if self.events == e.MOVED_RIGHT: reward[state][1] =  reward[state][1] - 1
    if self.events == e.MOVED_UP: reward[state][2] =  reward[state][2] - 1
    if self.events == e.MOVED_DOWN: reward[state][3] =  reward[state][3] - 1
    if self.events == e.WAITED: reward[state][4] =  reward[state][4] - 5
    if self.events == e.COIN_COLLECTED: reward[state][4] =  reward[state][4] + 100
    self.logger.debug(f'Reward: {reward}')
    self.logger.debug(f'Encountered {len(self.events)} game event(s)')
    
    Q = Q + alpha * (reward)
    np.savetxt("agent_code\my_agent\Q.txt", Q)
    
    self.logger.debug(f'Q: {Q}')

def end_of_episode(self):  
#    Q = np.loadtxt("agent_code\my_agent\Q.txt")
#    alpha = 1
#    gamma = 0.9 
#    
#    arena = self.game_state['arena']
#    accessible = []
#    for a in range(16):
#        for b in range(16):
#            if (arena[a][b] != -1):
#                accessible.append((a,b))
#    
#    for state in accessible:
#        if self.next_action == 'UP': next_state = accessible.index((x,y+1))
#        if self.next_action == 'DOWN': next_state = accessible.index((x,y-1))
#        if self.next_action == 'LEFT': next_state = accessible.index((x-1,y))
#        if self.next_action == 'RIGHT': next_state = accessible.index((x+1,y))
#        if self.next_action == 'WAIT': next_state = accessible.index((x,y))
#        max_Q_next = np.argmax(Q[next_state])    
#    Q = Q + alpha * (reward)
#    np.savetxt("agent_code\my_agent\Q.txt", Q)
#    
#    self.logger.debug(f'Q: {Q}')
#    
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')