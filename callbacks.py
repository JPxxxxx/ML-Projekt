import numpy as np
import random
from random import shuffle
from time import time, sleep
from collections import deque
import os.path as op

from settings import s
from settings import e

import sklearn.pipeline
import sklearn.preprocessing

from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

def statefun(arena,agent_row,agent_col,coins):   
    accessible = []
    for a in range(16):
        for b in range(16):
            if (arena[a][b] != -1):
                accessible.append((a,b))    
    state=accessible.index((agent_row,agent_col))
    for i in range(len(coins)):
        state = np.append(state,accessible.index((coins[i][0],coins[i][1])))
    print(state)
    return state

def setup(self):
    np.random.seed()    
    # Q matrix
    if op.isfile("Q.txt") != True:        
        Q = np.array([np.zeros((12,5),dtype=float)])
        np.savetxt("Q.txt", Q)
    self.coordinate_history = deque([], 20)
    self.logger.info('Initialize')
    
def act(self):    
    Q = np.loadtxt("Q.txt")
    arena = self.game_state['arena']
    coins = self.game_state['coins']
    x, y, _, bombs_left, score = self.game_state['self']
    self.coordinate_history.append((x,y))

    epsilon = 0.7
           
    if np.random.rand(1) <= epsilon:
        action_ideas = ['UP','DOWN','RIGHT','LEFT','WAIT']
        shuffle(action_ideas) 
        self.next_action = action_ideas.pop()
    else:
        action_ideas = ['UP','DOWN','RIGHT','LEFT','WAIT']
        shuffle(action_ideas) 
        
        f1 = x
        f2 = y
        d1 = np.abs(coins[1][0] - x) + np.abs(coins[1][1] - y)
        d2 = np.abs(coins[2][0] - x) + np.abs(coins[2][1] - y)
        d3 = np.abs(coins[3][0] - x) + np.abs(coins[3][1] - y)
        d4 = np.abs(coins[4][0] - x) + np.abs(coins[4][1] - y)
        d5 = np.abs(coins[5][0] - x) + np.abs(coins[5][1] - y)
        d6 = np.abs(coins[6][0] - x) + np.abs(coins[6][1] - y)
        d7 = np.abs(coins[7][0] - x) + np.abs(coins[7][1] - y)
        d8 = np.abs(coins[8][0] - x) + np.abs(coins[8][1] - y)
        d9 = np.abs(coins[9][0] - x) + np.abs(coins[9][1] - y)
    
        states = np.array([1,f1,f2,d1,d2,d3,d4,d5,d6,d7,d8,d9])

        act = np.argmax( (Q.transpose()).dot(states) )
        action_ideas=[]
        if act == 0: action_ideas.append('UP')
        if act == 1: action_ideas.append('DOWN')
        if act == 2: action_ideas.append('LEFT')
        if act == 3: action_ideas.append('RIGHT')
        if act == 4: action_ideas.append('WAIT')        
        self.next_action = action_ideas.pop()
    self.logger.info('Pick action at random')
    
def reward_update(self):   
    Q = np.loadtxt("Q.txt")
    arena = self.game_state['arena']
    coins = self.game_state['coins']
    x, y, _, bombs_left, score = self.game_state['self']
    state = statefun(arena,x,y,coins)
    index = 4
    next_state=np.array([x,y])
    if self.events[0] == e.MOVED_LEFT: 
        next_state = np.array([x-1,y])
    if self.events[0] == e.MOVED_RIGHT: 
        next_state =  np.array([x+1,y])
    if self.events[0] == e.MOVED_UP: 
        next_state = np.array([x,y+1])
    if self.events[0] == e.MOVED_DOWN: 
        next_state = np.array([x,y-1])
    if self.events[0] == e.WAITED: 
        next_state= np.array([x,y])
    if len(self.events)>1:
        if self.events[1] == e.COIN_COLLECTED:
            reward=100
            next_coins = coins[:-1]            
    else:
            reward=-1
            next_coins = coins
            
    index = 0
    if self.next_action == 'UP':
        index = 0
    if self.next_action == 'DOWN':
        index = 1
    if self.next_action == 'LEFT':
        index = 2
    if self.next_action == 'RIGHT':
        index = 3
    if self.next_action == 'WAIT':
        index = 4
    Next_state = statefun(arena,next_state[0],next_state[1],next_coins)
    Q[state][index]=(1-alpha)*Q[state][index]+alpha*(reward+ np.argmax(Q[Next_state]))
    
    np.savetxt("Q.txt", Q)    
    
    
    
def end_of_episode(self):  
    Q = np.loadtxt("Q.txt")
    arena = self.game_state['arena']
    coins = self.game_state['coins']
    alpha = 1
    gamma = 0.9    
    if op.isfile("coins_algo.txt") != True:
        collected = 9-len(coins)
        np.savetxt("coins_algo.txt",collected)
    else:
        collected = np.loadtxt("coins_algo.txt")
        collected = np.append(collected,9-len(coins))
        np.savetxt("coins_algo.txt",collected)
   # for state in accessible:
   #     if self.next_action == 'UP': next_state = (x,y+1)
   #     if self.next_action == 'DOWN': next_state =  (x,y-1)
   #     if self.next_action == 'LEFT': next_state =  (x-1,y)
   #     if self.next_action == 'RIGHT': next_state = (x+1,y)
   #     if self.next_action == 'WAIT': next_state = (x,y)
   #     max_Q_next = np.argmax(Q[next_state])    
   # Q[state][self.next_action] = Q[state][self.next_action] + alpha * (reward[state][self.next_action] + gamma * max_Q_next - Q[state][self.next_action])  
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')