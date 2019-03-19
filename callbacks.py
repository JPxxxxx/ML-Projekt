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


def setup(self):
    np.random.seed()    
    # Q matrix
    if op.isfile("Q.txt") != True:        
        Q = np.zeros((12,5),dtype = float)
        np.savetxt("Q.txt", Q)
    self.coordinate_history = deque([], 400)
    self.pick_history = deque([], 400)
    self.states_history = deque([], 400)
    self.index_history = deque([], 400)
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
        states = np.array([f2,f1,1])
        for i in range(len(coins)):
            states = np.append(states,(np.abs(coins[i][0] - x) + np.abs(coins[i][1] - y)))
        if len(states) < 12:
            for i in range(12 - len(states)):
                states = np.append(states , 0 )
            
        act = np.argmax( (Q.transpose()).dot(states) )
        if act == 0: action_ideas.append('UP')
        if act == 1: action_ideas.append('DOWN')
        if act == 2: action_ideas.append('LEFT')
        if act == 3: action_ideas.append('RIGHT')
        if act == 4: action_ideas.append('WAIT')        
        self.next_action = action_ideas.pop()
    self.states_history.append(states)
    self.pick_history.append(self.next_action)
    self.logger.info('Pick action at random')
    
def reward_update(self):   
    arena = self.game_state['arena']
    coins = self.game_state['coins']
    x, y, _, bombs_left, score = self.game_state['self']
    
    # reward matrix
    if (op.isfile("rewards.txt") == False):
        rewards = np.zeros((12,5),dtype = float)
        np.savetxt("rewards.txt", rewards)
    
    f1 = x
    f2 = y
    states = np.array([f2,f1,1])
    for i in range(len(coins)):
        states = np.append(states,(np.abs(coins[i][0] - x) + np.abs(coins[i][1] - y)))
    if len(states) < 12:
        for i in range(12 - len(states)):
            states = np.append(states , 0 )
    
    if self.events[0] == e.MOVED_LEFT: 
        next_state = np.array([x-1,y])
        reward = -1
    if self.events[0] == e.MOVED_RIGHT: 
        next_state =  np.array([x+1,y])
        reward = -1
    if self.events[0] == e.MOVED_UP: 
        next_state = np.array([x,y+1])
        reward = -1
    if self.events[0] == e.MOVED_DOWN: 
        next_state = np.array([x,y-1])
        reward = -1
    if self.events[0] == e.WAITED: 
        next_state= np.array([x,y])
        reward = -1
    if len(self.events)>1:
        if self.events[1] == e.COIN_COLLECTED:
            reward = 100
    else:
        reward = -10

            
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
    
    self.index_history.append(index)
    
    alpha = 0.3
    for i in range(12):
        rewards[i][index] = reward - ((Q.transpose()).dot(states))[index]

    np.savetxt("rewards.txt", rewards)    
    
    
def end_of_episode(self):  
    Q = np.loadtxt("Q.txt")
    rewards = np.loadtxt("rewards.txt")
    
    alpha = 0.3
    beta = 0.2
    
    for k in range(len(self.coordinate_history)):
        state = self.states_history.popleft()
        next_state = self.states_history[0]
        index = self.index_history.popleft()
        for i in range(12):
            Q[i][index] = Q[i][index] + alpha * (rewards[i][index] + beta * np.argmax((Q.transpose()).dot(next_state))) * state[i]
    
    np.savetxt("Q.txt", Q)
    
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')
    
    