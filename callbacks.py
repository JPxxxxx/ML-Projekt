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
        Q = np.zeros((4,4),dtype = float)
        np.savetxt("Q.txt", Q)
    self.pick_history = deque([], 400)
    self.states_history = deque([], 400)
    self.index_history = deque([], 400)
    self.logger.info('Initialize')
    
def act(self):    
    Q = np.loadtxt("Q.txt")
    arena = self.game_state['arena']
    coins = self.game_state['coins']
    x, y, _, bombs_left, score = self.game_state['self']

    epsilon = 0
   
    best_dist = 1000000        
    for i in range(len(coins)):
        dist = (np.abs(coins[i][0] - x) + np.abs(coins[i][1] - y))
        if dist < best_dist:
            dist = best_dist
            nearest_coin = i
    dist_x_left = 1/np.abs(coins[nearest_coin][0] - (x-1) )
    dist_x_right = 1/np.abs(coins[nearest_coin][0] - (x+1) )
    dist_y_up = 1/np.abs(coins[nearest_coin][1] - (y-1) )
    dist_y_down = 1/np.abs(coins[nearest_coin][1] - (y+1) )
    
    states = np.array([dist_x_left, dist_x_right, dist_y_up, dist_y_down])
           
    if np.random.rand(1) <= epsilon:
        action_ideas = ['UP','DOWN','RIGHT','LEFT']
        shuffle(action_ideas) 
        self.next_action = action_ideas.pop()
    else:
        action_ideas = ['UP','DOWN','RIGHT','LEFT']
        shuffle(action_ideas) 
            
        act = np.argmax( (Q.transpose()).dot(states) )
        if act == 0: action_ideas.append('UP')
        if act == 1: action_ideas.append('DOWN')
        if act == 2: action_ideas.append('LEFT')
        if act == 3: action_ideas.append('RIGHT')
        #if act == 4: action_ideas.append('WAIT')        
        self.next_action = action_ideas.pop()
    self.states_history.append(states)
    self.pick_history.append(self.next_action)
    self.logger.info('Pick action at random')
    
def reward_update(self):
    Q = np.loadtxt("Q.txt")
    arena = self.game_state['arena']
    coins = self.game_state['coins']
    x, y, _, bombs_left, score = self.game_state['self']
    
    # reward matrix
    if (op.isfile("rewards.txt") == False):
        rewards = np.zeros((4,4),dtype = float)
        np.savetxt("rewards.txt", rewards)
    else:
        rewards = np.loadtxt("rewards.txt")
        
    best_dist = 1000000        
    for i in range(len(coins)):
        dist = (np.abs(coins[i][0] - x) + np.abs(coins[i][1] - y))
        if dist < best_dist:
            dist = best_dist
            nearest_coin = i
    dist_x_left = np.abs(coins[nearest_coin][0] - (x-1) )
    dist_x_right = np.abs(coins[nearest_coin][0] - (x+1) )
    dist_y_up = np.abs(coins[nearest_coin][1] - (y-1) )
    dist_y_down = np.abs(coins[nearest_coin][1] - (y+1) )
    
    states = np.array([dist_x_left, dist_x_right, dist_y_up, dist_y_down])
    
    if self.events[0] == e.MOVED_LEFT: 
        reward = -0.4
    if self.events[0] == e.MOVED_RIGHT: 
        reward = -0.4
    if self.events[0] == e.MOVED_UP: 
        reward = -0.4
    if self.events[0] == e.MOVED_DOWN: 
        reward = -0.4
    #if self.events[0] == e.WAITED: 
    #    reward = -2
    if len(self.events)>1:
        if self.events[1] == e.COIN_COLLECTED:
            reward = 10
    else:
        reward = -1

            
    index = 0
    if self.next_action == 'UP':
        index = 0
    if self.next_action == 'DOWN':
        index = 1
    if self.next_action == 'LEFT':
        index = 2
    if self.next_action == 'RIGHT':
        index = 3
    #if self.next_action == 'WAIT':
    #    index = 4
    
    self.index_history.append(index)
    
    for i in range(4):
        rewards[i][index] = rewards[i][index] + reward - ((Q.transpose()).dot(states))[index]

    np.savetxt("rewards.txt", rewards)    
    
    
def end_of_episode(self):  
    Q = np.loadtxt("Q.txt")
    rewards = np.loadtxt("rewards.txt")
    
    alpha = 0.01
    beta = 0.005
    
    for k in range(len(self.states_history)):
        state = self.states_history.popleft()
        if len(self.states_history) != 0:
            next_state = self.states_history[0]
            index = self.index_history.popleft()
            for i in range(4):
                Q[i][index] = 0.1 * Q[i][index] + alpha * ( 0.1 * rewards[i][index] + beta * np.argmax((Q.transpose()).dot(next_state))) * state[i] * 0.1
    
    
    rewards = np.zeros((4,4),dtype = float)
    np.savetxt("Q.txt", Q)
    np.savetxt("rewards.txt", rewards)
    
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')
    
    