import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy import stats

class Environment:
    def __init__(self, n):
        self.steps = n
    
    def reset(self):
        self.current_state = 1
        self.initial_state = self.current_state
        
        return self.initial_state
    
    def step(self, action):
        
        if action.argmax():
            self.current_state += 1
            if self.current_state == self.steps:
                reward = 1
                return self.current_state, reward, np.zeros((1))
            else:
                return self.current_state, 0, np.ones((1))
        else:
            self.current_state = self.initial_state
            reward = 0
            return self.current_state, reward, np.zeros((1))
        
env = Environment(3)        

