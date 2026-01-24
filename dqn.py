import gymnasium as gym 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import math 
import random
import matplotlib.pyplot as plt 
import matplotlib 
from collections import namedtuple , deque 
from itertools import count 

#creating the environment 
env=gym.make('CartPole-v1')

#setting up matplotlib 
is_ipython='inline' in matplotlib.get_backend()
if is_ipython:
  from IPython import display 
  plt.ion()

#setting up gpu 

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#for reproducibility 
seed=42 
random.seed(seed)
torch.manual_seed(seed)
env.reset(seed=seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)
if torch.cuda.is_available():
  torch.cuda.manual_seed(seed)
  
%%writefile dqn.py 
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
  """Creating the model ,
  the main principle here is we are trying to approximate the Q-values , then 
during training , we observe the deviation from the original Q-values , found out
via the Bellman - Equation"""
    

  def __init__(self,n_observations,n_actions):
    super(DQN,self).__init__()
    self.layer1=torch.nn.Linear(n_observations,128) # first layer , mapping the observations to 128 neurons
    self.layer2=torch.nn.Linear(128,128) #layer 2 
    self.layer3=torch.nn.Linear(128,n_actions) # final layer mapping the outputs to the actions 


  def forward(self,x): 
    x=F.relu(self.layer1(x))  #adding activations for non-linearity 
    x=F.relu(self.layer2(x))
    x=self.layer3(x)

    return x  






