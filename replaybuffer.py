from collections import deque 
import random 

class ReplayBuffer(object):
  def __init__(self, capacity): 
    self.buffer=deque([],maxlen=capacity) # creating an empty list 

  def push(self,state,action,reward,next_state,done): 
    self.buffer.append((state,action,reward,next_state,done)) #taking a new step and storing it 
  
  def sample(self,batch_size): 
    return random.sample(self.buffer,batch_size) #a random sample from all the episodes. 

  def __len__(self): 
    return len(self.buffer)   #the length of number of memories 

