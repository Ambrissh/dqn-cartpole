import gymnasium as gym
import matplotlib.pyplot as plt 

batch_size=128 #batch size for the transitions
gamma=0.99 # discount factor
eps_start=0.9 #epsilon for exploration from the epsilon greedy policy
eps_end=0.01 #epsilon ends
eps_decay=2500 # the rate at which epsilon decays
tau=0.005 # the learning rate for the target network
lr=3e-4 # the learning rate for AdamW optimizer


n_actions=env.action_space.n #getting the number of actions
state,info=env.reset()
n_observations=len(state) #getting the number of observables

policy_net=DQN(n_observations,n_actions).to(device)
target_net=DQN(n_observations,n_actions).to(device)

# if we have 2 of the same networks , it would be like chasing a moving target , hence for stability we use the following line of code
target_net.load_state_dict(policy_net.state_dict())

optimizer=optim.AdamW(policy_net.parameters(),lr=lr,amsgrad=True) #choosing the optimizers with amsgrad for better stability
memory=ReplayBuffer(10000)

steps_done=0

def select_action(state):
  global steps_done
  sample=random.random()
  eps_threshold=eps_end+(eps_start-eps_end) * \
  math.exp(-1*steps_done/eps_decay)
  steps_done+=1
  if sample>eps_threshold: #just the epsilon greedy criterion , in this case if we explore with probability epsilon and exploit with probability 1-episilon
    with torch.no_grad()
    return policy_net(state).max(1).indices.view(1,1)

  else:
    return torch.tensor ([[env.action_space.sample()]], device=device , dtype=torch.long)

episode_durations=[]

def plot_durations(show_result=False):
  plt.figure(1)
  durations_t=torch.tensor(episode_durations , dtype=torch.long)
  if show_result:
    plt.title("Result")
  else:
    plt.clf()
    plt.title("Training..")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())

    #taking the average of the last 100 episodes
    if len(durations_t)>=100:
      means=durations_t.unfold(0,100,1).mean(1).view(-1)
      means=torch.cat((torch.zeros(99),means))
      plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
      if not show_result:
        display.display(plt.gcf())
        display.clear_output(wait=True)
      else:
        display.display(plt.gcf())
