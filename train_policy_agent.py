#import libraries
import gym
import random
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from dqn_agent import Agent
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, s_size=37, h_size=100, a_size=4):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)





def reinforce(env, policy, n_episodes=2000, max_t=1000, gamma=0.9, print_every=1):
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    brain_name = env.brain_names[0]
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        saved_log_probs = []
        rewards = []
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0] 
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]
            
            # state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break 
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)])
        
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        # if i_episode % print_every == 0:
        print('\rEpisode {}\tLatest Score: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, scores[-1], np.mean(scores_deque)), end="")
        # if np.mean(scores_deque)>=195.0:
            # print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            # break
        
    return policy, scores

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def main():
    env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")
    policy = Policy().to(device)

    # Train
    start_time = time.time() # Monitor Training Time  
    policy, scores = reinforce(env, policy, n_episodes=200, max_t=1000, gamma=1.0, print_every=100)
    print("\nTotal Training time = {:.1f} min".format((time.time()-start_time)/60))
    plot_rewards(scores)
   
def plot_rewards(scores):
    ma = moving_average(np.array(scores), 50)
    fig = plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(ma)), ma)
    plt.title('Score (Rewards)')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.grid(True)  
    plt.show()



if __name__ == "__main__":
    main()