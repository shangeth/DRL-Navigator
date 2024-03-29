#import libraries
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from dqn_agent import Agent
import time


def dqn(env, agent, n_episodes=3000, max_t=3000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]            # get the current state

        score = 0
        for t in range(max_t): 
            #action = np.random.randint(action_size)        # select an action
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tLatest Score: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, score, np.mean(scores_window)), end="")
    torch.save(agent.qnetwork_local.state_dict(), 'dqn_checkpoint2.pth')
    return scores

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def main():
    env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # DQN Agent
    agent = Agent(state_size=37, action_size=4, seed=42)

    # Train
    start_time = time.time() # Monitor Training Time  
    scores = dqn(env, agent, n_episodes=200, max_t=5000, eps_start=1.0, eps_end=0.01, eps_decay=0.995)
    print("\nTotal Training time = {:.1f} min".format((time.time()-start_time)/60))
    plot_rewards(scores)
   
def plot_rewards(scores):
    ma = moving_average(np.array(scores), 100)
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