from unityagents import UnityEnvironment
import numpy as np

def test_env():
    try:
        env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]

        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]

        # Action space
        action_size = brain.vector_action_space_size
        print('Number of actions:', action_size)

        # State space
        state = env_info.vector_observations[0]
        print('States look like:', state)
        state_size = len(state)
        print('States have length:', state_size)

        env_info = env.reset(train_mode=False)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0                                          # initialize the score
        while True:
            action = np.random.randint(action_size)        # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                break
            
        print("Score: {}".format(score))
        env.close()

    except:
        print('There is an error with the installation of UnityEnvironment!')

if __name__ == "__main__":
    test_env()