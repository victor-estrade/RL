import random
import matplotlib.pyplot as plt


import environments
import agents

def doEpisodes(env,agent,n_episodes,max_actions_per_episode,verbose=True):
    """ Perform several episodes with an enviroment and an agent.
    
    Args:
        env - an environment of type Environment
        agent - an agent of type Agent
        n_episodes - number of episodes to conduct
        max_actions_per_episode - maximum number of actions per episode
        verbose - whether to print intermediate results 
    
    Returns:
        all the rewards gathered during the episodes. It is a list of lists of
        size n_episodes X n_actions_per_episode
    
    Very similar to EpisodicExperiment.doEpisodes in PyBrain
    """
    all_rewards = []

    for episode in range(n_episodes):
        agent.newEpisode()
        rewards = []
        stepid = 0
        env.reset()
        
        if verbose:
            print('____________________\nINIT\n'+env.stateString()+'\n')

        while (not env.isFinished()) and stepid<=max_actions_per_episode:
            stepid += 1
            
            observation = env.getObservation()
            agent.integrateObservation(observation)
            
            action = agent.getAction()
            env.performAction(action)
            
            reward = env.getReward()
            agent.giveReward(reward)
            
            rewards.append(reward)
    
            if verbose:
                print('  action = '+env.actionString(action))
                print('  reward = '+str(reward))
                print(env.stateString()+'\n')
    
        
        if verbose:
            print('return = '+str(sum(rewards)))
            
        all_rewards.append(rewards)
        
    return all_rewards
        
def plotLearningCurve(learning_curve,ax):
    ''' Plot a learning curve.
    
    Args: 
        learning curve - a list or array of returns per episode
        ax - a matplotlib axis on which to plot 
    '''
    ax.plot(learning_curve)
    ax.set_xlabel('number of episodes')
    ax.set_ylabel('return')




if __name__ == '__main__':
    
    import sys
    if (len(sys.argv)<2):
        env_name = 'Maze'
    else:
        env_name = sys.argv[1]
        
    # Initialize the an environment
    from get_environment_from_name import get_environment_from_name
    env = get_environment_from_name(env_name)
    
    # Initialize the an agent
    from agents.agent_random import AgentRandom
    agent = AgentRandom(env.numStates(), env.numActions())
        
    n_episodes = 100
    max_actions_per_episode = 1000
    verbose = False
    
    all_rewards = doEpisodes(env,agent,n_episodes,max_actions_per_episode,verbose)
    
    # Get the return for each episode, and put it in learning curve
    learning_curve = [sum(rewards_per_epi) for rewards_per_epi in all_rewards ]
    fig = plt.figure(1,figsize=(12, 6))
    ax = fig.add_subplot(1,1,1)
    plotLearningCurve(learning_curve,ax)
    plt.show()

