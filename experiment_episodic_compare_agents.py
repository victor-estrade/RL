from __future__ import division, print_function

import random
import numpy as np
import matplotlib.pyplot as plt

import environments
import agents
from experiment_episodic import doEpisodes


def plotLearningCurves(learning_curves, ax):
    ''' Plot several learning curves.

    Args:
        learning curves - a dictionary, where the key is the name of the agent,
                          and the value are lists or arrays of returns per
                          episode.
        ax - a matplotlib axis on which to plot
    '''
    for agent_name, learning_curve in learning_curves.items():
        ax.plot(learning_curve, label='agent: ' + agent_name)
    ax.set_xlabel('number of episodes')
    ax.set_ylabel('return')
    ax.legend(loc=2, bbox_to_anchor=(1., 1.))


def doExperiments(env, agents, n_learning_sessions,
                  n_episodes, max_actions_per_episode, verbose=True):
    """ Perform several experiments with different agents.

    Args:
        env - an environment of type Environment
        agents - a dictionary where the keys are the names of agents, and the
                 values are agents of type Agent
        n_episodes - number of episodes to conduct
        max_actions_per_episode - maximum number of actions per episode
        verbose - whether to print intermediate results

    Returns:
        the mean learning curves for each experiments, it is a dictionary,
            where the key is the name of the agent,
            and the value are lists or arrays of returns per episode
            (averaged over n_learning_sessions).

    """
    learning_curves = {}
    for agent_name, agent in agents.items():

        curves_for_this_agents = np.zeros((n_learning_sessions, n_episodes,))
        for i_learning_session in range(n_learning_sessions):

            print('env=' + env_name + '   agent=' + agent_name +
                  '   learning session=' + str(i_learning_session + 1))

            # agent.reset()

            all_rewards = doEpisodes(
                env, agent, n_episodes, max_actions, verbose)

            # Get the return for each episode, and put it in learning curve
            lc = [sum(rewards_per_epi) for rewards_per_epi in all_rewards]

            curves_for_this_agents[i_learning_session, :] = lc

        mean_curves = np.mean(curves_for_this_agents, axis=0)

        learning_curves[agent_name] = mean_curves

    return learning_curves


if __name__ == '__main__':

    import sys
    if (len(sys.argv) < 2):
        env_name = 'Maze'
    else:
        env_name = sys.argv[1]

    # Initialize an environment
    from get_environment_from_name import get_environment_from_name
    env = get_environment_from_name(env_name)
    n_states = env.numStates()
    n_actions = env.numActions()

    # Make a dictionary, and fill it with agents
    agents_ = {}

    # from agents.agent_random import AgentRandom
    # agents_['Random'] = AgentRandom(n_states, n_actions)

    # from agents.agent_first_action import AgentFirstAction
    # agents_['FirstAction'] = AgentFirstAction(n_states, n_actions)

    from agents.agent_montecarlo_statevalues import AgentMonteCarloV
    agents_['MonteCarlo()'] = AgentMonteCarloV(n_states, n_actions)
    # for alpha in np.linspace(0.01, 0.3, num=3):
    #     for beta in np.linspace(0.9, 0.99, num=3):
    #         name = 'MonteCarlo(a={:0.2f},B={:0.2f})'.format(alpha, beta)
    #         agents_[name] = AgentMonteCarloV(n_states, n_actions,
    #                                          alpha=alpha, beta=beta)

    # from pybrain.rl.learners.valuebased import ActionValueTable
    # from pybrain.rl.agents import LearningAgent
    # from pybrain.rl.learners import Q, SARSA  # @UnusedImport

    # controller = ActionValueTable(n_states, n_actions)
    # controller.initialize(1.)
    # learner = Q()
    # agent = LearningAgent(controller, learner)
    # agents_['Pybrain'] = agent

    # LEARNING TESTS
    n_learning_sessions = 2
    n_episodes = 100
    max_actions = 1000
    verbose = False

    learning_curves = doExperiments(
        env, agents_, n_learning_sessions, n_episodes, max_actions, verbose)

    fig = plt.figure(1, figsize=(12, 6))
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])
    plotLearningCurves(learning_curves, ax)
    plt.show()

    # print('_____ Values _____')
    # print(agents_['MonteCarlo']._V)

    # print('_____ Q(s,a) _____')
    # print(agents_['MonteCarlo']._Q)
