import random

import numpy as np

from .agent import AgentDiscrete


class AgentMonteCarloV(AgentDiscrete):

    """Implementation of an agent that learns V values for a random policy.
    """

    def __init__(self, num_states, num_actions):

        # I have kept this compatible with both Python 2 and 3
        AgentDiscrete.__init__(self, num_states, num_actions)

        # Structure for storing the state values
        self._V = np.zeros((num_states,))  # Initialize values to 0

        # Later on for state/action values (Q), you will need something like:
        self._Q = np.zeros((num_states,num_actions,)) # Initialize Q-values

        # ANYTHING TO CODE HERE?
        # Any other things the agent needs to store to compute values?
        self.results = []
        for i in range(num_states):
            self.results.append([])
        self.observations = []
        self.rewards = []
        self.alpha = 1

    def newEpisode(self):
        """ Inform the agent that a new episode has started. """
        # ANYTHING TO CODE HERE?
        R = 0
        for o, r in reversed(zip(self.observations, self.rewards)):
            R += r
            self.results[o].append(R)

        for i, r_list in enumerate(self.results):
            if r_list:
                self._V[i] += self.alpha*(np.mean(r_list)-self._V[i])
        self.observations = []
        self.rewards = []

    def integrateObservation(self, obs):
        """ Integrate the current observation of the environment.
        Args:
            obs (int) : the observation the agent has made. This observation
            may be equivalent to the state of the environment.
        """
        # ANYTHING TO CODE HERE?
        self.observations.append(obs)

    def getAction(self):
        """ Return a chosen action.
        Returns:
            an action (int)
        """
        # Return a random action
        return random.randint(0, self._num_actions - 1)

    def giveReward(self, reward):
        """ Reward or punish the agent.
        Args:
            reward (double): reward if r is positive, punishment otherwise
        """
        # ANYTHING TO CODE HERE?
        self.rewards.append(reward)

    def getValues(self):
        """Get the values the agent assigns to each state.

        If the agent does not compute values, this should return None.

        Returns:
            The values (numpy array of floats), one for each state.
        """
        return self._V

    def getQValues(self):
        """Get the Q-values the agent assigns to each state/action pair.

        If the agent does not compute values, this should return None.

        Returns:
            The Q-values (numpy 2D array of floats),
                one for each state/action pair.
        """
        # This agent doesn't store Q-values, so return None
        # If it does store Q-values, this should become
        # return self._Q
        return None
