class AgentDiscrete:

    """ The interface an agent should conform to."""

    def __init__(self, num_states, num_actions):
        """Initializes a new agent.
        Args:
            num_states (int): Number of possible states the agent can make
            num_actions (int): Number of possible action the agent can perform
        """
        self._num_states = num_states
        self._num_actions = num_actions

    def newEpisode(self):
        """ Inform the agent that a new episode has started. """
        pass

    def integrateObservation(self, obs):
        """ Integrate the current observation of the environment.
        Args:
            obs (int) : the observation the agent has made. This observation
            may be equivalent to the state of the environment.
        """
        pass

    def getAction(self):
        """ Return a chosen action.
        Returns:
            an action (int)
        """
        raise NotImplementedError('subclasses must override getAction()!')

    def giveReward(self, reward):
        """ Reward or punish the agent.
        Args:
            reward (double): reward if r is positive, punishment otherwise
        """
        pass

    def getValues(self):
        """Get the values the agent assigns to each state.

        If the agent does not compute values, this should return None.

        Returns:
            The values (numpy array of floats), one for each state.
        """
        return None

    def getQValues(self):
        """Get the Q-values the agent assigns to each state/action pair.

        If the agent does not compute values, this should return None.

        Returns:
            The Q-values (numpy 2D array of floats),
                one for each state/action pair.
        """
        return None
