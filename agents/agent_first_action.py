from .agent import AgentDiscrete


class AgentFirstAction(AgentDiscrete):

    def __init__(self, num_states, num_actions):
        """See documentation in the base class."""
        # I have kept this compatible with both Python 2 and 3
        AgentDiscrete.__init__(self, num_states, num_actions)

    def newEpisode(self):
        """See documentation in the base class."""
        # This agent doesn't care when an episode is over, so this
        # function does nothing.
        pass

    def integrateObservation(self, obs):
        """See documentation in the base class."""
        # This agent simply ignores any observation it makes
        pass

    def getAction(self):
        """ This agent always returns the action "0"
        Returns:
            an action (int)
        """
        # Return first (zero) action
        return 0

    def giveReward(self, reward):
        """See documentation in the base class."""
        # This agent ignores all rewards it gets
        pass
