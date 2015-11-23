class Environment:
    """The interface an environment should conform to."""

    def __init__(self):
        """Initializes a new environment.
        
        Call the script with "--help" to see what these arguments are for a particular environment.  
        """
        pass
    
    def numActions(self):
        """Get the number of actions that the agent can execute in this environment.
        
        Returns: 
            The number of possible actions
        """
        raise NotImplementedError('subclasses must override numActions()!')
    
    def numStates(self):
        """Get number of possible states the agent can make in this environment.
        
        Returns: 
            The number of possible states
        """
        raise NotImplementedError('subclasses must override numStates()!')
    
    def isFinished(self):
        """Determine whether the environment is in a terminal state."""
        raise NotImplementedError('subclasses must override isFinished()!')
        
    def reset(self):
        """Sets an initial state."""
        raise NotImplementedError('subclasses must override reset()!')
    
    def performAction(self, action):
        """Project environment one step into the future.
         Args:
             action (int): The action the agent is performing
        """
        raise NotImplementedError('subclasses must override performAction()!')
    
    def getObservation(self):
        """Observe the state of the environment. 
        Returns:
            The state of the environment, or an observation thereof (int)
        """
        raise NotImplementedError('subclasses must override getObservation()!')
        
    def getReward(self):
        """ Compute and return the current reward (i.e. corresponding to the last action performed) """
        raise NotImplementedError('subclasses must override getReward()!')
        
    def stateString(self):
        """Print the current state as a string
        """
        raise NotImplementedError('subclasses must override stateString()!')

    def actionString(self,action):
        """Print the action as a string
        Args:
            action (int): The action to be printed
        """
        return str(action)

