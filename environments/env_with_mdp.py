import numpy as np
import random

from .environment import Environment
 

class EnvWithMDP(Environment):
    
    def __init__(self,mdp):
        """Initializes a new environment from a Markov Decision Process.
        
        Here, the environment is simply a wrapper around the MDP, to avoid
        model-free agents from accessing the mdp.
        
        Args:
            mdp : a Markov Decision Process
        """
        self._mdp = mdp
        self._prev_state = None # What was the previous state?
        self._cur_state = None  # What was is the current state?
        self.reset()


    def numActions(self):
        """ See documentation in base class."""
        return len(self._mdp.A)


    def numStates(self):
        """ See documentation in base class."""
        return len(self._mdp.S)


    def isFinished(self):
        """ See documentation in base class."""
        return self._mdp.isTerminalState(self._cur_state)


    def _sample_state(self,probabilities):
        # probabilities is "probability mass function", e.g. [0 0 0.8 0.2 0 0]
        
        # Cumulative distribution function,         e.g. [0 0 0.8 1.0 1.0 1.0]
        probabilities_cumul = np.cumsum(probabilities)
        
        # Find first value large than randomly (uniform) sampled value
        rand = random.random()
        for s in range(len(self._mdp.S)):
            if probabilities_cumul[s]>=rand:
                return  s
                
    
    def reset(self):
        """ See documentation in base class."""
        # Sample random state from initial state distribution
        self._cur_state = self._sample_state(self._mdp.I)
        self._prev_state = self._cur_state


    def performAction(self,cur_action):
        """ See documentation in base class."""
        
        # Convert to int if necessary
        if isinstance(cur_action, str):
            cur_action = self._mdp.A.index(cur_action)
        
        # Probability of going to next states, given current state and
        # action.
        P_s = self._mdp.P[self._cur_state,cur_action,:]
  
        # Sample random state from transition function
        new_state = self._sample_state(P_s)
        
        # The current becomes the old, and the new becomes the current.
        self._prev_state = self._cur_state
        self._cur_state = new_state
        
        
    def getReward(self):
        """ See documentation in base class."""
        return self._mdp.R[self._prev_state,self._cur_state]
    
    def getObservation(self):
        """ See documentation in base class."""
        return self._cur_state
        
    def __str__(self):
        return self.stateString()
        
    def stateString(self):
        """ See documentation in base class."""
        return self._mdp.stateString(self._cur_state);
        
    def actionString(self,action):
        """ See documentation in base class."""
        return str(self._mdp.A[action])

