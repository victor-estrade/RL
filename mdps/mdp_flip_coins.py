import numpy as np

from .markov_decision_process import MarkovDecisionProcess
                                                     
class MDPFlipCoin(MarkovDecisionProcess):
    """ Implementation of an MDP for flipping a coin (from the lectures)."""
    
    def __init__(self):
        name = "FlipCoin"
        
        S = ['INIT','HEADS','TAILS'] # State space 
        A = ['COIN1','COIN2'] # Action space
        n_states = len(S)
        n_actions = len(A)

        # Transition function S x A x S -> probability
        P = np.zeros((n_states,n_actions,n_states,))
        P[0,0,1] = 0.5 # INIT X COIN1 X HEADS
        P[0,0,2] = 0.5 # INIT X COIN1 X TAILS
        P[0,1,1] = 0.4 # INIT X COIN2 X HEADS
        P[0,1,2] = 0.6 # INIT X COIN2 X TAILS

        # Terminal states (not probabilities, but true/false)
        T = [False, True, True] # Both HEADS and TAILS are terminal states 
        
        # Initial state distribution: always start in INIT
        I = [1, 0, 0] 
  
        # Reward function
        R = np.zeros((n_states,n_states,)) 
        R[0,1] =   0 # Zero reward for HEADS
        R[0,2] = 100 # 100 reward for HEADS
        # Other transititions are not possible, no need to specifiy rewards.

        discount=1.0

        # Create the MDP by calling __init__ in the base class
        # MarkovDecisionProcess
        # I have kept this compatible with both Python 2 and 3
        MarkovDecisionProcess.__init__(self,S,T,I,A,P,R,discount,name)


    def stateString(self,cur_state):
        """See documentation in the base class."""
        if isinstance(cur_state,int):
            cur_state = self.S[cur_state] 
        return str(cur_state)


    def valuesString(self,values):
        """See documentation in the base class."""
        return str(values)


    def policyStringMode(self,policy):
        """See documentation in the base class."""
        return self.policyString(policy)


    def policyString(self,policy):
        """See documentation in the base class."""
        string = 'P(INIT,COIN1)='
        string += str(policy[0,0])
        string += ',    P(INIT,COIN2)='
        string += str(policy[0,1])
        return string

                                    
class MDPFlipTwoCoins(MarkovDecisionProcess):
    """ Implementation of an MDP for flipping two coins (from the lectures)."""
    
    def __init__(self):
        name = "FlipTwoCoins"

        S = ['INIT','HEADS_1','TAILS_1','HEADS_2','TAILS_2'] # State space 
        A = ['COIN_A','COIN_B'] # Action space
        n_states = len(S)
        n_actions = len(A)

        # Transition function S x A x S -> probability
        P = np.zeros((n_states,n_actions,n_states,))
        
        P[0,0,1] = 0.5 # INIT X COIN_A X HEADS_1
        P[0,0,2] = 0.5 # INIT X COIN_A X TAILS_1
        P[0,1,1] = 0.4 # INIT X COIN_B X HEADS_1
        P[0,1,2] = 0.6 # INIT X COIN_B X TAILS_1
        P[1:2,0,3] = 0.5 # HEADS_1/COINS_1 X COIN_A X HEADS_2
        P[1:2,0,4] = 0.5 # HEADS_1/COINS_1 X COIN_A X TAILS_2
        P[1:2,1,3] = 0.4 # HEADS_1/COINS_1 X COIN_B X HEADS_2
        P[1:2,1,4] = 0.6 # HEADS_1/COINS_1 X COIN_B X TAILS_2

        # Terminal states (not probabilities, but true/false)
        T = [False, False, False, True, True] # Both HEADS_2 and TAILS_2 are terminal states 
        
        # Initial state distribution: always start in INIT
        I = [1, 0, 0, 0, 0] 
  
        # Reward function
        R = np.zeros((n_states,n_states,)) 
        R[0,2] = 100 # 100 reward for HEADS on first time
        R[0,4] = 100 # 100 reward for HEADS on first time
        # Other transititions are not possible, no need to specifiy rewards.

        discount=1.0

        # Create the MDP by calling __init__ in the base class
        # MarkovDecisionProcess
        # I have kept this compatible with both Python 2 and 3
        MarkovDecisionProcess.__init__(self,S,T,I,A,P,R,discount,name)


    def stateString(self,cur_state):
        """See documentation in the base class."""
        if isinstance(cur_state,int):
            cur_state = self.S[cur_state] 
        return str(cur_state)


    def valuesString(self,values):
        """See documentation in the base class."""
        return str(values)


    def policyStringMode(self,policy):
        """See documentation in the base class."""
        return self.policyString(policy)
        

    def policyString(self,policy):
        """See documentation in the base class."""
        string = str(policy)
        return string
