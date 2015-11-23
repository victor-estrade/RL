import numpy as np
import random


class MarkovDecisionProcess:

    def __init__(self, S, T, I, A, P, R, discount, name='no name'):
        ''' Initialize a Markov Decision Process.

        Args:
            S : State space
            A : Action space
            P : Transition function,       P: S X A X S => probability
            R : Reward function,           R: S X S => reward
            T : Terminal states            T: S => boolean
            I : Initial state distribution I: S => probability
        '''
        self.S = S
        self.A = A
        self.P = P
        self.R = R
        self.T = T
        self.I = I
        self.discount = discount
        self.name = name

    def isTerminalState(self, state):
        """Determine whether a state is terminal or not.

        Args:
            state (int): The state whose 'terminality' is to be determined
        Returns:
            True is the state is terminal, False otherwise
        """
        return self.T[state]

    def __str__(self):
        """ Return a string representation of the MDP."""
        string = 'MarkovDecisionProcess "' + self.name + '"\n'
        string += '  S  (state space)                = ' + str(self.S) + '\n'
        string += '  T  (terminal states)            = ' + str(self.T) + '\n'
        string += '  I  (initial state distribution) = ' + str(self.I) + '\n'
        string += '  A  (action space)               = ' + str(self.A) + '\n'
        for a in range(len(self.A)):
            string += '  P_' + str(a) + '=\n'
            string += '' + str(self.P[:, a, :])
            string += '\n'
        string += '  R=\n' + str(self.R) + '\n'

        return string

    def stateString(self, cur_state):
        """ Return a string representation of the state.

        Arguments:
            cur_state (int) - the state
        """
        return str(self.S(cur_state))

    def policyString(self, policy):
        """ Return a string representation of a policy.

        Arguments:
            policy - the policy, a numpy array of size len(S) X len(A)
        """
        raise NotImplementedError('subclasses must override policyString()!')

    def policyStringMode(self, policy):
        """ Return a string representation of the argmax policy.

        Arguments:
            policy - the policy, a numpy array of size len(S) X len(A)
        """
        raise NotImplementedError(
            'subclasses must override policyStringMode()!')

    def valuesString(self, values):
        """ Return a string representation of the values.

        Arguments:
            values - the values, a numpy array of size len(S)
        """
        raise NotImplementedError('subclasses must override valuesString()!')

    def plotValues(self, ax, values, policy):
        """ Plot the values on an axis.

        Arguments:
            ax - matplotlib axis on which to plot
            values - the values, a numpy array of size len(S)
            policy - the policy, a numpy array of size len(S) X len(A)
        """
        pass
