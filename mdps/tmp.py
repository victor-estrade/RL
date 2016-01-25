import numpy as np
from matplotlib import pyplot as plt

from .markov_decision_process import MarkovDecisionProcess


class MDPGrid(MarkovDecisionProcess):
    """Markov Decision Process for a 2D Grid."""

    def __init__(self, n_rows=3, n_cols=4, stochasticity=0.0, discount=1.0):
        """ Initialize a 2D Grid MDP.

        Args:
            n_rows (int): number of rows in the grid
            n_cols (int): number of cols in the grid
            stochasticity (float) : level of stochasticity
                stochasticity = 0 : deterministic MDP
                example stochasticity = 0.2 : there is a 0.2 change that an
                action has no effect (i.e. the agent stays where it is)
            discount (float) : discount factor
        """
        name = "Grid"

        # State space
        n_states = n_rows*n_cols
        S = list(range(n_states))

        # Action space
        if n_rows == 1:
            A = ['LEFT', 'RIGHT']
        else:
            A = ['LEFT', 'RIGHT', 'UP', 'DOWN']
        n_actions = len(A)

        # Transition function S x A x S -> probability
        P = np.zeros((n_states, n_actions, n_states,))
        for i_row in range(n_rows):
            for i_col in range(n_cols):
                s = i_row*n_cols + i_col
                #          col
                #       1  2  3  4
                #
                # r 1   1  2  3  4
                # o 2   5  6  7  8 state 's'
                # w 3   9 10 11 12
                #

                # Here come the transitions for the 'LEFT' action
                LEFT = 0
                if i_col > 0:
                    P[s, LEFT, s-1] = 1.0-stochasticity  # Successful move LEFT
                    P[s, LEFT, s] = stochasticity  # Fail: stayed where you are
                else:
                    P[s, LEFT, s] = 1.0  # Always bumps into wall on far left

                # Here come the transitions for the 'RIGHT' action
                RIGHT = 1
                if i_col < (n_cols-1):
                    P[s, RIGHT, s+1] = 1.0-stochasticity  # Successful move RIGHT
                    P[s, RIGHT, s] = stochasticity  # Fail: stayed where you are
                else:
                    P[s, RIGHT, s] = 1.0  # Always bumps into wall on far left

                if n_actions > 2:
                    # Here come the transitions for the 'UP' action
                    UP = 2
                    if i_row > 0:
                        P[s, UP, s-n_cols] = 1.0-stochasticity  # Successful move UP
                        P[s, UP, s] = stochasticity  # Fail: stayed where you are
                    else:
                        P[s, UP, s] = 1.0  # Always bumps into wall on far left 

                    # Here come the transitions for the 'RIGHT' action
                    DOWN = 3
                    if i_row < (n_rows-1):
                        P[s, DOWN, s+n_cols] = 1.0-stochasticity  # Successful move
                        P[s, DOWN, s] = stochasticity  # Fail: stayed where you are
                    else:
                        P[s, DOWN, s] = 1.0  # Always bumps into wall on far left 

        # Terminal states (not probabilities, but true/false)
        T = [False] * n_states
        T[0] = True  # First state is a terminal state

        # Initial state distribution: uniform over all states except the
        # terminal state.
        n_nonterminal_states = n_states - sum(T)
        I = [float(1-terminal)/float(n_nonterminal_states) for terminal in T]

        # For terminal states, the probability of going to another state is
        # 0. This is obvious, but needs to be explicitly set, otherwise some
        # of the recursive equations will not work properly.
        for s in S:
            if T[s]:
                P[s, :, :] = 0

        # Reward function
        R = np.full((n_states, n_states,), -1.0)
        for s in S:
            if T[s]:
                R[:, s] = 100.0  # If you go to a terminal state, reward of 100

        # Create the MDP by calling __init__ in the base class
        # MarkovDecisionProcess
        # I have kept this compatible with both Python 2 and 3
        MarkovDecisionProcess.__init__(self, S, T, I, A, P, R, discount, name)

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.is_wall = None

    def stateString(self, cur_state):
        """See documentation in the base class."""
        string = ''
        for i_row in range(self.n_rows):
            string += '| '
            for i_col in range(self.n_cols):
                s = i_row*self.n_cols + i_col
                if s == cur_state:
                    string += 'A '
                else:
                    string += '. '
            string += '|'
            if i_row < (self.n_rows-1):
                string += '\n'
        return string

    def valuesString(self, values):
        """See documentation in the base class."""
        string = ''
        for i_row in range(self.n_rows):
            string += '| '
            for i_col in range(self.n_cols):
                s = i_row*self.n_cols + i_col
                if self.isTerminalState(s):
                    string += '      T '
                else:
                    string += '{: > 7.2f} '.format(values[s])
            string += '|'
            if i_row < (self.n_rows-1):
                string += '\n'

        return string

    def policyStringMode(self, policy):
        """See documentation in the base class."""
        string = ''
        for i_row in range(self.n_rows):
            string += '| '
            for i_col in range(self.n_cols):
                s = i_row*self.n_cols + i_col
                if self.isTerminalState(s):
                    string += '      T '
                else:
                    action = np.argmax(policy[s, :])
                    string += '{: >7} '.format(self.A[action])
            string += '|'
            if i_row < (self.n_rows-1):
                string += '\n'
        return string

    def policyString(self, policy):
        """See documentation in the base class."""
        string = ''
        for action in range(len(self.A)):
            string += '{: >7} \n'.format(self.A[action])
            for i_row in range(self.n_rows):
                string += '| '
                for i_col in range(self.n_cols):
                    s = i_row*self.n_cols + i_col
                    if self.isTerminalState(s):
                        string += '      T '
                    else:
                        string += '{: > 7.2f} '.format(policy[s, action])
                        # action = np.argmax(policy[s,:])
                        # string += '{: >7} '.format(self.A[action])
                string += '|'
                if i_row < (self.n_rows-1):
                    string += '\n'
            string += '\n'
        return string

    def plotValues(self, ax, values, policy=None):
        """See documentation in the base class."""

        # Make copy of values, and set values of terminal states to nan
        vals = values[:]
        # for ii, is_terminal in enumerate(self.T):
        #    if is_terminal:
        #        vals[ii] = float('nan')

        # Plot values as a 2D grid
        values_grid = vals.reshape(self.n_rows, self.n_cols)
        ax.imshow(values_grid, interpolation='none',
                  cmap=plt.get_cmap('Greens'))

        # Plot terminal states
        for ii in range(self.n_rows):
            for jj in range(self.n_cols):
                s = ii*self.n_cols + jj
                if self.T[s]:
                    ax.text(jj, ii, 'T', color='red')

        # If there is a policy, plot arrow, whose thickness represented their
        # probability.
        if policy is not None:
            for ii in range(self.n_rows):
                for jj in range(self.n_cols):
                    s = ii*self.n_cols + jj
                    if not self.T[s]:
                        ax.text(jj, ii, ('%1.1f' % values[s]), ha='center')
                        for a_i, a_name in enumerate(self.A):
                            p = policy[s, a_i]
                            if p > 0.01:
                                w = 0.01*p
                                if a_name == 'LEFT':
                                    ax.arrow(jj-0.1, ii, -0.2, 0.0, width=w)
                                if a_name == 'RIGHT':
                                    ax.arrow(jj+0.1, ii, +0.2, 0.0, width=w)
                                if a_name == 'UP':
                                    ax.arrow(jj, ii-0.1, 0.0, -0.2, width=w)
                                if a_name == 'DOWN':
                                    ax.arrow(jj, ii+0.1, 0.0, +0.2, width=w)
