import numpy as np
import random

from .environment import Environment


class EnvMaze(Environment):

    """ Environment for represting mazes (grids with walls). """

    def __init__(self, char_grid=None, stochasticity=0.0, bump_penalty=0):
        """ Initialize a maze environment.

        Arguments:
            char_grid - a list of lists of characters, where the characters
            represent  '.' => empty, 'T' => terminal, 'W' => wall

            stochasticity - probability that an action does not have any effect

            bump_penalty - penalty for bumping in to a wall, e.g. -10
        """
        if not char_grid:
            # Initialize with default grid
            char_grid = [
                ['T', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.'],
                ['W', 'W', '.', 'W', 'W'],
                ['.', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.'],
            ]

        self._char_grid = char_grid
        self._n_rows = len(char_grid)
        self._n_cols = len(char_grid[0])

        # Action space
        self.A = ['LEFT', 'RIGHT', 'UP', 'DOWN']

        # Current position of the robot
        self._agent_row = 0
        self._agent_col = 0

        # Did the agent bump into a wall when going from the previous state to
        # the current one?
        self._prev_bumped = False

        # Penalty for bumping into a wall
        self._bump_penalty = bump_penalty

        self._stochasticity = stochasticity

    def numActions(self):
        """ See documentation in base class."""
        return len(self.A)

    def numStates(self):
        """ See documentation in base class."""
        return self._n_rows * self._n_cols

    def isFinished(self):
        """ See documentation in base class."""
        return self._char_grid[self._agent_row][self._agent_col] == 'T'

    # def isTerminalState(self, state):
    #    """ See documentation in base class."""
    #    row = state//self._n_cols
    #    col = state%self._n_cols
    #    return self._char_grid[row][col] == 'T'

    def reset(self):
        """ See documentation in base class."""
        valid_agent_position = False
        while not valid_agent_position:
            row = random.randint(0, self._n_rows - 1)
            col = random.randint(0, self._n_cols - 1)
            char = self._char_grid[row][col]
            if char == '.' or char == ' ':
                self._agent_row = row
                self._agent_col = col
                valid_agent_position = True

    def performAction(self, action):
        """ See documentation in base class."""

        if random.random() < self._stochasticity:
            # State doesn't change
            self.last_reward = -1
            return

        new_agent_row = self._agent_row
        new_agent_col = self._agent_col

        if isinstance(action, int):
            action = self.A[action]

        if action == 'LEFT':
            new_agent_col -= 1
        if action == 'RIGHT':
            new_agent_col += 1
        if action == 'UP':
            new_agent_row -= 1
        if action == 'DOWN':
            new_agent_row += 1

        bumped = False
        if new_agent_row < 0 or new_agent_row >= self._n_rows:
            bumped = True
        elif new_agent_col < 0 or new_agent_col >= self._n_cols:
            bumped = True
        elif self._char_grid[new_agent_row][new_agent_col] == 'W':
            bumped = True

        if not bumped:
            # Didn't bump into wall: move agent
            self._agent_row = new_agent_row
            self._agent_col = new_agent_col

        # getReward needs to know if the agent bumped. So store it here.
        self._prev_bumped = bumped

    def getReward(self):
        """ Compute and return the current reward
        (i.e. corresponding to the last action performed)
        """

        # Found the exit! (all terminal states are exits)
        if self._char_grid[self._agent_row][self._agent_col] == 'T':
            return 100.0

        # Bumped into a wall
        if self._prev_bumped:
            return self._bump_penalty

        # Default penalty for moving around
        return -1.0

    def getObservation(self):
        """ See documentation in base class."""
        return self._agent_row * self._n_cols + self._agent_col

    def __str__(self):
        return self.stateString()

    def stateString(self):
        """ See documentation in base class."""
        string = ''
        for ii in range(self._n_rows):
            for jj in range(self._n_cols):
                if self._agent_row == ii and self._agent_col == jj:
                    string += 'A '
                else:
                    string += self._char_grid[ii][jj] + ' '
            string += '\n'
        return string

    def actionString(self, action):
        """ See documentation in base class."""
        return str(self.A[action])
