import random
import numpy as np

import mdps


def policyEvaluation(mdp, policy, n_iterations=1000, verbose=False, ax=None):
    """ Perform policy evaluation, i.e. determine the values given a policy.

    Args:
        mdp - a MarkovDecisionProcess
        policy - the policy, a numpy array of size len(S) X len(A)
        n_iterations - the number of iterations to run the algorithm
        verbose - whether to print intermediate results
        ax - if this is a matplotlib axes, you can plot on it

    Returns:
        the values, a numpy array of size len(S)

    """

    # This may be useful
    n_states = len(mdp.S)
    n_actions = len(mdp.A)

    # V contains the values
    V = np.zeros((n_states,))

    # Here are some functions that may be useful for print debugging and plots.
    print(mdp.policyString(policy) + '\n')
    print(mdp.valuesString(V) + '\n')
    if ax:
        mdp.plotValues(ax, V, policy)

    # IMPLEMENT POLICY EVALUATION HERE
    epsilon = 0.01
    for _ in range(n_iterations):
        delta = 0
        for i, state in enumerate(mdp.S):
            value = V[i]
            tmp = np.sum(mdp.P[i]*(mdp.R[i]+mdp.discount*V), axis=1)
            V[i] = np.sum(policy[i]*tmp)
            delta = max(delta, value - V[i])
        if delta < epsilon:
            break

    # Return values for the policy
    return V


def valueIteration(mdp, policy=None, n_iterations=1000, verbose=False,
                   ax=None):
    """ Perform value iteration, i.e. determine the optimal and values.

    Args:
        mdp - a MarkovDecisionProcess
        policy - an (optional) initial policy, a numpy array, size |S| X |A|
        n_iterations - the number of iterations to run the algorithm
        verbose - whether to print intermediate results
        ax - if this is a matplotlib axes, you can plot on it

    Returns a tuple with:
        the (optimal) values, a numpy array of size len(S)
        the (optimal) policy, a numpy array of size len(S) X len(A)

    """

    n_states = len(mdp.S)
    n_actions = len(mdp.A)

    # If no policy is passed, initialize a random one.
    if policy is None:
        # Initialize random policy
        policy = np.full((n_states, n_actions), 1.0 / n_actions)

    # V contains the values
    V = np.zeros((n_states,))

    # IMPLEMENT VALUE ITERATION HERE
    epsilon = 0.01
    for _ in range(n_iterations):
        delta = 0
        for i in range(n_states):
            value = V[i]
            tmp = np.sum(mdp.P[i]*(mdp.R[i]+mdp.discount*V), axis=1)
            V[i] = np.max(tmp)
            delta = max(delta, value - V[i])
        if delta < epsilon:
            break

    for i, state in enumerate(mdp.S):
        tmp = np.sum(mdp.P[i, :, :]*(mdp.R[i, :]+mdp.discount*V), axis=1)
        policy[i] = np.argmax(tmp)

    # Return (optimal?) values and (optimal?) policy
    return (V, policy)


if __name__ == '__main__':

    # INITIALIZE MARKOV DECISION PROCESS

    # Default is to not plot anything
    ax1 = None
    ax2 = None

    # Get the name of the MDP from the command line.
    import sys
    if (len(sys.argv) < 2):
        mdp_name = 'Grid'
    else:
        mdp_name = sys.argv[1]

    # Initialize an MDP
    if mdp_name == 'FlipCoin':
        from mdps.mdp_flip_coins import MDPFlipCoin
        mdp = MDPFlipCoin()

    elif mdp_name == 'FlipTwoCoins':
        from mdps.mdp_flip_coins import MDPFlipTwoCoins
        mdp = MDPFlipTwoCoins()

    elif mdp_name == 'Grid':
        from mdps.mdp_grid import MDPGrid

        # Test different parameters and see what happens!
        n_rows = 2
        n_cols = 4
        stochasticity = 0.0
        discount = 1.0
        mdp = MDPGrid(n_rows, n_cols, stochasticity, discount)

        # Prepare axis for plotting
        # This may lead to Python2/Python3 issues...
        enable_plotting = False
        if enable_plotting:
            import matplotlib.pyplot as plt
            fig = plt.figure(1, figsize=(12, 6))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

    else:
        raise LookupError('Cannot find MDP with name "' + mdp_name + '"')

    verbose = True  # You can use this to enable/surpress output

    # INITIALIZATION DONE: DO POLICY EVALUATION

    # Initialize a policy that always returns a random action
    # Test different policies and see what happens!
    n_states = len(mdp.S)
    n_actions = len(mdp.A)
    policy = np.full((n_states, n_actions), 1.0 / n_actions)

    # Evaluate the policy
    print("________________________________________________________")
    print("Performing policy evaluation.")
    values = policyEvaluation(mdp, policy, 100, verbose, ax1)

    # Print the result
    print("POLICY\n" + mdp.policyString(policy) + '\n')
    print("ARGMAX POLICY\n" + mdp.policyStringMode(policy) + '\n')
    print("VALUES\n" + mdp.valuesString(values) + '\n')

    # Plot final result
    if ax1:
        mdp.plotValues(ax1, values, policy)

    # POLICY EVALUATION DONE: DO VALUE ITERATION

    # Perform value iteration
    print("________________________________________________________")
    print("Performing value iteration.")
    policy = None
    (values, policy) = valueIteration(mdp, policy, 100, verbose, ax2)

    # Print the result
    print("POLICY\n" + mdp.policyString(policy) + '\n')
    print("ARGMAX POLICY\n" + mdp.policyStringMode(policy) + '\n')
    print("VALUES\n" + mdp.valuesString(values) + '\n')

    # Plot final result
    if ax2:
        mdp.plotValues(ax2, values, policy)

    if ax1 and ax2:
        plt.show()
