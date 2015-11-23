import environments

def get_environment_from_name(env_name):
    """ Initialize environments with default parameters, given a name.
    
    Arguments:
        env_name - Name of the environment
        
    Returns:
        an environment of class Environment.
    """
    
    if env_name == 'FlipCoin':
        import mdps
        from mdps.mdp_flip_coins import MDPFlipCoin
        from environments.env_with_mdp import EnvWithMDP        
        return EnvWithMDP(MDPFlipCoin())
        
    elif env_name == 'FlipTwoCoins':
        from mdps.mdp_flip_coins import MDPFlipTwoCoins
        from environments.env_with_mdp import EnvWithMDP
        return EnvWithMDP(MDPFlipTwoCoins())

    elif env_name == 'Grid':
        from mdps.mdp_grid import MDPGrid
        from environments.env_with_mdp import EnvWithMDP
        n_rows = 4
        n_cols = 5
        stochasticity = 0.0
        discount = 1.0
        mdp = MDPGrid(n_rows,n_cols,stochasticity,discount)
        return EnvWithMDP(mdp)
        
    elif env_name == 'Maze':
        from environments.env_maze import EnvMaze
        char_grid = [
            ['T','.','.','.','.'],
            ['W','W','W','.','W'],
            ['.','.','W','.','.'],
            ['.','W','W','.','.'],
            ['.','.','.','.','.'],
        ]
        stochasticity=0.1
        bump_penalty=-10
        return EnvMaze(char_grid,stochasticity,bump_penalty)
        
    else:
        raise LookupError('Cannot find Environment with name "'+env_name+'"')
    
    
