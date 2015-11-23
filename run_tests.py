


def _test_mdps():
    
    import mdps
    from mdps.mdp_flip_coins import MDPFlipCoin, MDPFlipTwoCoins
    from mdps.mdp_grid import MDPGrid
    
    print('____________________________________________')
    mdp = MDPFlipCoin()
    print(mdp)

    print('____________________________________________')
    mdp = MDPFlipTwoCoins()
    print(mdp)

    print('____________________________________________')
    n_rows=2
    n_cols=3
    stochasticity=0.1
    mdp = MDPGrid(n_rows,n_cols,stochasticity)
    print(mdp)
    
    
def _test_environments():
    
    import environments
    from environments.env_maze import EnvMaze
    
    char_grid = [
        ['T','.','.','.','.'],
        ['.','.','.','.','.'],
        ['.','.','.','.','.'],
        ['W','W','.','W','W'],
        ['.','.','.','.','.'],
        ['.','.','.','.','.'],
    ]
    stochasticity=0.1
    bump_penalty=-10
    maze = EnvMaze(char_grid,stochasticity,bump_penalty)
    
    maze.reset()
    print(maze.stateString())
    maze.performAction('LEFT')
    print(maze.stateString())
    maze.performAction('UP')
    print(maze.stateString())

    from get_environment_from_name import get_environment_from_name
    
    names = ['FlipCoin', 'FlipTwoCoins', 'Grid', 'Maze']
    
    for env_name in names:
        print('____________________________________________')
        print(env_name)
        
        env = get_environment_from_name(env_name)
        is_finished = env.isFinished()
        print(is_finished)
        
        env.reset()
        print(env)
        action = 0
        env.performAction(action)
        print('')
        print(env)
        reward = env.getReward()
        print(reward)
        is_finished = env.isFinished()
        print(is_finished)
        
        if env_name == 'Grid':
            env.performAction('UP')
            
    
if __name__ == '__main__':
    _test_mdps()    
    _test_environments()
    
