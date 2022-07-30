# monte_carlo_tree_search
Implementation of Monte Carlo Tree Search (MCTS) in python, with a playable connect4 game to allow a player to play against an AI using MCTS.

--About MCTS--
MCTS was notably used (amongst many other techniques) by Google's AlphaGo program, which defeated celebrated professional Go player Lee Sedol in 2016.

MCTS is an algorithm for choosing an action in a simulatable game or environemnt that has multiple choices as well as "rewards" specified for different outcomes at the end of the game / environment. Each turn it explores a decision tree using a combination of tree search, monte carlo simulation, and an exploration strategy to prrioritize which nodes to investigate most thoroughly.

MCTS is effective for decision trees that are wide (many actions) but can struggle with trees that are deep (many turns / time steps).

The implementation of MCTS is game-agnostic, meaning it can easily be used for other games, provided they are implemented to include the right methods, as outlined in MCTS.py.


--Getting Started--
To play connect4 against the MCTS algorithm, simply clone the repository and run connect4_curses.py from command line.


--pytorch dependency--
The Connect4 implementation currently uses pytorch tensors, due to the context it was built in (as a basic test environment for testing reinforcement learning algorithms and using neural network evaluation to replace the randomized rollout policy). I hope to convert these tensors to numpy arrays in the future to remove the dependency on pytorch.
