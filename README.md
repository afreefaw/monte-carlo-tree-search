# Monte Carlo Tree Search (MCTS)

Implementation of Monte Carlo Tree Search (MCTS) in python, with a playable Connect4 game to allow a player to play against an AI using MCTS.

## Getting Started

To play Connect4 against the MCTS algorithm, simply clone the repository and install requirements: 
```bash
pip install -r -requirements.txt
```
For windows machines, you must also
```bash
pip install windows-curses
```
Then run:
```bash
python connect4_curses.py
```
When playing against the AI, use the number keys 1-7 to select where to play your piece. Try setting the AI 'thinking time' higher for more of a challenge.

## About MCTS

MCTS was notably used (amongst many other techniques) by Google Deepmind's AlphaGo, which defeated celebrated professional Go player Lee Sedol in 2016.

MCTS is an algorithm for choosing an action in a simulatable game or environment that has multiple choices as well as "rewards" specified for different outcomes at the end of the game / environment. Each turn it explores a decision tree using a combination of tree search, Monte Carlo simulation, and an exploration strategy to prioritize which nodes to investigate most thoroughly.

This implementation of MCTS is game-agnostic, meaning it can easily be used for other games, provided they are implemented to include the right methods, as outlined in `MCTS.py`.

## Further Info

The Connect4 implementation currently uses PyTorch tensors, due to the context in which it was built (testing use of neural network evaluation as a rollout policy).
