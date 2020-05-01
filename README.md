# Learning to play chess with supervised learning


## Goal of this Project 
The idea of this project is to train a neural network with supervised learning to play chess and some variants of chess like Three Check, King of the Hill, Crazyhouse etc. The algorithms used are based on [this](https://arxiv.org/abs/1908.06660) paper.


## Data Set
The games for the chess variants were downloaded from the [lichess database](https://database.lichess.org/). Only games over a certain ELO threshold were considered to train the neural network.


## Game Play
The network is used together with Monte-Carlo Tree Search (MCTS) to play a game, as described in the AlphaZero paper. 