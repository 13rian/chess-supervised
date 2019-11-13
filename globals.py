import torch

class CONST:
	WHITE = 1
	BLACK = 0

	BOARD_WIDTH = 8 	    				 # the width of the board (number of columns)
	BOARD_HEIGHT = 8						 # the height of the board
	BOARD_SIZE = BOARD_WIDTH * BOARD_HEIGHT  # the size of the board

	# normalization constants
	MAX_TOTAL_MOVES = 500 					 # normalization constant for the number of moves played
	MAX_PROGRESS_COUNTER = 40 				 # normalization constant for the half move clock


class Config:
	# n threads for the mcts
	n_processes = 1

	# torch devices for training and evaluation
	evaluation_device = torch.device('cuda')		# the pytorch device that is used for evaluation
	training_device = torch.device('cuda') 		# the pytorch device that is used for training

	# hyperparameters
	cycle_count = 1000  	# the number of alpha zero cycles
	episode_count = 200  	# the number of games that are self-played in one cycle 2000
	epoch_count = 2  		# the number of times all training examples are passed through the network 10
	mcts_sim_count = 200  	# the number of simulations for the monte-carlo tree search 800
	c_puct = 4 	 			# the higher this constant the more the mcts explores 4
	temp = 1  				# the temperature, controls the policy value distribution
	temp_threshold = 42  	# up to this move the temp will be temp, otherwise 0 (deterministic play)
	alpha_dirich = 1  		# alpha parameter for the dirichlet noise (0.03 - 0.3 az paper, 10/ avg n_moves) 0.3
	n_filters = 128  		# the number of filters in the conv layers 128
	learning_rate = 0.001  	# the learning rate of the neural network
	weight_decay = 1e-4     # weight decay to prevent overfitting, should be twice as large as L2 regularization const
	dropout = 0.2  			# dropout probability for the fully connected layers 0.3
	n_blocks = 10  			# number of residual blocks
	batch_size = 256  		# the batch size of the experience buffer for the neural network training
	window_size = 100 		# size of the training window (number of cycles for the training data)


process_pool = None