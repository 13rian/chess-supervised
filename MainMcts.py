from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils import data
import random
import numpy as np
import logging
import torch
import chess.variant


from utils import utils
import board_representation
import game
import mcts


# @utils.profile
def mainTrain():
    # The logger
    utils.init_logger(logging.DEBUG, file_name="log/chess_sl.log.log")
    logger = logging.getLogger('MCTS')

    np.set_printoptions(suppress=True, precision=6)


    # set the random seed
    random.seed(a=None, version=2)
    np.random.seed(seed=None)


    # parameters
    network_dir = "networks"
    network_file = network_dir + "/" + "network_batch_12000.pt"
    training_progress_dir = "training_progress"

    # load the network
    logger.info("load the neural network: " + network_file)
    net = torch.load(network_file, map_location='cuda')


    board = chess.Board()

    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")

    bit_board = board_representation.board_to_matrix(board)

    policy, value = net(torch.Tensor(bit_board).unsqueeze(0).cuda())
    print(policy)
    print(value)

    print("move: ", board_representation.policy_to_move(policy.detach().cpu().numpy(), board.turn))

    print(board.legal_moves)


    # find the policy with the help of mcts
    board = chess.Board()
    board.push_san("g4")
    board.push_san("e5")
    board.push_san("f4")

    chess_board = game.ChessBoard()
    chess_board.chess_board = board
    policy = mcts.mcts_policy(chess_board, 200, net, 1, 0)

    print(board_representation.policy_to_move(policy, chess_board.chess_board.turn))





if __name__ == '__main__':
    mainTrain()
