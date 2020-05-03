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


# @utils.profile
def mainTrain():
    # The logger
    utils.init_logger(logging.DEBUG, file_name="log/chess_sl.log.log")
    logger = logging.getLogger('Sup Learning')

    np.set_printoptions(suppress=True, precision=6)


    # set the random seed
    random.seed(a=None, version=2)
    np.random.seed(seed=None)


    # parameters
    network_dir = "networks"
    network_file = network_dir + "/" + "network_batch_17000.pt"

    # load the network
    logger.info("load the neural network: " + network_file)
    net = torch.load(network_file, map_location='cpu')


    board = chess.Board()

    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")

    bit_board = board_representation.board_to_matrix(board)

    policy, value = net(torch.Tensor(bit_board).unsqueeze(0))
    print(policy)
    print(value)

    print("move: ", board_representation.policy_to_move(policy.detach().numpy()))

    print(board.legal_moves)







if __name__ == '__main__':
    mainTrain()
