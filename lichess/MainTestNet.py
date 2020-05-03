from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils import data
import random
import numpy as np
import logging
import torch
import chess.variant


from utils import utils
import mcts
from globals import Config
import networks
import data_processing
import data_storage


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
    variant = "threeCheck"
    network_dir = "networks/" + variant
    network_file = network_dir + "/" + "network_batch_13071.pt"

    # load the network
    logger.info("load the neural network: " + network_file)
    net = torch.load(network_file, map_location='cpu')


    board = chess.variant.ThreeCheckBoard()

    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Bc4")
    board.push_san("Bc5")
    board.push_san("Bxf7+")

    board.push_san("Kxf7")
    board.push_san("Qf3+")
    board.push_san("Kg6")
    board.push_san("Qg3")
    print(board.legal_moves)







if __name__ == '__main__':
    mainTrain()
