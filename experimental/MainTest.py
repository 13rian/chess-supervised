import logging
import random
import chess
import torch

import numpy as np

from utils import utils
from globals import CONST, Config
import data_processing
import board_representation
import networks
import data_storage

import tables


#@utils.profile
def mainTrain():
    # The logger
    utils.init_logger(logging.DEBUG, file_name="../log/chess_sl.log")
    logger = logging.getLogger('Chess_SL')

    # set the random seed
    # set the random seed
    random.seed(a=None, version=2)
    np.random.seed(seed=None)

    logger.debug("start the main test program")

    # test the rise network
    network = networks.RiseNet(Config.learning_rate, Config.n_blocks, Config.n_se_blocks, Config.n_filters,
                               Config.se_ratio, Config.n_mobile_filters, Config.n_filter_inc, Config.weight_decay)
    network = data_storage.net_to_device(network, Config.training_device)

    board = chess.Board()
    input = board_representation.board_to_matrix(board)
    input = torch.tensor(input)
    input = input.to(Config.training_device, dtype=torch.float)
    input = input.unsqueeze(0)


    res = network(input)


    board = chess.Board()
    board.push_san("g4")
    board.push_san("e5")
    board.push_san("f4")
    board.push_uci("d8h4")
    # board.push_san("Qh4")

    print(board.turn == chess.WHITE)


    list = [1, 2, 3, 4]
    list.remove(2)
    print(list)



    test_str = "_5_6"
    print(test_str.split("_"))


    # get the fen string of a board
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    board.push_san("Nc6")
    board.push_san("Bc4")
    board.push_san("Bc5")
    board.push_san("Qe2")
    board.push_san("d6")
    board.push_san("Nc3")
    board.push_san("Bd7")
    board.push_san("b3")
    board.push_san("Qe7")
    fen_string = board.fen()
    print(fen_string)
    print("n-bytes: ", len(fen_string.encode('utf-8')))



    filter = data_processing.get_compression_filter()
    data_file = tables.open_file("../king-base-light-avg.h5", mode='r', filters=filter)


    print(data_file.root.data.shape[0])
    state = data_file.root.data[2, 0:CONST.STATE_SIZE]
    policy_idx = int(data_file.root.data[2, -2])
    value = data_file.root.data[100, -1]

    state = state.reshape(CONST.INPUT_CHANNELS, CONST.BOARD_HEIGHT, CONST.BOARD_WIDTH)

    policy = np.zeros(board_representation.LABEL_COUNT)
    policy[policy_idx] = 1


    pgn_file = open("../pgns/KingBaseLite2019-B00-B19.pgn")
    game = chess.pgn.read_game(pgn_file)  # read out the next game from the pgn
    while game is not None:
        result = data_processing.value_from_result(game.headers["Result"])
        if result is None:
            print(game)

        game = chess.pgn.read_game(pgn_file)  # read out the next game from the pgn
        for move in game.mainline_moves():
            if move.uci() == "0000":
                print(game)
                print(move)





if __name__ == '__main__':
    mainTrain()