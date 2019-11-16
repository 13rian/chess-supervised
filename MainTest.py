import logging
import random
import chess

import numpy as np

from utils import utils
from globals import CONST
import data_processing
import board_representation

import tables


#@utils.profile
def mainTrain():
    # The logger
    utils.init_logger(logging.DEBUG, file_name="log/chess_sl.log")
    logger = logging.getLogger('Chess_SL')

    # set the random seed
    # set the random seed
    random.seed(a=None, version=2)
    np.random.seed(seed=None)

    logger.debug("start the main test program")

    filter = data_processing.get_compression_filter()
    data_file = tables.open_file("king-base-light.h5", mode='r', filters=filter)


    print(data_file.root.data.shape[0])
    state = data_file.root.data[2, 0:CONST.STATE_SIZE]
    policy_idx = int(data_file.root.data[2, -2])
    value = data_file.root.data[100, -1]

    state = state.reshape(CONST.INPUT_CHANNELS, CONST.BOARD_HEIGHT, CONST.BOARD_WIDTH)

    policy = np.zeros(board_representation.LABEL_COUNT)
    policy[policy_idx] = 1


    pgn_file = open("pgns/KingBaseLite2019-B00-B19.pgn")
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