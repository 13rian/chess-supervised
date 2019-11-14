import logging
import random
import chess

import numpy as np

from utils import utils
import globals


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

    a = np.array([[1,2,3,4], [5,6,7,8]])
    print(a)
    print(a.flatten())

    print("length of all possible uci moves: ", len(globals.ALL_MOVES))

    # mirror a move
    move = chess.Move.from_uci("a1a2")
    from_square = chess.square_mirror(move.from_square)
    to_square = chess.square_mirror(move.to_square)
    print("mirrored a1a2: ", chess.Move(from_square, to_square, move.promotion, move.drop))



if __name__ == '__main__':
    mainTrain()