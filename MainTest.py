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

    print(board)
    print(" mirror:")
    board = board.mirror()
    print(board)

    print(board.has_queenside_castling_rights(chess.BLACK))


    import board_representation
    board = chess.Board()
    bit_board = board_representation.board_to_matrix(board)
    print(bit_board[0])

    board.push_san("e4")
    bit_board = board_representation.board_to_matrix(board)
    print(bit_board[0])
    print(" ")
    print(bit_board[6])



    a = np.array([1, 2, 3])
    b = np.array([4])
    c = np.array([5, 6, 7])
    print(np.concatenate((a, b, c)))


if __name__ == '__main__':
    mainTrain()