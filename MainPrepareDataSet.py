import logging
import os

import chess.pgn

from utils import utils
import pgn_processing
import board_representation
from globals import CONST


pgn_dir = "pgns"
pgn_file = "pgns/KingBaseLite2019-A80-A99.pgn"


# The logger
utils.init_logger(logging.DEBUG, file_name="log/chess_sl.log")
logger = logging.getLogger('Chess_SL')


# get all pgn files
path_list = os.listdir(pgn_dir)
print(path_list)





pgn = open(pgn_file)
# for i in range(1000000):
#     game = chess.pgn.read_game(pgn)
#
#     value = pgn_processing.value_from_result(game.headers["Result"])
#     print(value)
#
#     if game is None:
#         logger.debug("parsed all games, count {}".format(i + 1))
#         break

# test to get the training example of the first position
board = chess.Board()
game = chess.pgn.read_game(pgn)
for move in game.mainline_moves():
    print(move)
    board.push(move)

print(game)


# get the board representation of the last position
board_matrix = board_representation.board_to_matrix(board)
# print(board_matrix[0])
board_matrix = board_matrix.flatten()
# board_matrix = board_matrix.reshape(CONST.BOARD_INPUT_FEATURES, CONST.BOARD_WIDTH, CONST.BOARD_HEIGHT)
# print(board_matrix[0])


# get the value of the game
value = pgn_processing.value_from_result(game.headers["Result"])
print("value of the first game: ", value)


