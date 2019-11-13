import logging
import random

import chess.pgn

from utils import utils


pgn_file = "test_data/KingBaseLite2019-A80-A99.pgn"


# The logger
utils.init_logger(logging.DEBUG, file_name="log/chess_sl.log")
logger = logging.getLogger('Chess_SL')


pgn = open(pgn_file)
# for i in range(1000000):
#     game = chess.pgn.read_game(pgn)
#     if game is None:
#         logger.debug("parsed all games, count {}".format(i + 1))
#         break

# test to
board = chess.Board()
game = chess.pgn.read_game(pgn)
for move in game.mainline_moves():
    board.push(move)