import logging
import random
import time

import numpy as np
import chess.pgn

from utils import utils
from globals import CONST, PlaneIndex
import board_representation


# The logger
utils.init_logger(logging.DEBUG, file_name="../log/chess_sl.log")
logger = logging.getLogger('Chess_SL')



########################################################################################
#                                   test the pieces                                    #
########################################################################################
logger.info("start the board representation tests")

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
bit_board = board_representation.board_to_matrix(board)

################################# white pieces
# white pawn on b3
if bit_board[PlaneIndex.white_pawns][5][1] < 0.9:
    logger.error("white pawn is not on b3")

# white knight on c3
if bit_board[PlaneIndex.white_knights][5][2] < 0.9:
    logger.error("white knight is not on c3")

# white bishop on c1
if bit_board[PlaneIndex.white_bishops][7][2] < 0.9:
    logger.error("white bishop is not on c1")

# white rook on a1
if bit_board[PlaneIndex.white_rooks][7][0] < 0.9:
    logger.error("white rook is not on a1")

# white queen on e2
if bit_board[PlaneIndex.white_queens][6][4] < 0.9:
    logger.error("white queen is not on e2")

# white king on e1
if bit_board[PlaneIndex.white_kings][7][4] < 0.9:
    logger.error("white king is not on e1")


################################# black pieces
# black pawn on d6
if bit_board[PlaneIndex.black_pawns][2][3] < 0.9:
    logger.error("black pawn is not on d6")

# black knight on g8
if bit_board[PlaneIndex.black_knights][0][6] < 0.9:
    logger.error("black knight is not on g8")

# black bishop on d7
if bit_board[PlaneIndex.black_bishops][1][3] < 0.9:
    logger.error("black bishop is not on d7")

# black rook on h8
if bit_board[PlaneIndex.black_rooks][0][7] < 0.9:
    logger.error("black rook is not on h8")

# black queen on e7
if bit_board[PlaneIndex.black_queens][1][4] < 0.9:
    logger.error("black queen is not on e7")

# black king on e8
if bit_board[PlaneIndex.black_kings][0][4] < 0.9:
    logger.error("black king is not on e8")




########################################################################################
#                                   test repetitions                                   #
########################################################################################
board = chess.Board()
board.push_san("e4")
board.push_san("e5")
board.push_san("Nf3")
board.push_san("Nc6")
bit_board = board_representation.board_to_matrix(board)

# both repetitions should be zero
if np.sum(bit_board[PlaneIndex.rep2]) > 0.1:
    logger.error("no repetition: first repetition feature is not 0")

if np.sum(bit_board[PlaneIndex.rep3]) > 0.1:
    logger.error("no repetition: second repetition feature is not 0")



# repeat the position once
board.push_san("Ng1")
board.push_san("Nb8")
bit_board = board_representation.board_to_matrix(board)

if np.sum(bit_board[PlaneIndex.rep2]) < 63.9:
    logger.error("one repetition: first repetition feature is not 1")

if np.sum(bit_board[PlaneIndex.rep3]) > 0.1:
    logger.error("one repetition: second repetition feature is not 0")


# repeat the position twice
board.push_san("Nf3")
board.push_san("Nc6")
board.push_san("Ng1")
board.push_san("Nb8")
bit_board = board_representation.board_to_matrix(board)

if np.sum(bit_board[PlaneIndex.rep2]) < 63.9:
    logger.error("two repetition: first repetition feature is not 1")

if np.sum(bit_board[PlaneIndex.rep3]) < 63.9:
    logger.error("two repetition: second repetition feature is not 1")



########################################################################################
#                                   en passant                                         #
########################################################################################
board = chess.Board()
board.push_san("b4")
board.push_san("e5")
board.push_san("b5")
board.push_san("c5")
bit_board = board_representation.board_to_matrix(board)


# en passant square is c6
if bit_board[PlaneIndex.en_passant][2][2] < 0.9:
    logger.error("en passant square is not c6")




########################################################################################
#                             total moves played                                       #
########################################################################################
board = chess.Board()
board.push_san("d4")        # 1. move
board.push_san("f5")        # 1. move
board.push_san("c4")        # 2. move
board.push_san("Nf6")       # 2. move
board.push_san("g3")        # 3. move
board.push_san("e6")        # 3. move
board.push_san("Bg2")       # 4. move
bit_board = board_representation.board_to_matrix(board)

if np.sum(bit_board[PlaneIndex.tot_moves]) * CONST.MAX_TOTAL_MOVES < 3.9:
    logger.error("position is in the 4th move but the feature is not set to 5")




########################################################################################
#                                   castling rights                                    #
########################################################################################
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
bit_board = board_representation.board_to_matrix(board)


# white kingside castling
if np.sum(bit_board[PlaneIndex.white_castling_kingside]) < 63.9:
    logger.error("white can castle kingside: castle feature is not 1")

# white queenside castling
if np.sum(bit_board[PlaneIndex.white_castling_queenside]) < 63.9:
    logger.error("white can castle queenside: castle feature is not 1")

# black kingside castling
if np.sum(bit_board[PlaneIndex.black_castling_kingside]) < 63.9:
    logger.error("black can castle kingside: castle feature is not 1")

# black queenside castling
if np.sum(bit_board[PlaneIndex.black_castling_queenside]) < 63.9:
    logger.error("black can castle queenside: castle feature is not 1")


# move whites kingside rook and blacks queenside rook to give up castling rights
board.push_san("Rg1")
board.push_san("Rb8")
bit_board = board_representation.board_to_matrix(board)

# white kingside castling
if np.sum(bit_board[PlaneIndex.white_castling_kingside]) > 0.1:
    logger.error("white can not castle kingside: castle feature is not 0")

# white queenside castling
if np.sum(bit_board[PlaneIndex.white_castling_queenside]) < 63.9:
    logger.error("white can castle queenside: castle feature is not 1")

# black kingside castling
if np.sum(bit_board[PlaneIndex.black_castling_kingside]) < 63.9:
    logger.error("black can castle kingside: castle feature is not 1")

# black queenside castling
if np.sum(bit_board[PlaneIndex.black_castling_queenside]) > 0.1:
    logger.error("black can not castle queenside: castle feature is not 0")


# move the black king and the white queenside rook to lose all castling rights
board.push_san("Rb1")
board.push_san("Kf8")
bit_board = board_representation.board_to_matrix(board)


# white kingside castling
if np.sum(bit_board[PlaneIndex.white_castling_kingside]) > 0.1:
    logger.error("white can not castle kingside: castle feature is not 0")

# white queenside castling
if np.sum(bit_board[PlaneIndex.white_castling_queenside]) > 0.1:
    logger.error("white can not castle queenside: castle feature is not 0")

# black kingside castling
if np.sum(bit_board[PlaneIndex.black_castling_kingside]) > 0.1:
    logger.error("black can not castle kingside: castle feature is not 0")

# black queenside castling
if np.sum(bit_board[PlaneIndex.black_castling_queenside]) > 0.1:
    logger.error("black can not castle queenside: castle feature is not 0")




########################################################################################
#                            no of progress count                                      #
########################################################################################
board = chess.Board()
board.push_san("e4")            # 0 pawn moved
board.push_san("e5")            # 0 pawn moved
board.push_san("Nf3")           # 1 no pawn, no capture
board.push_san("Nc6")           # 2 no pawn, no capture
bit_board = board_representation.board_to_matrix(board)


# no progress count is two since the last two half moves no piece was captured and no pawn moved
if np.sum(bit_board[PlaneIndex.no_progress_count]) * CONST.MAX_PROGRESS_COUNTER < 1.9:
    logger.error("no progress count should be 2 but is smaller")


# make a pawn move and reset the counter
board.push_san("b4")
bit_board = board_representation.board_to_matrix(board)

# progress count should be 0
if np.sum(bit_board[PlaneIndex.no_progress_count]) * CONST.MAX_PROGRESS_COUNTER > 0.1:
    logger.error("no progress count should be 0 but is larger")


logger.info("board representation tests finished, there should be no error log so far")



# # test how long the board representation needs
# import time
# board = chess.Board()
# board.push_san("e4")
# board.push_san("e5")
# board.push_san("Nf3")
# board.push_san("Nc6")
# board.push_san("Bc4")
# board.push_san("Bc5")
# board.push_san("Qe2")
# board.push_san("d6")
# board.push_san("Nc3")
# board.push_san("Bd7")
# board.push_san("b3")
# board.push_san("Qe7")
# start = time.time()
# for i in range(10000):
#     mat = board_representation.board_to_matrix(board)
# print(time.time() - start)




# # test how fast random games can be played
# # create the chess board
# board = chess.Board()
#
# nGames = 100
#
# start = time.time()
# for _ in range(nGames):
#     board = chess.Board()
#     while not board.is_game_over():
#         legalMoves = board.legal_moves
#         randomMove = random.choice([move for move in board.legal_moves])
#         randomMove = randomMove.uci()
#         board.push_uci(randomMove)
#
# end = time.time()
# print("time for one random game: {}".format((end - start) / nGames))