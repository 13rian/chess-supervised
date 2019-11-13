import numpy as np
import chess

from globals import CONST


def int_to_board(number):
    """
    converts the passed integer number to the bitboard representation
    :param number:  integer number
    :return: bit board representation
    """
    byte_arr = np.array([number], dtype=np.uint64).view(np.uint8)
    board_mask = np.unpackbits(byte_arr).reshape(-1, 8)[::-1, ::-1]
    return board_mask


def board_to_matrix(board):
    bit_board = np.empty((22, 8, 8))

    # masks where the pieces are
    white_mask = board.occupied_co[chess.WHITE]
    black_mask = board.occupied_co[chess.BLACK]

    # white pieces
    bit_board[0] = int_to_board(board.pawns & white_mask)
    bit_board[1] = int_to_board(board.knights & white_mask)
    bit_board[2] = int_to_board(board.bishops & white_mask)
    bit_board[3] = int_to_board(board.rooks & white_mask)
    bit_board[4] = int_to_board(board.queens & white_mask)
    bit_board[5] = int_to_board(board.kings & white_mask)

    # black pieces
    bit_board[6] = int_to_board(board.pawns & black_mask)
    bit_board[7] = int_to_board(board.knights & black_mask)
    bit_board[8] = int_to_board(board.bishops & black_mask)
    bit_board[9] = int_to_board(board.rooks & black_mask)
    bit_board[10] = int_to_board(board.queens & black_mask)
    bit_board[11] = int_to_board(board.kings & black_mask)


    # repetitions 2 planes
    # first repetition
    if board.is_repetition(2):
        bit_board[12] = np.ones((8, 8))
    else:
        bit_board[12] = np.zeros((8, 8))

    # second repetition
    if board.is_repetition(3):
        bit_board[13] = np.ones((8, 8))
    else:
        bit_board[13] = np.zeros((8, 8))


    # en-passant: square where en-passant capture is possible
    if board.ep_square is not None:
        bit_board[14] = int_to_board(1 << board.ep_square)
    else:
        bit_board[14] = np.zeros((8, 8))


    # player color board
    if board.turn == chess.WHITE:
        bit_board[15] = np.ones((8, 8))
    else:
        bit_board[15] = np.zeros((8, 8))


    # total move count
    bit_board[16] = board.fullmove_number/CONST.MAX_TOTAL_MOVES * np.ones((8, 8))


    # white castling
    # kingside
    if board.has_kingside_castling_rights(chess.WHITE):
        bit_board[17] = np.ones((8, 8))
    else:
        bit_board[17] = np.zeros((8, 8))


    # queenside
    if board.has_queenside_castling_rights(chess.WHITE):
        bit_board[18] = np.ones((8, 8))
    else:
        bit_board[18] = np.zeros((8, 8))


    # black castling
    # kingside
    if board.has_kingside_castling_rights(chess.BLACK):
        bit_board[19] = np.ones((8, 8))
    else:
        bit_board[19] = np.zeros((8, 8))

    # queenside
    if board.has_queenside_castling_rights(chess.BLACK):
        bit_board[20] = np.ones((8, 8))
    else:
        bit_board[20] = np.zeros((8, 8))


    # no progress count
    bit_board[21] = board.halfmove_clock/CONST.MAX_PROGRESS_COUNTER * np.ones((8, 8))

    return bit_board