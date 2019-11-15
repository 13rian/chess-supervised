import copy

import numpy as np
import chess

from globals import CONST



########################################################################################################
#                                   output representation (policy)                                     #
########################################################################################################
def __all_moves__():
    """
    returns a list of all possible chess moves
    :return:
    """
    possible_moves = []

    # all queen moves
    queen = chess.Piece(chess.QUEEN, chess.WHITE)
    for square in chess.SQUARES:
        board = chess.Board().empty()
        board.set_piece_at(square, queen)

        for move in board.legal_moves:
            possible_moves.append(str(move))


    # all knight moves
    knight = chess.Piece(chess.KNIGHT, chess.WHITE)
    for square in chess.SQUARES:
        board = chess.Board().empty()
        board.set_piece_at(square, knight)

        for move in board.legal_moves:
            possible_moves.append(str(move))


    # white pawn promotion
    back_rank = [chess.A8, chess.B8, chess.C8, chess.D8, chess.E8, chess.F8, chess.G8, chess.H8]
    seventh_rank = [chess.A7, chess.B7, chess.C7, chess.D7, chess.E7, chess.F7, chess.G7, chess.H7]

    # simple promotion
    pawn = chess.Piece(chess.PAWN, chess.WHITE)
    for square in seventh_rank:
        board = chess.Board().empty()
        board.set_piece_at(square, pawn)

        for move in board.legal_moves:
            possible_moves.append(str(move))

    # promotion with a capture
    knight = chess.Piece(chess.KNIGHT, chess.BLACK)
    knight_board = chess.Board().empty()
    for square in back_rank:
        knight_board.set_piece_at(square, knight)

    for square in seventh_rank:
        board = copy.deepcopy(knight_board)
        board.set_piece_at(square, pawn)

        for move in board.legal_moves:
            possible_moves.append(str(move))


    # # black pawn promotion
    # first_rank = [chess.A1, chess.B1, chess.C1, chess.D1, chess.E1, chess.F1, chess.G1, chess.H1]
    # second_rank = [chess.A2, chess.B2, chess.C2, chess.D2, chess.E2, chess.F2, chess.G2, chess.H2]
    #
    # # simple promotion
    # pawn = chess.Piece(chess.PAWN, chess.BLACK)
    # for square in second_rank:
    #     board = chess.Board().empty()
    #     board.set_piece_at(square, pawn)
    #     board.turn = chess.BLACK
    #
    #     for move in board.legal_moves:
    #         possible_moves.append(str(move))
    #
    # # promotion with a capture
    # knight = chess.Piece(chess.KNIGHT, chess.WHITE)
    # knight_board = chess.Board().empty()
    # for square in first_rank:
    #     knight_board.set_piece_at(square, knight)
    #
    # for square in second_rank:
    #     board = copy.deepcopy(knight_board)
    #     board.set_piece_at(square, pawn)
    #     board.turn = chess.BLACK
    #
    #     for move in board.legal_moves:
    #         possible_moves.append(str(move))

    # sort the list
    possible_moves.sort()

    return possible_moves


def mirror_move(move):
    """
    mirrors the passed chess move
    :param move:    chess move object
    :return:        mirrored move
    """
    start_square = chess.square_mirror(move.from_square)
    end_square = chess.square_mirror(move.to_square)
    return chess.Move(start_square, end_square, move.promotion, move.drop)



def white_move_labels():
    """
    returns a list with all possible chess moves
    :return:
    """
    return __all_moves__()


# def black_move_labels():
#     """
#     returns a list with all possible chess moves mirrored
#     :return:
#     """
#     white_moves = __all_moves__()
#
#     # mirror all the white moves
#     black_moves = [None] * len(white_moves)
#     for i, label in enumerate(white_moves):
#         mv = chess.Move.from_uci(label)
#         mv_mirrored = mirror_move(mv)
#         black_moves[i] = mv_mirrored.uci()
#
#     return black_moves


def labels_to_lookup_table(labels):
    """
    returns a lookup table with the move as key and the policy index as value
    :param labels:      the move labels
    :return:
    """
    lookup_table = {}
    for i, label in enumerate(labels):
        lookup_table[label] = i

    return lookup_table


# move labels
WHITE_MOVE_LABELS = white_move_labels()  # labels for all white moves
# BLACK_MOVE_LABLES = black_move_labels()  # labels for all black moves

LABEL_COUNT = len(WHITE_MOVE_LABELS)     # total number of labels


# lookup tables to find the move indices, key: uci move, value: policy index
WHITE_MOVE_TABLE = labels_to_lookup_table(WHITE_MOVE_LABELS)
# BLACK_MOVE_TABLE = labels_to_lookup_table(BLACK_MOVE_LABLES)



# def move_to_policy(move, player=CONST.WHITE):
#     """
#     converts the passed chess move to a one hot encoded policy vector
#     :param move:    chess move object
#     :param player:  white or black player
#     :return:        one hot vector that defines the policy for this move
#     """
#
#     if player == CONST.WHITE:
#         policy_index = WHITE_MOVE_TABLE[move.uci()]
#     else:
#         policy_index = BLACK_MOVE_TABLE[move.uci()]
#
#     policy = np.zeros(LABLE_COUNT, dtype=np.bool)
#     policy[policy_index] = 1
#
#     return policy


# def policy_to_move(policy, player=CONST.WHITE):
#     """
#     converts the passed move policy vector to a chess move, the move with the highest probability
#     is chosen
#     :param policy:      move policy vector
#     :param player:      white or black player
#     :return:            chess move object
#     """
#     # get the move with the highest probability
#     move_index = np.argmax(policy)
#
#     if player == CONST.WHITE:
#         uci_move = WHITE_MOVE_LABLES[move_index]
#     else:
#         uci_move = BLACK_MOVE_LABLES[move_index]
#
#     return chess.Move.from_uci(uci_move)


def move_to_policy(move, turn=chess.WHITE):
    """
    converts the passed chess move to a one hot encoded policy vector
    the move is always viewed from the white perspective
    :param move:    chess move object
    :param turn:    indicates if it is white's or black's turn
    :return:        one hot vector that defines the policy for this move
    """
    # mirror the the moves for the black player since the board is always viewed from the white perspective
    if turn == chess.BLACK:
        move = mirror_move(move)

    policy_index = WHITE_MOVE_TABLE[move.uci()]
    policy = np.zeros(LABEL_COUNT)
    policy[policy_index] = 1

    return policy


def policy_to_move(policy):
    """
    converts the passed move policy vector to a chess move, the move with the highest probability
    is chosen. the move is always viewed from the white perspective
    :param policy:      move policy vector
    :return:            chess move object
    """
    # get the move with the highest probability
    move_index = np.argmax(policy)
    uci_move = WHITE_MOVE_LABELS[move_index]
    return chess.Move.from_uci(uci_move)




########################################################################################################
#                                   input representation (chess board)                                 #
########################################################################################################
def int_to_board(number):
    """
    converts the passed integer number to the bitboard representation
    :param number:  integer number
    :return:        bit board representation
    """
    byte_arr = np.array([number], dtype=np.uint64).view(np.uint8)
    board_mask = np.unpackbits(byte_arr).reshape(-1, 8)[::-1, ::-1]
    return board_mask


def board_to_matrix(board):
    """
    creates the matrix board representation that is used as input for the neural network
    the position is always viewed from the white perspective
    :param board:   chess board
    :return:        matrix that is used for the neural network input
    """
    # mirror the board for the black player
    if board.turn == chess.BLACK:
        board = board.mirror()      # operation is not inplace (leaves the original board untouched)

    bit_board = np.empty((CONST.INPUT_CHANNELS, CONST.BOARD_HEIGHT, CONST.BOARD_WIDTH))

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
        bit_board[12] = np.ones((CONST.BOARD_HEIGHT, CONST.BOARD_WIDTH))
    else:
        bit_board[12] = np.zeros((CONST.BOARD_HEIGHT, CONST.BOARD_WIDTH))

    # second repetition
    if board.is_repetition(3):
        bit_board[13] = np.ones((CONST.BOARD_HEIGHT, CONST.BOARD_WIDTH))
    else:
        bit_board[13] = np.zeros((CONST.BOARD_HEIGHT, CONST.BOARD_WIDTH))


    # en-passant: square where en-passant capture is possible
    if board.ep_square is not None:
        bit_board[14] = int_to_board(1 << board.ep_square)
    else:
        bit_board[14] = np.zeros((CONST.BOARD_HEIGHT, CONST.BOARD_WIDTH))


    # player color board
    if board.turn == chess.WHITE:
        bit_board[15] = np.ones((CONST.BOARD_HEIGHT, CONST.BOARD_WIDTH))
    else:
        bit_board[15] = np.zeros((CONST.BOARD_HEIGHT, CONST.BOARD_WIDTH))


    # total move count
    bit_board[16] = board.fullmove_number/CONST.MAX_TOTAL_MOVES * np.ones((CONST.BOARD_HEIGHT, CONST.BOARD_WIDTH))


    # white castling
    # kingside
    if board.has_kingside_castling_rights(chess.WHITE):
        bit_board[17] = np.ones((CONST.BOARD_HEIGHT, CONST.BOARD_WIDTH))
    else:
        bit_board[17] = np.zeros((CONST.BOARD_HEIGHT, CONST.BOARD_WIDTH))


    # queenside
    if board.has_queenside_castling_rights(chess.WHITE):
        bit_board[18] = np.ones((CONST.BOARD_HEIGHT, CONST.BOARD_WIDTH))
    else:
        bit_board[18] = np.zeros((CONST.BOARD_HEIGHT, CONST.BOARD_WIDTH))


    # black castling
    # kingside
    if board.has_kingside_castling_rights(chess.BLACK):
        bit_board[19] = np.ones((CONST.BOARD_HEIGHT, CONST.BOARD_WIDTH))
    else:
        bit_board[19] = np.zeros((CONST.BOARD_HEIGHT, CONST.BOARD_WIDTH))

    # queenside
    if board.has_queenside_castling_rights(chess.BLACK):
        bit_board[20] = np.ones((CONST.BOARD_HEIGHT, CONST.BOARD_WIDTH))
    else:
        bit_board[20] = np.zeros((CONST.BOARD_HEIGHT, CONST.BOARD_WIDTH))


    # no progress count
    bit_board[21] = board.halfmove_clock/CONST.MAX_PROGRESS_COUNTER * np.ones((CONST.BOARD_HEIGHT, CONST.BOARD_WIDTH))

    return bit_board