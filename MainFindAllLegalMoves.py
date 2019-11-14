import copy

import chess



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

    print("queen + knight moves: ", len(possible_moves))


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

    print("queen + knight + white pawn promotion moves: ", len(possible_moves))

    # black pawn promotion
    first_rank = [chess.A1, chess.B1, chess.C1, chess.D1, chess.E1, chess.F1, chess.G1, chess.H1]
    second_rank = [chess.A2, chess.B2, chess.C2, chess.D2, chess.E2, chess.F2, chess.G2, chess.H2]

    # simple promotion
    pawn = chess.Piece(chess.PAWN, chess.BLACK)
    for square in second_rank:
        board = chess.Board().empty()
        board.set_piece_at(square, pawn)
        board.turn = chess.BLACK

        for move in board.legal_moves:
            possible_moves.append(str(move))

    # promotion with a capture
    knight = chess.Piece(chess.KNIGHT, chess.WHITE)
    knight_board = chess.Board().empty()
    for square in first_rank:
        knight_board.set_piece_at(square, knight)

    for square in second_rank:
        board = copy.deepcopy(knight_board)
        board.set_piece_at(square, pawn)
        board.turn = chess.BLACK

        for move in board.legal_moves:
            possible_moves.append(str(move))


    # sort the list
    possible_moves.sort()

    return possible_moves



possible_moves = __all_moves__()

print(possible_moves)
print("length of all legal moves: ", len(possible_moves))



