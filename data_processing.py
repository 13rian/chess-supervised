import logging
import os

import chess.pgn
import tables
import numpy as np

import board_representation
from globals import CONST


# The logger
logger = logging.getLogger('DataProc')


pgn_dir = "pgns"                            # directory that contains all the pgn files for the data set
data_set_file = 'king-base-light.h5'        # name of the output file


# row size is number of channels + policy + value
example_size = CONST.INPUT_CHANNELS * CONST.BOARD_HEIGHT * CONST.BOARD_WIDTH + board_representation.LABEL_COUNT + 1


def create_data_set():
    """
    creates the data set from all games in the pgn folder
    the data set is in hdf5 format and compressed because most of the matrices are sparse
    :return:
    """
    # create a new data file (compression level needs to be between 0 and 9, 0 is no compression and the fastest)
    # 6 seemed to be the best time / size trade off. at least 1 is necessary otherwise the file size will explode
    # the zlib algorithm is lossless
    compression_filter = tables.Filters(complib='zlib', complevel=1)
    file = tables.open_file(data_set_file, mode='w', filters=compression_filter)
    atom = tables.Float64Atom()
    array_c = file.create_earray(file.root, 'data', atom, (0, example_size))


    # add load all pgn files and add the games to the data set
    path_list = os.listdir(pgn_dir)
    game_count = 0
    for pgn_file_name in path_list:
        pgn_file_path = pgn_dir + "/" + pgn_file_name
        pgn_file = open(pgn_file_path)
        logger.info("start to process file {}".format(pgn_file_name))


        # read out all games in the pgn file
        game = chess.pgn.read_game(pgn_file)  # read out the next game from the pgn
        while game is not None:
            board = chess.Board()               # create a new board

            # get the value of the game
            result = value_from_result(game.headers["Result"])
            if result is None:
                game = chess.pgn.read_game(pgn_file)  # read out the next game from the pgn
                continue

            # go through all moves and append the data to the data file
            for move in game.mainline_moves():
                try:
                    network_input = board_representation.board_to_matrix(board)
                    network_input = network_input.flatten()
                    policy = board_representation.move_to_policy(move, board.turn)
                    value = np.array([result]) if board.turn == chess.WHITE else np.array([-result])

                    training_example = np.concatenate((network_input, policy, value))
                    training_example = np.expand_dims(training_example, axis=0)
                    array_c.append(training_example)

                    # make the move to get the next board position
                    board.push(move)

                except Exception as e:
                    # ignore the rest of the game if an error occurs
                    logging.error("error in the current game: ", exc_info=True)
                    continue

            game = chess.pgn.read_game(pgn_file)  # read out the next game from the pgn

            game_count += 1
            if game_count % 1000 == 0:
                logger.debug("processed {} games".format(game_count))

        pgn_file.close()

    logger.info("total number of games processed: {}".format(game_count))
    file.close()


    # # read out some data
    # f = tables.open_file(data_set_file, mode='r', filters=compression_filter)
    # # print(f.root.data[1:10, 2:20])  # slicing is possible
    # start = time.time()
    # for i in range(10000):
    #     arr = f.root.data[i, :]
    # print("read-time: ", time.time() - start)
    # f.close()


def value_from_result(result):
    """
    returns the value of a chess game result
    1-0:        white won
    1/2-1/2:    draw
    0-1:        black won
    :param result:
    :return:
    """
    if result == "1-0":
        return 1

    if result == "1/2-1/2":
        return 0

    if result == "0-1":
        return -1

    logger.error("result string no recognized: {}".format(result))