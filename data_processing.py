import logging
import os
import sqlite3

import chess.pgn
import tables
import numpy as np
import pickle
from torch.utils import data

import board_representation
from globals import CONST


# The logger
logger = logging.getLogger('DataProc')



# row size is number of channels + policy + value
full_example_size = CONST.INPUT_CHANNELS * CONST.BOARD_HEIGHT * CONST.BOARD_WIDTH + 1 + 1
avg_example_size = CONST.INPUT_CHANNELS * CONST.BOARD_HEIGHT * CONST.BOARD_WIDTH + board_representation.LABEL_COUNT + 1


def create_fen_db(db_file, pgn_dir, elo_threshold=0):
    """
    saves all positions in a sqlite db, moves and results are added to the same position with _
    as a delimiter
    :param db_file:         the file path to the db file to create
    :param pgn_dir:         the directory from which all the pgns are read
    :param elo_threshold:   minimal elo of the weaker player
    :return:
    """
    # check if the file already exists
    if os.path.exists(db_file):
        logger.debug("db file {} already exists".format(db_file))
        return

    # connect to the db file
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

    cursor.execute(sql_create_table)            # create the table
    cursor.execute(sql_create_index)            # create the index
    connection.commit()


    # add load all pgn files and add the games to the data set
    path_list = os.listdir(pgn_dir)
    game_count = 0
    for pgn_file_name in path_list:
        if not pgn_file_name.endswith(".pgn"):
            continue

        pgn_file_path = pgn_dir + "/" + pgn_file_name
        pgn_file = open(pgn_file_path)
        logger.info("start to process file {}".format(pgn_file_name))


        # read out all games in the pgn file
        game = chess.pgn.read_game(pgn_file)  # read out the next game from the pgn
        while game is not None:
            # skip games below elo threshold
            min_elo = min(game.headers["WhiteElo"], game.headers["BlackElo"])
            if int(min_elo) < elo_threshold:
                game = chess.pgn.read_game(pgn_file)  # read out the next game from the pgn
                continue

            # get the value of the game
            result = value_from_result(game.headers["Result"])
            if result is None:
                game = chess.pgn.read_game(pgn_file)  # read out the next game from the pgn
                continue

            # go through all moves and append the data to the data file
            board = chess.Board()  # create a new board
            for move in game.mainline_moves():
                try:
                    fen = board.fen()
                    move_idx = board_representation.move_to_index(move, board.turn)

                    # define the db update
                    position = (fen, '')
                    pos_update = ('_' + str(move_idx), result, fen)

                    cur = connection.cursor()
                    cur.execute(sql_position_insert, position)
                    cur.execute(sql_position_update, pos_update)
                    # connection.commit()                             # commit takes a lot of time

                    # make the move to get the next board position
                    board.push(move)

                except Exception as e:
                    # ignore the rest of the game if an error occurs
                    logging.error("error in the current game: ", exc_info=True)
                    continue

            game = chess.pgn.read_game(pgn_file)  # read out the next game from the pgn

            game_count += 1

            # commit the sql commands to the db file
            if game_count % 100 == 0:
                connection.commit()

            if game_count % 5000 == 0:
                connection.commit()
                logger.debug("processed {} games".format(game_count))

        # commit the last few games
        connection.commit()

        pgn_file.close()

    logger.info("total number of games processed: {}".format(game_count))



def create_averaged_data_set(db_file, data_set_file):
    """
    creates the averaged hdf5 file with all positions
    :param db_file:         the file path of the db file
    :param data_set_file:   the file path of the hdf5 file to create
    :return:
    """
    # check if the file already exists
    if os.path.exists(data_set_file):
        logger.debug("data set file {} already exists".format(data_set_file))
        return


    # create a new data file (compression level needs to be between 0 and 9, 0 is no compression and the fastest)
    # 6 seemed to be the best time / size trade off. at least 1 is necessary otherwise the file size will explode
    # the zlib algorithm is lossless
    compression_filter = tables.Filters(complib='zlib', complevel=1)
    file = tables.open_file(data_set_file, mode='w', filters=compression_filter)
    atom = tables.Float64Atom()
    array_c = file.create_earray(file.root, 'data', atom, (0, avg_example_size))


    # open a connection to the db file
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()


    # query all positions
    cursor.execute("SELECT * FROM position;")
    row = cursor.fetchone()

    position_count = 0
    while row is not None:
        try:
            # create the board from the fen string
            fen = row[1]
            board = chess.Board(fen=fen)

            # create the neural network input of the state
            network_input = board_representation.board_to_matrix(board)
            network_input = network_input.flatten()

            # create the policy
            policy = np.zeros(board_representation.LABEL_COUNT)
            move_indices = row[2].strip('_').split("_")
            for move_idx in move_indices:
                policy[int(move_idx)] += 1

            # normalize the policy
            policy /= row[4]


            # calculate the value from the networks perspective
            value = row[3] / row[4]
            value = np.array([value]) if board.turn == chess.WHITE else np.array([-value])


            # add the training example to the hdf5 file
            training_example = np.concatenate((network_input, policy, value))
            training_example = np.expand_dims(training_example, axis=0)
            array_c.append(training_example)


            # log the progress
            position_count += 1
            if position_count % 150000 == 0:
                logger.debug("processed {} positions".format(position_count))


            # fetch the next fow
            row = cursor.fetchone()


        except Exception as e:
            # ignore the current row if an error occurred
            logging.error("error processing the current row: ", exc_info=True)
            continue


    logger.info("total number of processed positions: {}".format(position_count))
    file.close()





def create_data_set(data_set_file, pgn_dir):
    """
    creates the data set from all games in the pgn folder
    the data set is in hdf5 format and compressed because most of the matrices are sparse
    :param data_set_file:         path of the hdf5 file to create
    :return:
    """
    # check if the file already exists
    if os.path.exists(data_set_file):
        logger.debug("data set file {} already exists".format(data_set_file))
        return


    # create a new data file (compression level needs to be between 0 and 9, 0 is no compression and the fastest)
    # 6 seemed to be the best time / size trade off. at least 1 is necessary otherwise the file size will explode
    # the zlib algorithm is lossless
    compression_filter = tables.Filters(complib='zlib', complevel=1)
    file = tables.open_file(data_set_file, mode='w', filters=compression_filter)
    atom = tables.Float64Atom()
    array_c = file.create_earray(file.root, 'data', atom, (0, full_example_size))


    # load all pgn files and add the games to the data set
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
                    move_idx = np.array([board_representation.move_to_index(move, board.turn)])
                    # policy = board_representation.move_to_policy(move, board.turn)
                    value = np.array([result]) if board.turn == chess.WHITE else np.array([-result])

                    training_example = np.concatenate((network_input, move_idx, value))
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
            if game_count % 5000 == 0:
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



# def create_fen_dict():
#     """
#     creates a dict with the fen notation as key and all moves played in a list
#     :return:
#     """
#     fen_dict = {}
#
#     # add load all pgn files and add the games to the data set
#     path_list = os.listdir(pgn_dir)
#     game_count = 0
#     for pgn_file_name in path_list:
#         pgn_file_path = pgn_dir + "/" + pgn_file_name
#         pgn_file = open(pgn_file_path)
#         logger.info("start to process file {}".format(pgn_file_name))
#
#
#         # read out all games in the pgn file
#         game = chess.pgn.read_game(pgn_file)  # read out the next game from the pgn
#         while game is not None:
#             board = chess.Board()               # create a new board
#
#             # get the value of the game
#             result = value_from_result(game.headers["Result"])
#             if result is None:
#                 game = chess.pgn.read_game(pgn_file)  # read out the next game from the pgn
#                 continue
#
#             # go through all moves and append the data to the data file
#             for move in game.mainline_moves():
#                 try:
#                     move_idx = board_representation.move_index(move, board.turn)
#                     "-" + str(move_idx)
#                     fen = board.fen()
#                     if fen in fen_dict:
#                         fen_dict[fen][0].append(move_idx)
#                         fen_dict[fen][1].append(result)
#
#                     else:
#                         fen_dict[fen] = ([move_idx], [result])
#
#                     # make the move to get the next board position
#                     board.push(move)
#
#                 except Exception as e:
#                     # ignore the rest of the game if an error occurs
#                     logging.error("error in the current game: ", exc_info=True)
#                     continue
#
#             game = chess.pgn.read_game(pgn_file)  # read out the next game from the pgn
#
#             game_count += 1
#             if game_count % 5000 == 0:
#                 # save the dict
#                 with open(fen_dict_file, 'wb') as output:
#                     pickle.dump(fen_dict, output, pickle.HIGHEST_PROTOCOL)
#
#                 logger.debug("processed {} games".format(game_count))
#
#         pgn_file.close()
#
#     logger.info("total number of games processed: {}".format(game_count))
#
#     # save the dict
#     with open(fen_dict_file, 'wb') as output:
#         pickle.dump(fen_dict, output, pickle.HIGHEST_PROTOCOL)



def create_elo_dict(pgn_dir):
    """
    creates a dict that contains the min elo as key and the count as value
    :param pgn_dir:         the directory containing all the pgn files
    :return:
    """
    elo_dict = {}
    for pgn_file_name in os.listdir(pgn_dir):
        if not pgn_file_name.endswith(".pgn"):
            continue

        pgn_file_path = pgn_dir + "/" + pgn_file_name
        pgn_file = open(pgn_file_path)
        print("start to process file {}".format(pgn_file_name))

        # read out all games in the pgn file
        game = chess.pgn.read_game(pgn_file)  # read out the next game from the pgn
        while game is not None:
            min_elo = min(game.headers["WhiteElo"], game.headers["BlackElo"])
            if min_elo in elo_dict.keys():
                elo_dict[min_elo] += 1
            else:
                elo_dict[min_elo] = 1

            game = chess.pgn.read_game(pgn_file)  # read out the next game from the pgn

        pgn_file.close()

    return elo_dict



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

    logger.error("result string not recognized: {}".format(result))


def get_compression_filter():
    """
    returns the compression filter to write and read the hdf5 file
    :return:
    """
    compression_filter = tables.Filters(complib='zlib', complevel=1)
    return compression_filter


class Dataset(data.Dataset):
    def __init__(self, file_path):
        """
        data set for the neural network training
        :param file_path:      path to the data set file
        """
        self.file_path = file_path
        self.data_file = None

        data_file = self.open_data_file()
        self.size = data_file.root.data.shape[0]
        data_file.close()

    def __len__(self):
        return self.size


    def __getitem__(self, index):
        """
        returns a sample with the passed index
        :param index:   index of the sample
        :return:        state, policy, value
        """
        if self.data_file is None:
            self.data_file = self.open_data_file()

        state = self.data_file.root.data[index, 0:CONST.STATE_SIZE].reshape(CONST.INPUT_CHANNELS, CONST.BOARD_HEIGHT, CONST.BOARD_WIDTH)
        policy = self.data_file.root.data[index, CONST.STATE_SIZE:-1]
        value = self.data_file.root.data[index, -1]

        return state, policy, value


    def open_data_file(self):
        """
        opens the file with the data set
        :return:
        """
        return tables.open_file(self.file_path, mode='r', filters=get_compression_filter())




#############################################################################################################
#                                        sql definitions                                                    #
#############################################################################################################
sql_create_table = """CREATE TABLE IF NOT EXISTS position (
                        id integer PRIMARY KEY,
                        fen text,
                        moves text,
                        result integer,
                        count integer
                    );"""

# create the index of the fen
sql_create_index = """CREATE UNIQUE INDEX IF NOT EXISTS idx_position_fen ON position(fen);"""

# insert a new fen position if it does not exist
sql_position_insert = """INSERT OR IGNORE INTO position(fen,moves,result,count)
                         VALUES(?,?,0,0);"""

# update an existing fen position
sql_position_update = """UPDATE position SET 
                            moves = moves || ?,
                            result = result + ?,
                            count = count + 1
                         WHERE fen = ?;"""