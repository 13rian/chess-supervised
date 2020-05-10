import logging
import time

from utils import utils
import data_processing


# @utils.profile
def main():
    # The logger
    utils.init_logger(logging.DEBUG, file_name="log/chess_sl.log")
    logger = logging.getLogger('Chess_SL')

    pgn_dir = "pgns"                                # dir containing all the pgn files
    db_file = 'fen_positions.db'                    # name of the sql data set file
    data_set_file = 'king-base-light-raw.h5'        # name of the raw dataset file
    data_set_file_avg = 'king-base-light-avg.h5'    # name of the output file


    # create the fen_dict in order to average the positions
    start = time.time()
    data_processing.create_fen_db(db_file, pgn_dir)
    # data_processing.create_fen_dict()
    elapsed_time = time.time() - start
    logger.info("time to create the db file: {}".format(elapsed_time))


    # create the averaged data set
    start = time.time()
    data_processing.create_averaged_data_set(db_file, data_set_file_avg)
    elapsed_time = time.time() - start
    logger.info("time to create the averaged data set: {}".format(elapsed_time))




    # # create the data set from the pgn files
    # start = time.time()
    # data_processing.create_data_set(data_set_file)
    # elapsed_time = time.time() - start
    logger.info("time to create the data set from the pgn files: {}".format(elapsed_time))


main()
