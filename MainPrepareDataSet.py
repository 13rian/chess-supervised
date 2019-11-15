import logging
import time

from utils import utils
import data_processing


# @utils.profile
def main():
    # The logger
    utils.init_logger(logging.DEBUG, file_name="log/chess_sl.log")
    logger = logging.getLogger('Chess_SL')

    # create the data set from the pgn files
    start = time.time()
    data_processing.create_data_set()
    elapsed_time = time.time() - start

    logger.info("time to create the data set from the pgn files: {}".format(elapsed_time))


main()
