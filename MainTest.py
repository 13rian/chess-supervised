import logging
import random

from utils import utils


#@utils.profile
def mainTrain():
    # The logger
    utils.init_logger(logging.DEBUG, file_name="log/chess_sl.log")
    logger = logging.getLogger('Chess_SL')

    # set the random seed
    random.seed(a=None, version=2)

    logger.debug("start the main test program")





if __name__ == '__main__':
    mainTrain()