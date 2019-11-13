import logging
import random

import numpy as np

from utils import utils


#@utils.profile
def mainTrain():
    # The logger
    utils.init_logger(logging.DEBUG, file_name="log/chess_sl.log")
    logger = logging.getLogger('Chess_SL')

    # set the random seed
    # set the random seed
    random.seed(a=None, version=2)
    np.random.seed(seed=None)

    logger.debug("start the main test program")





if __name__ == '__main__':
    mainTrain()