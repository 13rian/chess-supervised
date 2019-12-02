import matplotlib.pyplot as plt
from torch.utils import data
import random
import numpy as np
import logging
import torch


from utils import utils
from globals import Config
import networks
import data_processing
import data_storage


# @utils.profile
def mainTrain():
    # The logger
    utils.init_logger(logging.DEBUG, file_name="log/chess_sl.log.log")
    logger = logging.getLogger('Sup Learning')

    np.set_printoptions(suppress=True, precision=6)


    # set the random seed
    random.seed(a=None, version=2)
    np.random.seed(seed=None)


    # parameters
    Config.learning_rate = 0.001
    Config.weight_decay = 1e-4
    Config.n_blocks = 10
    Config.n_filters = 128
    epochs = 1
    training_set_path = "king-base-light-full.h5"

    # define the parameters for the training
    params = {'batch_size': 512,
              'shuffle': True,
              'num_workers': 1}


    # create the data set class
    training_set = data_processing.Dataset(training_set_path)
    training_generator = data.DataLoader(training_set, **params)
    logger.info("training set created, length: {}".format(training_set.__len__()))




    # create a new network to train
    network = networks.ResNet(Config.learning_rate, Config.n_blocks, Config.n_filters, Config.weight_decay)
    network = data_storage.net_to_device(network, Config.training_device)


    # list for the plots
    epoch_list = []
    value_loss = []
    policy_loss = []


    # execute the training by looping over all epochs
    network.train()
    for epoch in range(epochs):
        epoch_list.append(epoch)
        avg_loss_p = 0
        avg_loss_v = 0
        tot_batch_count = 0

        # training
        for state_batch, policy_batch, value_batch in training_generator:
            # send the data to the gpu
            state_batch = state_batch.to(Config.training_device, dtype=torch.float)
            value_batch = value_batch.unsqueeze(1).to(Config.training_device, dtype=torch.float)
            policy_batch = policy_batch.to(Config.training_device, dtype=torch.float)

            # execute one training step
            loss_p, loss_v = network.train_step(state_batch, policy_batch, value_batch)
            avg_loss_p += loss_p
            avg_loss_v += loss_v
            tot_batch_count += 1

            if tot_batch_count % 1 == 0:
                logger.debug("epoch {}: trained {} batches so far".format(epoch, tot_batch_count))
                if tot_batch_count == 10:
                    break

        logger.debug("epoch {}: finished training".format(epoch))


if __name__ == '__main__':
    mainTrain()
