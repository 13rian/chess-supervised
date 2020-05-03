from pathlib import Path

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
    epochs = 3
    training_set_path = "king-base-light-avg.h5"
    network_dir = "networks"
    training_progress_dir = "training_progress"


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


    # create all needed folders
    Path(network_dir).mkdir(parents=True, exist_ok=True)
    Path(training_progress_dir).mkdir(parents=True, exist_ok=True)


    # list for the plots
    batches = []
    policy_loss = []
    value_loss = []
    tot_batch_count = 0
    current_batch_count = 0
    current_value_loss = 0
    current_policy_loss = 0

    # execute the training by looping over all epochs
    for epoch in range(epochs):
        # training
        for state_batch, policy_batch, value_batch in training_generator:
            # send the data to the gpu
            state_batch = state_batch.to(Config.training_device, dtype=torch.float)
            value_batch = value_batch.unsqueeze(1).to(Config.training_device, dtype=torch.float)
            policy_batch = policy_batch.to(Config.training_device, dtype=torch.float)

            # execute one training step
            loss_p, loss_v = network.train_step(state_batch, policy_batch, value_batch)
            current_policy_loss += loss_p
            current_value_loss += loss_v
            current_batch_count += 1
            tot_batch_count += 1

            if tot_batch_count % 100 == 0:
                logger.debug("epoch {}: trained {} batches so far".format(epoch, tot_batch_count))
                batches.append(tot_batch_count)
                policy_loss.append(current_policy_loss / current_batch_count)
                value_loss.append(current_value_loss / current_batch_count)

                current_policy_loss = 0
                current_value_loss = 0
                current_batch_count = 0

                if tot_batch_count % 1000 == 0:
                    network_path = "{}/network_batch_{}.pt".format(network_dir, tot_batch_count)
                    torch.save(network, network_path)

                    np.save(training_progress_dir + "/value_loss.npy", np.array(value_loss))
                    np.save(training_progress_dir + "/policy_loss.npy", np.array(policy_loss))
                    np.save(training_progress_dir + "/batches.npy", np.array(batches))

        # save the last network
        network_path = "{}/network_batch_{}.pt".format(network_dir, tot_batch_count)
        torch.save(network, network_path)

        np.save(training_progress_dir + "/value_loss.npy", np.array(value_loss))
        np.save(training_progress_dir + "/policy_loss.npy", np.array(policy_loss))
        np.save(training_progress_dir + "/batches.npy", np.array(batches))

        logger.debug("epoch {}: finished training".format(epoch))

    # plot the loss versus the number of seen batches
    # plot the value training loss
    fig1 = plt.figure(1)
    plt.plot(batches, value_loss)
    axes = plt.gca()
    axes.grid(True, color=(0.9, 0.9, 0.9))
    plt.title("Average Value Training Loss")
    plt.xlabel("Training Samples")
    plt.ylabel("Value Loss")
    fig1.show()

    # plot the training policy loss
    fig2 = plt.figure(2)
    plt.plot(batches, policy_loss)
    axes = plt.gca()
    axes.grid(True, color=(0.9, 0.9, 0.9))
    plt.title("Average Policy Training Loss")
    plt.xlabel("Training Samples")
    plt.ylabel("Policy Loss")
    fig2.show()

    plt.show()


if __name__ == '__main__':
    mainTrain()
