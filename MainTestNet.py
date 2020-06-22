from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import random
import numpy as np
import logging
import torch
import chess.variant

from utils import utils
import board_representation
import data_processing


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
    network_dir = "networks"
    network_file = network_dir + "/" + "network_batch_158436.pt"
    training_progress_dir = "training_progress"


    # count the games in the database
    dict_file = "elo_dict.pkl"
    if not Path(dict_file).is_file():
        elo_dict = data_processing.create_elo_dict("pgns")

        with open(dict_file, 'wb') as f:
            pickle.dump(elo_dict, f, pickle.HIGHEST_PROTOCOL)

    with open(dict_file, 'rb') as f:
        elo_dict = pickle.load(f)
        elo = []
        count = []
        tot_count = 0
        for key in sorted(elo_dict):
            elo.append(int(key))
            value = elo_dict[key]
            tot_count += value
            count.append(tot_count)

    # plot the training policy loss
    fig1 = plt.figure(1)
    plt.plot(elo, count)
    axes = plt.gca()
    axes.grid(True, color=(0.9, 0.9, 0.9))
    plt.title("Dataset Statistics")
    plt.xlabel("Minimal ELO")
    plt.ylabel("Total Number of Games")
    fig1.show()
    plt.show()



    # load the network
    logger.info("load the neural network: " + network_file)
    net = torch.load(network_file, map_location='cpu')


    board = chess.Board()

    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")

    bit_board = board_representation.board_to_matrix(board)

    policy, value = net(torch.Tensor(bit_board).unsqueeze(0))
    print(policy)
    print(value)

    print("move: ", board_representation.policy_to_move(policy.detach().numpy(), board.turn))

    print(board.legal_moves)





    # plot the learning progress
    value_loss = np.load(training_progress_dir + "/value_loss.npy")
    policy_loss = np.load(training_progress_dir + "/policy_loss.npy")
    batches = np.load(training_progress_dir + "/batches.npy")

    # plot the loss versus the number of seen batches
    # plot the value training loss
    fig2 = plt.figure(2)
    plt.plot(batches, value_loss)
    axes = plt.gca()
    axes.grid(True, color=(0.9, 0.9, 0.9))
    plt.title("Average Value Training Loss")
    plt.xlabel("Training Samples")
    plt.ylabel("Value Loss")
    fig2.show()

    # plot the training policy loss
    fig3 = plt.figure(3)
    plt.plot(batches, policy_loss)
    axes = plt.gca()
    axes.grid(True, color=(0.9, 0.9, 0.9))
    plt.title("Average Policy Training Loss")
    plt.xlabel("Training Samples")
    plt.ylabel("Policy Loss")
    fig3.show()

    plt.show()




if __name__ == '__main__':
    mainTrain()
