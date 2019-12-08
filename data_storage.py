import logging
import os
import shutil

import torch

from globals import Config
import networks

logger = logging.getLogger('Data Storage')
temp_dir = "temp_net"


def net_to_device(net, device):
    """
    sends the network to the passed device
    :param net:     the network to transfer into the cpu
    :param device:  the device to which the network is sent
    :return:
    """
    net_path = "{}/temp_net.pt".format(temp_dir)

    # ensure that the temp dir exists and is empty and
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)


    # put the model on the gpu
    if device.type == "cuda":
        torch.save({'state_dict': net.state_dict()}, net_path)
        cuda_net = net
        cuda_net.cuda()
        checkpoint = torch.load(net_path, map_location='cuda')
        cuda_net.load_state_dict(checkpoint['state_dict'])
        shutil.rmtree(temp_dir)
        return cuda_net

    # put the model on the cpu
    if device.type == "cpu":
        torch.save({'state_dict': net.state_dict()}, net_path)
        cpu_net = net
        checkpoint = torch.load(net_path, map_location='cpu')
        cpu_net.load_state_dict(checkpoint['state_dict'])
        shutil.rmtree(temp_dir)
        return cpu_net

    logger.error("device type {} is not known".format(device.type))
    return None
