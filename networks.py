import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from globals import CONST, Config
import board_representation



###############################################################################################################
#                                           ResNet                                                            #
###############################################################################################################
class ConvBlock(nn.Module):
    """
    define one convolutional block
    """

    def __init__(self, n_filters):
        super(ConvBlock, self).__init__()
        self.action_size = 7
        self.conv1 = nn.Conv2d(2, n_filters, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class ResBlock(nn.Module):
    """
    defines the residual block of the ResNet
    """

    def __init__(self, n_filters):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)  #bias=False
        self.bn1 = nn.BatchNorm2d(n_filters)

        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        # save the input for the skip connection
        residual = x

        # conv1
        out = F.relu(self.bn1(self.conv1(x)))

        # conv2 with the skip connection
        out = F.relu(self.bn2(self.conv2(out)) + residual)

        return out


class SEBlock(nn.Module):
    """
    defines a squeeze and excitation block (Czech et al. 2019)
    """

    def __init__(self, n_filters):
        super(SEBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)  #bias=False
        self.bn1 = nn.BatchNorm2d(n_filters)

        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        # save the input for the skip connection
        residual = x

        # conv1
        out = F.relu(self.bn1(self.conv1(x)))

        # conv2 with the skip connection
        out = F.relu(self.bn2(self.conv2(out)) + residual)

        return out



class OutBlock(nn.Module):
    """
    define the alpha zero output block with the value and the policy head
    """

    def __init__(self, n_filters):
        super(OutBlock, self).__init__()
        self.value_filters = 32
        self.policy_filters = 32

        self.conv1_v = nn.Conv2d(n_filters, self.value_filters, kernel_size=1)  # value head
        self.bn1_v = nn.BatchNorm2d(self.value_filters)
        self.fc1_v = nn.Linear(self.value_filters * CONST.BOARD_SIZE, 256)
        self.fc2_v = nn.Linear(256, 1)

        self.conv1_p = nn.Conv2d(n_filters, self.policy_filters, kernel_size=1)  # policy head
        self.bn1_p = nn.BatchNorm2d(self.policy_filters)
        self.fc1_p = nn.Linear(self.policy_filters * CONST.BOARD_SIZE, board_representation.LABEL_COUNT)
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        # value head
        v = F.relu(self.bn1_v(self.conv1_v(x)))
        v = v.view(-1, self.value_filters * CONST.BOARD_SIZE)  # channels*board size
        v = F.relu(self.fc1_v(v))
        v = F.relu(self.fc2_v(v))
        v = torch.tanh(v)

        # policy head
        p = F.relu(self.bn1_p(self.conv1_p(x)))
        p = p.view(-1, self.policy_filters*CONST.BOARD_SIZE)
        p = F.relu(self.fc1_p(p))
        p = self.logsoftmax(p).exp()
        return p, v


class ResNet(nn.Module):
    """
    defines a resudual neural network that ends in fully connected layers
    the network has a policy and a value head
    """

    def __init__(self, learning_rate, n_blocks, n_filters, weight_decay=0):
        super(ResNet, self).__init__()

        self.n_channels = n_filters
        self.n_blocks = n_blocks

        # initial convolutional block
        self.conv = ConvBlock(n_filters)

        # residual blocks
        for i in range(n_blocks):
            setattr(self, "res{}".format(i), ResBlock(n_filters))

        # output block with the policy and the value head
        self.outblock = OutBlock(n_filters)

        # define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)


    def forward(self, x):
        # initial convolutional block
        out = self.conv(x)

        # residual blocks
        for i in range(self.n_blocks):
            out = getattr(self, "res{}".format(i))(out)

        # output block with the policy and value head
        out = self.outblock(out)
        return out


    def train_step(self, batch, target_p, target_v):
        """
        executes one training step of the neural network
        :param batch:           tensor with data [batchSize, nn_input_size]
        :param target_p:        policy target
        :param target_v:        value target
        :return:                policy loss, value loss
        """

        # send the tensors to the used device
        data = batch.to(Config.training_device)

        self.optimizer.zero_grad()  # reset the gradients to zero in every epoch
        prediction_p, prediction_v = self(data)  # pass the data through the network to get the prediction

        # create the label
        target_p = target_p.to(Config.training_device)
        target_v = target_v.to(Config.training_device)
        # criterion_p = nn.KLDivLoss()
        # criterion_p(torch.log(1e-8 + prediction_p), target_p)
        criterion_v = nn.MSELoss()

        # define the loss
        # loss_p = - torch.sum(target_p * torch.log(1e-8 + prediction_p), 1).mean() / target_p.shape[1]
        loss_p = - torch.sum(target_p * torch.log(1e-8 + prediction_p), 1).mean()
        loss_v = criterion_v(prediction_v, target_v)
        loss = loss_p + loss_v
        loss.backward()  # back propagation
        self.optimizer.step()  # make one optimization step
        return loss_p, loss_v
