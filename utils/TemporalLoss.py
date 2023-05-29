import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import cv2
import numpy as np

import random


''' Optical flow warping function '''
def warp(x, flo, padding_mode='border'):
    B, C, H, W = x.size()

    # Mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid - flo

    # Scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, padding_mode=padding_mode, mode='nearest')
    return output


''' Regularization from paper: Consistent Video Style Transfer via Compound Regularization'''
class TemporalLoss(nn.Module):
    def __init__(self, data_sigma=True, data_w=True, noise_level=0.001,
                 motion_level=8, shift_level=10):

        super(TemporalLoss, self).__init__()
        self.MSE = torch.nn.MSELoss()

        self.data_sigma = data_sigma
        self.data_w = data_w
        self.noise_level = noise_level
        self.motion_level = motion_level
        self.shift_level = shift_level

    """ Flow should have most values in the range of [-1, 1]. 
        For example, values x = -1, y = -1 is the left-top pixel of input, 
        and values  x = 1, y = 1 is the right-bottom pixel of input.
        Flow should be from pre_frame to cur_frame """

    def GaussianNoise(self, ins, mean=0, stddev=0.001):
        stddev = stddev + random.random() * stddev
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))

        if ins.is_cuda:
            noise = noise.cuda()
        return ins + noise

    def GenerateFakeFlow(self, height, width):
        ''' height: img.shape[0]
            width:  img.shape[1] '''

        if self.motion_level > 0:
            flow = np.random.normal(0, scale=self.motion_level, size=[height // 100, width // 100, 2])
            flow = cv2.resize(flow, (width, height))
            flow[:, :, 0] += random.randint(-self.shift_level, self.shift_level)
            flow[:, :, 1] += random.randint(-self.shift_level, self.shift_level)
            flow = cv2.blur(flow, (100, 100))
        else:
            flow = np.ones([width, height, 2])
            flow[:, :, 0] = random.randint(-self.shift_level, self.shift_level)
            flow[:, :, 1] = random.randint(-self.shift_level, self.shift_level)

        return torch.from_numpy(flow.transpose((2, 0, 1))).float()

    def GenerateFakeData(self, first_frame):
        ''' Input should be a (H,W,3) numpy of value range [0,1]. '''

        if self.data_w:
            forward_flow = self.GenerateFakeFlow(first_frame.shape[2], first_frame.shape[3])
            if first_frame.is_cuda:
                forward_flow = forward_flow.cuda()
            forward_flow = forward_flow.expand(first_frame.shape[0], 2, first_frame.shape[2], first_frame.shape[3])
            second_frame = warp(first_frame, forward_flow)
        else:
            second_frame = first_frame.clone()
            forward_flow = None

        if self.data_sigma:
            second_frame = self.GaussianNoise(second_frame, stddev=self.noise_level)

        return second_frame, forward_flow

    def forward(self, first_frame, second_frame, forward_flow):
        if self.data_w:
            first_frame = warp(first_frame, forward_flow)

        temporalloss = torch.mean(torch.abs(first_frame - second_frame))

        return temporalloss, first_frame


