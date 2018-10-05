import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


# 2-layer GRU model
class FeatAggregate(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, out_size=128, batchn=64):
        super(FeatAggregate, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.layerCnt = 2
        self.lstm1 = nn.GRU(self.input_size,self.out_size,self.layerCnt, batch_first=True)

    def forward(self, feats):

        output_seq,out_test = self.lstm1(feats)
        return output_seq

# Visual-audio multimodal metric learning: LSTM*2+FC*2
class VAMetric(nn.Module):
    def __init__(self):
        super(VAMetric, self).__init__()
        self.VFeatPool = FeatAggregate(1024, 512, 128)
        self.AFeatPool = FeatAggregate(128, 128, 128)
        self.fc = nn.Linear(128, 64)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):
        vfeat = self.VFeatPool(vfeat)
        afeat = self.AFeatPool(afeat)
        vfeat = self.fc(vfeat)
        afeat = self.fc(afeat)

        distance = F.pairwise_distance(vfeat, afeat)

        return distance, torch.mean(distance[0:vfeat.size(0) / 2 - 1]), torch.mean(
            distance[vfeat.size(0) / 2:vfeat.size(0) - 1])


# Visual-audio multimodal metric learning: MaxPool+FC
class VAMetric2(nn.Module):
    def __init__(self, framenum=120):
        super(VAMetric2, self).__init__()
        self.mp = nn.MaxPool1d(framenum)
        self.vfc = nn.Linear(1024, 128)
        self.fc = nn.Linear(128, 96)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):
        # aggregate the visual features
        vfeat = self.mp(vfeat)
        vfeat = vfeat.view(-1, 1024)
        vfeat = F.relu(self.vfc(vfeat))
        vfeat = self.fc(vfeat)

        # aggregate the auditory features
        afeat = self.mp(afeat)
        afeat = afeat.view(-1, 128)
        afeat = self.fc(afeat)

        return F.pairwise_distance(vfeat, afeat)


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, label):
        loss = torch.mean((1 - label) * torch.pow(dist, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        return loss


class GRU_net(nn.Module):
    def __init__(self, framenum=120):
        super(GRU_net, self).__init__()
        #convolution before gru-cell
        self.vconv2 = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=(1, 1024), stride=1)
        self.aconv2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=1)

        self.vconv3 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=1)
        self.aconv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=1)
        # two gru layers
        self.VFeatPool = FeatAggregate(128, 128, 128)
        self.AFeatPool = FeatAggregate(128, 128, 128)

        # two layer convolution after gru
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2, 128), stride=128)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(32,8), stride=1)

        # fcn network
        self.fc3 = nn.Linear(in_features=7232, out_features=2048)
        self.fc4 = nn.Linear(in_features=2048, out_features=2)
        self.init_params()

    def init_params(self):
        for m in self.modules():

            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat, train_status=False):
        vfeat = vfeat.view(vfeat.size(0), -1, vfeat.size(1), vfeat.size(2))
        afeat = afeat.view(afeat.size(0), -1, afeat.size(1), afeat.size(2))
        vfeat = F.sigmoid(self.vconv2(vfeat))
        afeat = F.sigmoid(self.aconv2(afeat))
        vfeat = F.sigmoid(self.vconv3(vfeat))
        afeat = F.sigmoid(self.aconv3(afeat))

        vfeat = vfeat.transpose(1, 2)
        afeat = afeat.transpose(1, 2)
        vfeat = vfeat.contiguous()
        afeat = afeat.contiguous()
        vfeat = vfeat.view(vfeat.size(0), vfeat.size(1), -1)
        afeat = afeat.view(afeat.size(0), vfeat.size(1), -1)
        vfeat = self.VFeatPool(vfeat)
        afeat = self.AFeatPool(afeat)

        vfeat = vfeat.contiguous()
        afeat = afeat.contiguous()
        vfeat = vfeat.view(vfeat.size(0), 1, 1, -1)
        afeat = afeat.view(afeat.size(0), 1, 1, -1)

        vafeat = torch.cat((vfeat, afeat), dim=2)
        vafeat = self.conv1(vafeat)
        vafeat = vafeat.view(vafeat.size(0),1, vafeat.size(1), -1)
        vafeat = self.conv2(vafeat)
        # vafeat = self.mp(vafeat)

        vafeat = vafeat.view([vafeat.size(0), -1])
        vafeat = self.fc3(vafeat)
        vafeat = F.relu(vafeat)
        vafeat = self.fc4(vafeat)

        if train_status:
            prob = vafeat
        else:
            prob = F.softmax(vafeat)
        return prob

# only to test the git hub

class StableBCELoss(nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, inputdata, target):
        #neg_abs = - input.abs()
        dist = inputdata-target
        loss = torch.pow(dist,2)
        #loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()
