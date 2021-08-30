import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        
        self.liner = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        # self.liner = nn.Sequential(nn.Linear(32, 16), nn.Sigmoid())
        self.out = nn.Sequential(nn.Linear(16, 1),nn.Sigmoid())

    def forward_one(self, x):
        x = x.float()
        # x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return out


# for test
if __name__ == '__main__':
    net = Siamese()
    print(net)
    # print(list(net.parameters()))
