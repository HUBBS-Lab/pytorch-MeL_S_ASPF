import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class FNN(nn.Module):

    def __init__(self):
        super(FNN, self).__init__()
        
        self.liner = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3),

            # nn.Linear(64, 32),
            # nn.ReLU(),
            # nn.Linear(32, 16),
            # nn.ReLU(),
            # nn.Linear(16, 8),
            # nn.ReLU(),
            # nn.Linear(8, 3),
        )
        # self.liner = nn.Sequential(nn.Linear(32, 16), nn.Sigmoid())
        # self.out = nn.Sequential(nn.Linear(16, 1),nn.Sigmoid())

    def forward(self, x):
        x = x.float()
        # x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x



# for test
if __name__ == '__main__':
    net = FNN()
    print(net)
    # print(list(net.parameters()))
