import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.hidden = nn.Linear(4,256)
        self.output = nn.Linear(256,2)
        self.ReLU = nn.ReLU()

    def forward(self,x):
        x = self.hidden(x)
        x = self.ReLU(x)
        x = self.output(x)
        return x

    def save_model(self, path):
        torch.save(self.state_dict(), path)