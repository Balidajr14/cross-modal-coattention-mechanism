import torch
import torch.nn as nn
import torch.nn.functional as F


class Emr_Encoder(nn.Module):
    def __init__(self, feature_size, embedding_size, num_units, dropout):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(feature_size, embedding_size, requires_grad=True))
        self.lin = nn.Linear(20, 128)
        self.hidden1 = nn.Linear(128, 256)
        self.hidden2 = nn.Linear(256, feature_size)
        self.drop = nn.Dropout(dropout)


    def forward(self, x):
        x = self.lin(x)
        x = self.drop(x)
        x = F.relu(x)
        x = self.hidden1(x)
        x = self.drop(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = torch.unsqueeze(x, 2) # add in dimension
        x = self.embeddings * x
        return x

class Image_Encoder(nn.Module):
    def __init__(self, dropout_p, output_size):
        super().__init__()

        self.conv1 = nn.Conv3d(1, 16, 3, 1)
        self.maxpool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(16, 32, 3, 1)
        self.maxpool2 = nn.MaxPool3d(2)
        self.do1 = nn.Dropout(dropout_p)
        self.conv3 = nn.Conv3d(32, output_size, 3, 1)
        #self.maxpool3 = nn.MaxPool3d(2)
        #self.conv4 = nn.Conv3d(64, 64, 1, 1)

        self.out_channels = 64

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.do1(x)
        x = self.conv3(x)
        '''
        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)
        x = self.conv4(x)
        '''
        return x