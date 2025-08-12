import torch
import torch.nn as nn
import torch.nn.functional as F


class PCTransitionModel(nn.Module):
    def __init__(self, point_size, latent_size):
        super(PCTransitionModel, self).__init__()
        
        self.latent_size = int(latent_size / 2)
        self.point_size = point_size
        
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.latent_size, 1)

        self.conv4 = torch.nn.Conv1d(3, 64, 1)
        self.conv5 = torch.nn.Conv1d(64, 128, 1)
        self.conv6 = torch.nn.Conv1d(128, self.latent_size, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.latent_size)

        # self.bn4 = nn.BatchNorm1d(64)
        # self.bn5 = nn.BatchNorm1d(128)
        # self.bn6 = nn.BatchNorm1d(self.latent_size)

        self.act_fc1 = nn.Linear(8, 64)
        self.act_fc2 = nn.Linear(64, 128)
        self.act_fc3 = nn.Linear(128, self.latent_size)
        
        self.dec1 = nn.Linear(int(self.latent_size*2),256)
        self.dec2 = nn.Linear(256,256)
        self.dec3 = nn.Linear(256,self.point_size*3)

    def state_encoder(self, x): 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_size)
        return x

    def action_encoder(self, a):
        a = F.relu(self.act_fc1(a))
        a = F.relu(self.act_fc2(a))
        a = self.act_fc3(a)
        return a
        
    # def action_encoder(self, x):
    #     x = F.relu(self.bn4(self.conv4(x)))
    #     x = F.relu(self.bn5(self.conv5(x)))
    #     x = self.bn6(self.conv6(x))
    #     x = torch.max(x, 2, keepdim=True)[0]
    #     x = x.view(-1, self.latent_size)
    #     return x
        
    def decoder(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x.view(-1, self.point_size, 3)
    
    def forward(self, x_t, a_t):
        x_l = self.state_encoder(x_t)
        a_l = self.action_encoder(a_t)
        l = torch.cat([x_l, a_l], dim=1)
        delta = self.decoder(l)
        return delta
    