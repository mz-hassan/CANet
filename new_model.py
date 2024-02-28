import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
import torchvision.models as models

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # Convolutional layers with BatchNorm
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(2048)
        
        # Max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.pool(torch.relu(self.bn5(self.conv5(x))))
        
        return x

class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 2048),
        )
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.dropout(x)
        avg_out = self.mlp(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.mlp(self.max_pool(x).view(x.size(0), -1))
        temp = avg_out + max_out

        channel_attention = self.sigmoid(temp)
        return x * channel_attention.view(x.size(0), x.size(1), 1, 1)
    
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.bn(self.conv1(concat))
        spatial_attention = self.sigmoid(out)
        return x * spatial_attention

class CBAM(nn.Module):
    def __init__(self):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention()
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
    
class CrossSAttention(nn.Module):
    def __init__(self):
        super(CrossSAttention, self).__init__()

        self.mlp = Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 1024)
        )

    def forward(self, x, y):
        a = y * F.sigmoid(self.mlp(y))
        return x + a
    
class CANet_reg(nn.Module):
    def __init__(self, tune, model, cross_attention):
        super(CANet_reg, self).__init__()

        self.cross_attention = cross_attention

        if model=="resnet":
            self.model = models.resnet50(pretrained=True)
            self.model =  nn.Sequential(*list(self.model.children())[:-2])
            
            if not tune:
                for param in self.model.parameters():
                        param.requires_grad = False
        
        if model=="cnn":
            self.model = CustomCNN()

        self.cbam_dr = CBAM()
        self.cbam_dme = CBAM()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc_dr = nn.Linear(2048, 1024)
        self.fc_dme = nn.Linear(2048, 1024)

        self.fc_dr_s = nn.Linear(1024, 2)
        self.fc_dme_s = nn.Linear(1024, 3)

        if cross_attention:
            self.cross_dr = CrossSAttention()
            self.cross_dme = CrossSAttention()

            self.fc_dr_r = nn.Linear(1024, 2)
            self.fc_dme_r = nn.Linear(1024, 3)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.model(x)
        dr = self.cbam_dr(x)
        dme = self.cbam_dme(x)

        dr = self.avg_pool(x).view(dr.size(0), -1)
        dme = self.avg_pool(x).view(dme.size(0), -1)

        dr = self.dropout(self.fc_dr(dr))
        dme = self.dropout(self.fc_dme(dme))

        dr_s = self.fc_dr_s(dr) # self prediction
        dme_s = self.fc_dme_s(dme)

        if self.cross_attention:
            dr_r = self.dropout(self.cross_dr(dr, dme))
            dme_r = self.dropout(self.cross_dme(dme, dr))

            dr_r = self.fc_dr_r(dr_r) # relation prediction
            dme_r = self.fc_dme_r(dme_r)
        
            return dr_s, dr_r, dme_r, dme_s
        _ = 0
        return dr_s, _, _, dme_s