import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        self.mlp1 = nn.Sequential( 
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        num_points = points.size(1)

        x = points.transpose(2, 1)  # (B, 3, N)
        x = self.mlp1(x)            # (B, 64, N)
        x = self.mlp2(x)            # (B, 1024, N)

        x = nn.MaxPool1d(num_points)(x)  # (B, 1024, 1)
        x = x.view(-1, 1024)             # (B, 1024)

        x = self.fc(x)                   # (B, num_classes) 
        return x

# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        self.mlp1 = nn.Sequential( 
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.seg_mlp = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512), 
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),   
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),   
            nn.ReLU(),
            nn.Conv1d(128, num_seg_classes, 1)
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        
        num_points = points.size(1)
        x = points.transpose(2, 1)  # (B, 3, N)
        point_feat = self.mlp1(x)   # (B, 64, N)

        x = self.mlp2(point_feat)   # (B, 1024, N)
        x = nn.MaxPool1d(num_points)(x)  # (B, 1024, 1)
        global_feat = x.repeat(1, 1, num_points)  # (B, 1024, N)

        x = torch.cat([point_feat, global_feat], dim=1)  # (B, 1088, N)
        x = self.seg_mlp(x)  # (B, num_seg_classes, N)
        x = x.transpose(2, 1)  # (B, N, num_seg_classes)
        return x



