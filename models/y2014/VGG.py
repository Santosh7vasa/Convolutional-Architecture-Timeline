import torch 
from torch import nn

class VGG(nn.Module):
    
    def __init__(self):
        super(VGG,self).__init__()
        #(feature_map - kernel + 2*padding)/stride + 1
        self.conv_1 = nn.Conv2d(in_channels = 3, out_channels=32,kernel_size=3,padding=1)
        self.conv_2 = nn.Conv2d(in_channels = 32, out_channels=64,kernel_size=3,stride=1,padding=1)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv_3 = nn.Conv2d(in_channels = 64, out_channels=64,kernel_size=3,padding=1)
        self.conv_4 = nn.Conv2d(in_channels = 64, out_channels=128,kernel_size=3, stride=1,padding=1)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(in_features=128*8*8, out_features = 512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)
        
        
        
    def forward(self,x):
        x = nn.functional.relu(self.conv_1(x))
        x = nn.functional.relu(self.conv_2(x))
        x = nn.functional.max_pool2d(x,2)
        x = self.dropout1(x)

        x = nn.functional.relu(self.conv_3(x))
        x = nn.functional.relu(self.conv_4(x))
        x = nn.functional.max_pool2d(x,2)
        x = self.dropout2(x)
        x = x.view(-1, 128*8*8)
        x = self.fc1(x)
        x = self.fc2(x)
        return x