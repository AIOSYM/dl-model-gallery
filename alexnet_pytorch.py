import torch
import torch.nn as nn
import torch.nn.functional as F 

input_channel = 3
num_classes = 2

class AlexNet(nn.Module):
    
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 96, kernel_size=11, stride=4, padding=0)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=1)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(6*6*256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.lrn = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.lrn(x)
        x = F.max_pool2d(x)
        x = F.relu(self.conv2(x))
        x = self.lrn(x)
        x = F.max_pool2d(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5
        x = F.max_pool2d(x)
        x = x.view(-1, self.num_flat_features(x)) #6x6x256
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
if __name__ == '__main__':
    net = AlexNet()
    print("Implementation of {} in PyTorch".format(net.__class__.__name__))
    print(net)