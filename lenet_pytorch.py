import torch
import torch.nn as nn
import torch.nn.functional as F 

input_channel = 3
img_height, img_width = 32, 32
num_classes = 10

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 6, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.fc1   = nn.Linear(5*5*16, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.sigmoid(F.max_pool2d(x, 2))
        x = self.conv2(x)
        x = F.sigmoid(F.max_pool2d(x, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == '__main__':
    net = LeNet()
    print("Implementation of {} in PyTorch".format(net.__class__.__name__))
    print(net)