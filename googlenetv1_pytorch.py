import torch 
import torch.nn as nn 
import torch.nn.functional as F 

img_width, img_height = 224, 224
input_channel = 3
num_classes = 2

class InceptionModule(nn.Module):
    
    def __init__(self, input_f1, ch1x1_1, ch1x1_2, ch1x1_3, ch1x1_4, ch3x3, ch5x5 ):
        super(InceptionModule, self).__init__()
        
        self.conv1 = nn.Conv2d(input_f1, ch1x1_1, kernel_size=1, padding=0)
        
        self.conv2_1 = nn.Conv2d(input_f1, ch1x1_2, kernel_size=1, padding=0)
        self.conv2_2 = nn.Conv2d(ch1x1_2, ch3x3, kernel_size=3, padding=1)
        
        self.conv3_1 = nn.Conv2d(input_f1, ch1x1_3, kernel_size=1, padding=0)
        self.conv3_2 = nn.Conv2d(ch1x1_3, ch5x5, kernel_size=5, padding=2)
        
        self.conv4_2 = nn.Conv2d(input_f1, ch1x1_4, kernel_size=1, padding=0)
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        
        x2 = F.relu(self.conv2_1(x))
        x2 = F.relu(self.conv2_2(x2))
        
        x3 = F.relu(self.conv3_1(x))
        x3 = F.relu(self.conv3_2(x3))
        
        x4 = F.max_pool2d(x, 3, padding=1, stride=1)
        x4 = F.relu(self.conv4_2(x4))
        
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x

class GoogLeNetv1(nn.Module):
    
    def __init__(self):
        super(GoogLeNetv1, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, padding=0, stride=2)
        
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1)
        self.conv2_2 = nn.Conv2d(64, 192, kernel_size=3, padding=1, stride=1)
        
        self.inception3a = InceptionModule(192, 64, 96, 16, 32, 128, 32)
        self.inception3b = InceptionModule(256, 128, 128, 32, 64, 192, 96)
    

        
        
        