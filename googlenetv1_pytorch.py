import torch 
import torch.nn as nn 
import torch.nn.functional as F 

img_width, img_height = 224, 224
input_channel = 3
num_classes = 2

class InceptionModule(nn.Module):
    def __init__(self, input_filter1):
        super(InceptionModule, self).__init__()
        
    def forward(self, x):
        return x
        
        