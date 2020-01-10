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
        
        self.inception4a = InceptionModule(480, 192, 96, 16, 64, 208, 48)
        self.inception4b = InceptionModule(512, 160, 112, 24, 64, 224, 64)
        self.inception4c = InceptionModule(512, 128, 128, 24, 64, 256, 64)
        self.inception4d = InceptionModule(512, 112, 144, 32, 64, 288, 64)
        self.inception4e = InceptionModule(528, 256, 160, 32, 128, 320, 128)
        
        self.inception5a = InceptionModule(832, 256, 160, 32, 128, 320, 128)
        self.inception5b = InceptionModule(832, 384, 192, 48, 128, 384, 128)
        
        self.linear = nn.Linear(1024, num_classes)
        
        self.aux1_conv1 = nn.Conv2d(512, 128, kernel_size=1, padding=0, stride=1)
        self.aux1_linear1 = nn.Linear(14*14*128, 1024)
        self.aux1_linear2 = nn.Linear(1024, num_classes)
        
        self.aux2_conv1 = nn.Conv2d(528, 128, kernel_size=1, padding=0, stride=1)
        self.aux2_linear1 = nn.Linear(14*14*128, 1024)
        self.aux2_linear2 = nn.Linear(1024, num_classes)
         
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 3, padding=1, stride=2)
        x = F.local_response_norm(x, size=1)
        
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.local_response_norm(x, size=1)
        x = F.max_pool2d(x, 3, padding=1, stride=2)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = F.max_pool2d(x, 3, padding=1, stride=2)
        
        x = self.inception4a(x)
        
        x_aux1 = F.avg_pool2d(x, 5, padding=2, stride=1)
        x_aux1 = F.relu(self.aux1_conv1(x_aux1))
        x_aux1 = x_aux1.view(-1, 14*14*128)
        x_aux1 = F.relu(self.aux1_linear1(x_aux1))
        x_aux1 = F.dropout(x_aux1, p=0.7)
        x_aux1 = self.aux1_linear2(x_aux1)
        x_aux1 = F.softmax(x_aux1, dim=1)
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        
        x_aux2 = F.avg_pool2d(x, 5, padding=2, stride=1)
        x_aux2 = F.relu(self.aux2_conv1(x_aux2))
        x_aux2 = x_aux2.view(-1, 14*14*28)
        x_aux2 = F.relu(self.aux2_linear1(x_aux2))
        x_aux2 = F.dropout(x, p=0.7)
        x_aux2 = self.aux2_linear2(x_aux2)
        x_aux2 = F.softmax(x_aux2, dim=1)
        
        x = self.inception4e(x)
        x = F.max_pool2d(x, 3, padding=1, stride=2)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = F.avg_pool2d(x, 7, padding=0, stride=1)
        x = x.view(-1, 1024)
        x = self.linear(x)
        x = F.softmax(x, dim=1)  
        
        return x, x_aux1, x_aux2
    
if __name__ == '__main__':
    net = GoogLeNetv1()
    print("Implementation of {} in PyTorch".format(net.__class__.__name__))
    print(net)      
        
        
        
        
        
        
        
    

        
        
        