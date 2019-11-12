import torch
from torch import nn, optim
from torch.nn import functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        
        in_stride = 2 if downsample else 1
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, stride=in_stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        )
        
        self.resize_input = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 1, stride=in_stride, padding=0)
        )
        
    def forward(self, x):
        x_ = self.conv(x)
        residual = self.resize_input(x)
        return x_ + residual
    
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, downsample=False)
        
    def forward(self, x, x_stored):
        x = self.up(x)
        x = torch.cat([x, x_stored], dim=1)
        x = self.conv(x)
        return x

class ResUNet(nn.Module):

    def __init__(self, in_ch, out_ch, f=64):
        super().__init__()
        self.inconv = DoubleConv(in_ch, f, downsample=False)
        self.down1 = DoubleConv(f, 2*f)   
        self.down2 = DoubleConv(2*f, 4*f)  
        self.down3 = DoubleConv(4*f, 8*f)  
        self.up1 = UpBlock(8*f, 4*f)
        self.up2 = UpBlock(4*f, 2*f)
        self.up3 = UpBlock(2*f, f)
        self.outconv = nn.Conv2d(f, out_ch, 1)
    
    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = x4
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outconv(x)
        return x