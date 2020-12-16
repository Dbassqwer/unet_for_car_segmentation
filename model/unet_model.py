import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # self.inc = DoubleConv(n_channels, 24)
        # self.down1 = Down(24, 32)
        # self.down2 = Down(32, 64)
        # self.down3 = Down(64, 96)
        # self.down4 = Down(96, 96)
        # self.up1 = Up(192, 64, bilinear)
        # self.up2 = Up(128, 32, bilinear)
        # self.up3 = Up(64, 24, bilinear)
        # self.up4 = Up(48, 24, bilinear)
        # self.outc = OutConv(24, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        x5 = self.down4(x4)
        # print(x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNet_light(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_light, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 24)
        self.down1 = Down(24, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 96)
        self.down4 = Down(96, 96)
        self.up1 = Up(192, 64, bilinear)
        self.up2 = Up(128, 32, bilinear)
        self.up3 = Up(64, 24, bilinear)
        self.up4 = Up(48, 24, bilinear)
        self.outc = OutConv(24, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        x5 = self.down4(x4)
        # print(x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    from torchsummary import summary
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet_light(n_channels=3, n_classes=1).to(device)
    # print(model)
    summary(net,(3,512,512))