import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class _EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, polling=True, bn=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        self.encode = nn.Sequential(*layers)
        self.pool = None
        if polling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, bn=False):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, bn=False):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(in_channels, 64, polling=False, bn=bn)
        self.enc2 = _EncoderBlock(64, 128, bn=bn)
        self.enc3 = _EncoderBlock(128, 256, bn=bn)
        self.enc4 = _EncoderBlock(256, 512, bn=bn)
        self.polling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlock(512, 1024, 512, bn=bn)
        self.dec4 = _DecoderBlock(1024, 512, 256, bn=bn)
        self.dec3 = _DecoderBlock(512, 256, 128, bn=bn)
        self.dec2 = _DecoderBlock(256, 128, 64, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc4))
        dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], mode='bilinear',
                                                  align_corners=True), enc4], 1))
        dec3 = self.dec3(torch.cat([dec4, enc3], 1))
        dec2 = self.dec2(torch.cat([dec3, enc2], 1))
        dec1 = self.dec1(torch.cat([dec2, enc1], 1))
        final = self.final(dec1)
        return final


class _EncoderBlockV2(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, polling=True, bn=False):
        super(_EncoderBlockV2, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        self.encode = nn.Sequential(*layers)
        self.pool = None
        if polling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        return self.encode(x)


class _DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, bn=False):
        super(_DecoderBlockV2, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.decode(x)


class UNetV2(nn.Module):
    def __init__(self, num_classes, in_channels=3, bn=False):
        super(UNetV2, self).__init__()
        # self.US=nn.Upsample(size=[self.ny,self.nx],mode='bicubic')
        self.enc1 = _EncoderBlockV2(in_channels, 64, polling=False, bn=bn)
        self.enc2 = _EncoderBlockV2(64, 128, bn=bn)
        self.enc3 = _EncoderBlockV2(128, 256, bn=bn)
        self.enc4 = _EncoderBlockV2(256, 512, bn=bn)
        self.polling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlockV2(512, 1024, 512, bn=bn)
        self.dec4 = _DecoderBlockV2(1024, 512, 256, bn=bn)
        self.dec3 = _DecoderBlockV2(512, 256, 128, bn=bn)
        self.dec2 = _DecoderBlockV2(256, 128, 64, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )
        self.conv_8 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.conv_4 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.conv_2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        # x = self.US(x)
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc4))
        dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc4], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc1], 1))
        final = self.final(dec1)
        return final

class UNetsingle(nn.Module):
    def __init__(self,nx,ny,num_classes, in_channels=3, bn=False):
        super(UNetsingle, self).__init__()
        self.nx = nx
        self.ny = ny
        self.US=nn.Upsample(size=[self.ny,self.nx],mode='bicubic')
        self.enc1 = _EncoderBlockV2(in_channels, 64, polling=False, bn=bn)
        self.enc2 = _EncoderBlockV2(64, 128, bn=bn)
        self.enc3 = _EncoderBlockV2(128, 256, bn=bn)
        self.enc4 = _EncoderBlockV2(256, 512, bn=bn)
        self.polling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlockV2(512, 1024, 512, bn=bn)
        self.dec4 = _DecoderBlockV2(1024, 512, 256, bn=bn)
        self.dec3 = _DecoderBlockV2(512, 256, 128, bn=bn)
        self.dec2 = _DecoderBlockV2(256, 128, 64, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )
        self.conv_8 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.conv_4 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.conv_2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        x = self.US(x)
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc4))
        dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc4], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc1], 1))
        final = self.final(dec1)
        return final

class UNetupsample(nn.Module):
    def __init__(self,nx,ny,num_classes, in_channels=3, bn=False):
        super(UNetupsample, self).__init__()
        self.nx = nx
        self.ny = ny
        self.US=nn.Upsample(size=[self.ny,self.nx],mode='bicubic')
        self.enc1 = _EncoderBlockV2(in_channels, 64, polling=False, bn=bn)
        self.enc2 = _EncoderBlockV2(64, 128, bn=bn)
        self.enc3 = _EncoderBlockV2(128, 256, bn=bn)
        self.enc4 = _EncoderBlockV2(256, 512, bn=bn)
        self.polling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlockV2(512, 1024, 512, bn=bn)
        self.dec4 = _DecoderBlockV2(1024, 512, 256, bn=bn)
        self.dec3 = _DecoderBlockV2(512, 256, 128, bn=bn)
        self.dec2 = _DecoderBlockV2(256, 128, 64, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )
        self.conv_8 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.conv_4 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.conv_2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, lr, hr):
        lr = self.US(lr)
        x = torch.cat([lr,hr],dim=1)
        # print(x.shape)
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc4))
        dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc4], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc1], 1))
        final = self.final(dec1)
        return final

class UNetupnew(nn.Module):
    def __init__(self,nx,ny,num_classes, in_channels=3, bn=False):
        super(UNetupnew, self).__init__()
        self.nx = nx
        self.ny = ny
        # self.US=nn.Upsample(size=[self.ny,self.nx],mode='bicubic')
        self.enc1 = _EncoderBlockV2(in_channels, 64, polling=False, bn=bn)
        self.enc2 = _EncoderBlockV2(64, 128, bn=bn)
        self.enc3 = _EncoderBlockV2(128, 256, bn=bn)
        self.enc4 = _EncoderBlockV2(256, 512, bn=bn)
        self.polling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlockV2(512, 1024, 512, bn=bn)
        self.dec4 = _DecoderBlockV2(1024, 512, 256, bn=bn)
        self.dec3 = _DecoderBlockV2(1024, 256, 128, bn=bn)
        self.dec2 = _DecoderBlockV2(256, 128, 64, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )
        self.conv_8 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.conv_4 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.conv_2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.conv = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=1, stride=1, padding=0)
        initialize_weights(self)

    def forward(self, x, y):
        # x = self.US(x)
        enc1 = self.enc1(x)
        # print('e1:',enc1.shape)
        enc2 = self.enc2(enc1)
        # print('e2:',enc2.shape)
        enc3 = self.enc3(enc2)
        # print('e3:',enc3.shape)
        enc4 = self.enc4(enc3)
        # print('e4:',enc4.shape)
        center = self.center(self.polling(enc4))
        # print('center',center.shape)
        
        dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc4], 1))
        y = self.conv(y)
        dec4 = torch.cat([dec4,y],dim=1)
        # print('d4:',dec4.shape)
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc3], 1))
        # print('d3:',dec3.shape)
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc2], 1))
        # print('d2:',dec2.shape)
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc1], 1))
        # print('d1:',dec1.shape)
        final = self.final(dec1)
        # print('final:',final.shape)
        return final

class UNetlr(nn.Module):
    def __init__(self,nx,ny,num_classes, in_channels=3, bn=False):
        super(UNetlr, self).__init__()
        self.nx = nx
        self.ny = ny
        self.US=nn.Upsample(size=[self.ny,self.nx],mode='bicubic')
        self.enc1 = _EncoderBlockV2(in_channels, 64, polling=False, bn=bn)
        self.enc2 = _EncoderBlockV2(64, 128, bn=bn)
        self.enc3 = _EncoderBlockV2(128, 256, bn=bn)
        self.enc4 = _EncoderBlockV2(256, 512, bn=bn)
        self.polling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlockV2(512, 1024, 512, bn=bn)
        self.dec4 = _DecoderBlockV2(1024, 512, 256, bn=bn)
        self.dec3 = _DecoderBlockV2(512, 256, 128, bn=bn)
        self.dec2 = _DecoderBlockV2(256, 128, 64, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )
        self.conv_8 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.conv_4 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.conv_2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, lr, lr_before):
        lr = self.US(lr)
        lr_before = self.US(lr_before)
        x = torch.cat([lr,lr_before],dim=1)
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc4))
        dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc4], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc1], 1))
        final = self.final(dec1)
        return final

class UNetlr3(nn.Module):
    def __init__(self,nx,ny,num_classes, in_channels=3, bn=False):
        super(UNetlr3, self).__init__()
        self.nx = nx
        self.ny = ny
        self.US=nn.Upsample(size=[self.ny,self.nx],mode='bicubic')
        self.enc1 = _EncoderBlockV2(in_channels, 64, polling=False, bn=bn)
        self.enc2 = _EncoderBlockV2(64, 128, bn=bn)
        self.enc3 = _EncoderBlockV2(128, 256, bn=bn)
        self.enc4 = _EncoderBlockV2(256, 512, bn=bn)
        self.polling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlockV2(512, 1024, 512, bn=bn)
        self.dec4 = _DecoderBlockV2(1024, 512, 256, bn=bn)
        self.dec3 = _DecoderBlockV2(512, 256, 128, bn=bn)
        self.dec2 = _DecoderBlockV2(256, 128, 64, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )
        self.conv_8 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.conv_4 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.conv_2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, lr, lr_before, lr_before1):
        lr = self.US(lr)
        lr_before = self.US(lr_before)
        lr_before1 = self.US(lr_before1)
        x = torch.cat([lr,lr_before,lr_before1],dim=1)
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc4))
        dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc4], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc1], 1))
        final = self.final(dec1)
        return final

class UNethr(nn.Module):
    def __init__(self, num_classes, in_channels=3, bn=False):
        super(UNethr, self).__init__()
        self.enc1 = _EncoderBlockV2(in_channels, 64, polling=False, bn=bn)
        self.enc2 = _EncoderBlockV2(64, 128, bn=bn)
        self.enc3 = _EncoderBlockV2(128, 256, bn=bn)
        self.enc4 = _EncoderBlockV2(256, 512, bn=bn)
        self.polling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlockV2(512, 1024, 512, bn=bn)
        self.dec4 = _DecoderBlockV2(1024, 512, 256, bn=bn)
        self.dec3 = _DecoderBlockV2(512, 256, 128, bn=bn)
        self.dec2 = _DecoderBlockV2(256, 128, 64, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )
        self.conv_8 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.conv_4 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.conv_2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, hr_before, hr_now):
        x = torch.cat([hr_before, hr_now],dim=1)
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc4))
        dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc4], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc1], 1))
        final = self.final(dec1)
        return final

class UNetMPTN(nn.Module):
    def __init__(self,nx,ny,num_classes, in_channels=3, bn=False):
        super(UNetMPTN, self).__init__()
        self.nx = nx
        self.ny = ny
        self.US=nn.Upsample(size=[self.ny,self.nx],mode='bicubic')
        self.enc1 = _EncoderBlockV2(in_channels, 64, polling=False, bn=bn)
        self.enc2 = _EncoderBlockV2(64, 128, bn=bn)
        self.enc3 = _EncoderBlockV2(128, 256, bn=bn)
        self.enc4 = _EncoderBlockV2(256, 512, bn=bn)
        self.polling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlockV2(512, 1024, 512, bn=bn)
        self.dec4 = _DecoderBlockV2(1024, 512, 256, bn=bn)
        self.dec3 = _DecoderBlockV2(512, 256, 128, bn=bn)
        self.dec2 = _DecoderBlockV2(256, 128, 64, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )
        self.conv_8 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.conv_4 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.conv_2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, lr, hr_before, hr):
        lr = self.US(lr)
        x = torch.cat([lr,hr_before,hr],dim=1)
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc4))
        dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc4], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc1], 1))
        final = self.final(dec1)
        return final

class UNetMPTN1(nn.Module):
    def __init__(self,nx,ny,num_classes, in_channels=3, bn=False):
        super(UNetMPTN1, self).__init__()
        self.nx = nx
        self.ny = ny
        self.US=nn.Upsample(size=[self.ny,self.nx],mode='bicubic')
        self.enc1 = _EncoderBlockV2(in_channels, 64, polling=False, bn=bn)
        self.enc2 = _EncoderBlockV2(64, 128, bn=bn)
        self.enc3 = _EncoderBlockV2(128, 256, bn=bn)
        self.enc4 = _EncoderBlockV2(256, 512, bn=bn)
        self.polling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlockV2(512, 1024, 512, bn=bn)
        self.dec4 = _DecoderBlockV2(1024, 512, 256, bn=bn)
        self.dec3 = _DecoderBlockV2(512, 256, 128, bn=bn)
        self.dec2 = _DecoderBlockV2(256, 128, 64, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )
        self.conv_8 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.conv_4 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.conv_2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, lr_before, lr, hr_before, hr):
        lr = self.US(lr)
        lr_before = self.US(lr_before)
        x = torch.cat([lr_before,lr,hr_before,hr],dim=1)
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc4))
        dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc4], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc1], 1))
        final = self.final(dec1)
        return final

class UNetMPTN2(nn.Module):
    def __init__(self,nx,ny,num_classes, in_channels=3, bn=False):
        super(UNetMPTN2, self).__init__()
        self.nx = nx
        self.ny = ny
        self.US=nn.Upsample(size=[self.ny,self.nx],mode='bicubic')
        self.enc1 = _EncoderBlockV2(in_channels, 64, polling=False, bn=bn)
        self.enc2 = _EncoderBlockV2(64, 128, bn=bn)
        self.enc3 = _EncoderBlockV2(128, 256, bn=bn)
        self.enc4 = _EncoderBlockV2(256, 512, bn=bn)
        self.polling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlockV2(512, 1024, 512, bn=bn)
        self.dec4 = _DecoderBlockV2(1024, 512, 256, bn=bn)
        self.dec3 = _DecoderBlockV2(512, 256, 128, bn=bn)
        self.dec2 = _DecoderBlockV2(256, 128, 64, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if bn else nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )
        self.conv_8 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.conv_4 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.conv_2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, lr, hr_before, hr):
        lr = self.US(lr)
        x = torch.cat([lr,hr_before,hr],dim=1)
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc4))
        dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc4], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc1], 1))
        final = self.final(dec1)
        return final
# net = UNetV2(in_channels=1, num_classes=1).to(device)  