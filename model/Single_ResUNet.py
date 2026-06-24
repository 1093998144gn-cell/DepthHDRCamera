import torch
import torch.nn as nn
from collections import namedtuple

# 输出容器
OutputsContainer = namedtuple('OutputContainer', field_names=['est_rgb'])

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm_layer(out_ch),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.conv_block(x) + self.shortcut(x))

class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ResidualConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        out = self.pool(skip)
        return out, skip

class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ResidualConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class SingleEncoderSingleDecoderResUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=64, n_layers=4):
        super().__init__()
        self.n_layers = n_layers
        
        # 1. 初始输入
        self.encoder_input = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)
        
        # 2. 编码器：记录每一层的通道数
        self.encoder_downblocks = nn.ModuleList()
        chs = [base_ch]
        for i in range(n_layers):
            self.encoder_downblocks.append(DownsampleBlock(chs[-1], chs[-1] * 2))
            chs.append(chs[-1] * 2)
        
        # 此时 chs 为 [64, 128, 256, 512, 1024]
        
        # 3. 瓶颈层
        self.bottleneck = ResidualConvBlock(chs[-1], chs[-1])
        
        # 4. 解码器：从 chs 倒序取值计算拼接维度
        self.upblocks = nn.ModuleList()
        # 循环：i=0(1024+1024->512), i=1(512+512->256), i=2(256+256->128), i=3(128+128->64)
        for i in range(n_layers):
            in_c = chs[n_layers - i] + chs[n_layers - i] # 当前输入 + 同层skip
            out_c = chs[n_layers - i - 1]
            self.upblocks.append(UpsampleBlock(in_c, out_c))
            
        # 5. 输出
        self.output = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.encoder_input(x)
        
        skips = []
        for block in self.encoder_downblocks:
            x, skip = block(x)
            skips.append(skip)
            
        x = self.bottleneck(x)
        
        for i, block in enumerate(self.upblocks):
            skip = skips[self.n_layers - 1 - i]
            x = block(x, skip)
            
        return OutputsContainer(est_rgb=torch.sigmoid(self.output(x)))