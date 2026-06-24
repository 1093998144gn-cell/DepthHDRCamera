import torch
import torch.nn as nn
from collections import namedtuple

# 输出容器定义
OutputsContainer = namedtuple('OutputContainer', field_names=['est_spectrals', 'est_depthmaps'])


# 残差卷积块（带有残差连接）
class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer=nn.BatchNorm2d, momentum=0.01):
        super().__init__()
        bias = False if norm_layer is not nn.Identity else True

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias),
            norm_layer(out_ch, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=bias),
            norm_layer(out_ch, momentum=momentum),
        )
        # 快捷路径：如果通道数不同，使用 1x1 卷积调整
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)  # 快捷路径
        out = self.conv_block(x)
        out = out + residual  # 残差连接
        out = nn.ReLU()(out)  # 加法后应用激活
        return out


# 下采样块（编码器部分）
class DownsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.block = ResidualConvBlock(in_ch, out_ch, norm_layer=norm_layer)
        self.downsample = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.block(x)
        y = x  # 保存跳跃连接的特征
        x = self.downsample(x)
        return x, y


# 上采样块（解码器部分，接受单编码器的跳跃连接）
class UpsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.block = ResidualConvBlock(in_ch, out_ch, norm_layer=norm_layer)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, y):
        x = self.upsample(x)
        x = torch.cat([x, y], dim=1)  # 连接单编码器的跳跃特征
        x = self.block(x)
        return x


# 单编码器-双解码器 Res-UNet
class SingleEncoderDualDecoderResUNet(nn.Module):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.color_ch = 3  # 输入颜色通道数（RGB）
        self.n_layers = 4  # 下采样/上采样层数
        self.spectral_ch = 29  # 高光谱输出通道数
        self.depth_ch = 1  # 深度图输出通道数
        self.base_ch = hparams.model_base_ch  # 基础通道数，例如 32
        norm_layer = nn.BatchNorm2d

        # 通道数列表
        self.encoder_channels = [self.base_ch * (2 ** i) for i in range(self.n_layers)]  # [32, 64, 128, 256]
        self.bottleneck_ch = self.base_ch * (2 ** self.n_layers)  # 512

        # 单编码器（下采样路径）
        self.encoder_input = nn.Sequential(
            nn.Conv2d(self.color_ch, self.base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.base_ch),
            nn.ReLU(),
        )
        self.encoder_downblocks = nn.ModuleList()
        for i in range(self.n_layers):
            in_ch = self.base_ch if i == 0 else self.encoder_channels[i - 1]
            out_ch = self.encoder_channels[i]
            self.encoder_downblocks.append(DownsampleBlock(in_ch, out_ch, norm_layer))

        # 瓶颈层（连接编码器和解码器）
        self.bottleneck = ResidualConvBlock(self.encoder_channels[-1], self.bottleneck_ch, norm_layer)

        # 高光谱解码器（上采样路径）
        self.spectral_upblocks = nn.ModuleList()
        for i in range(self.n_layers):
            in_ch = (self.bottleneck_ch if i == 0 else self.encoder_channels[self.n_layers - i]) + \
                    self.encoder_channels[self.n_layers - i - 1]
            out_ch = self.encoder_channels[self.n_layers - i - 1]
            self.spectral_upblocks.append(UpsampleBlock(in_ch, out_ch, norm_layer))
        self.spectral_output = nn.Conv2d(self.base_ch, self.spectral_ch, kernel_size=1, bias=True)

        # 深度解码器（上采样路径）
        self.depth_upblocks = nn.ModuleList()
        for i in range(self.n_layers):
            in_ch = (self.bottleneck_ch if i == 0 else self.encoder_channels[self.n_layers - i]) + \
                    self.encoder_channels[self.n_layers - i - 1]
            out_ch = self.encoder_channels[self.n_layers - i - 1]
            self.depth_upblocks.append(UpsampleBlock(in_ch, out_ch, norm_layer))
        self.depth_output = nn.Conv2d(self.base_ch, self.depth_ch, kernel_size=1, bias=True)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, captimgs, *args, **kwargs):
        b_sz, c_sz, h_sz, w_sz = captimgs.shape  # 输入形状，例如 [4, 3, 256, 256]

        # 单编码器（下采样）
        x = self.encoder_input(captimgs)
        enc_features = []
        for block in self.encoder_downblocks:
            x, y = block(x)
            enc_features.append(y)

        # 瓶颈层
        x = self.bottleneck(x)  # 形状：[b_sz, base_ch * (2^n_layers), h_sz/16, w_sz/16]

        # 高光谱解码器（上采样）
        x_spectral = x
        for i, block in enumerate(self.spectral_upblocks):
            x_spectral = block(x_spectral, enc_features[self.n_layers - 1 - i])
        est_spectrals = torch.sigmoid(self.spectral_output(x_spectral))  # 形状：[b_sz, 31, h_sz, w_sz]

        # 深度解码器（上采样）
        x_depth = x
        for i, block in enumerate(self.depth_upblocks):
            x_depth = block(x_depth, enc_features[self.n_layers - 1 - i])
        est_depthmaps = torch.sigmoid(self.depth_output(x_depth))  # 形状：[b_sz, 1, h_sz, w_sz]

        # 返回输出
        outputs = OutputsContainer(
            est_spectrals=est_spectrals,
            est_depthmaps=est_depthmaps
        )
        return outputs


# hparams = type('HParams', (), {'model_base_ch': 32})()
# model = SingleEncoderDualDecoderResUNet(hparams)
# input_tensor = torch.randn(4, 3, 256, 256)
# outputs = model(input_tensor)
# print(outputs.est_spectrals.shape)  # [4, 31, 256, 256]
# print(outputs.est_depthmaps.shape)  # [4, 1, 256, 256]