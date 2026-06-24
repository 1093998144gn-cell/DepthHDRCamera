import torch
import torch.nn as nn
from collections import namedtuple

# 输出容器定义
OutputsContainer = namedtuple('OutputContainer', field_names=['est_spectrals', 'est_depthmaps'])

# 残差卷积块
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
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_block(x)
        out = out + residual
        out = nn.ReLU()(out)
        return out

# 下采样块（编码器）
class DownsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.block = ResidualConvBlock(in_ch, out_ch, norm_layer=norm_layer)
        self.downsample = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.block(x)
        y = x  # 保存跳跃连接
        x = self.downsample(x)
        return x, y

# 上采样块（解码器）
class UpsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.block = ResidualConvBlock(in_ch, out_ch, norm_layer=norm_layer)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x, y):
        x = self.upsample(x)
        x = torch.cat([x, y], dim=1)
        residual = self.shortcut(x)
        x = self.block(x)
        x = x + residual
        x = nn.ReLU()(x)
        return x

# 单编码器-单解码器 Res-UNet
class SingleEncoderSingleDecoderResUNet(nn.Module):
    def __init__(self, spectral_ch, *args, **kwargs):
        super().__init__()
        self.color_ch = 3  # 输入 RGB 通道
        self.n_layers = 4  # 下采样/上采样层数
        self.spectral_ch = spectral_ch  # 高光谱通道数（例如 25）
        self.depth_ch = 1  # 深度通道数
        self.output_ch = self.spectral_ch + self.depth_ch  # 联合输出通道
        self.base_ch = 64  # 基础通道数，增加容量
        norm_layer = nn.BatchNorm2d

        # 通道数
        self.encoder_channels = [self.base_ch * (2 ** i) for i in range(self.n_layers)]  # [64, 128, 256, 512]
        self.bottleneck_ch = self.base_ch * (2 ** self.n_layers)  # 1024

        # 编码器
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

        # 瓶颈层
        self.bottleneck = ResidualConvBlock(self.encoder_channels[-1], self.bottleneck_ch, norm_layer)

        # 解码器
        self.upblocks = nn.ModuleList()
        for i in range(self.n_layers):
            in_ch = (self.bottleneck_ch if i == 0 else self.encoder_channels[self.n_layers - i]) + \
                    self.encoder_channels[self.n_layers - i - 1]
            out_ch = self.encoder_channels[self.n_layers - i - 1]
            self.upblocks.append(UpsampleBlock(in_ch, out_ch, norm_layer))
        self.output = nn.Conv2d(self.base_ch, self.output_ch, kernel_size=1, bias=True)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.output:
                    nn.init.xavier_normal_(m.weight)  # Xavier for output layer
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, captimgs, *args, **kwargs):
        # 编码器
        x = self.encoder_input(captimgs)
        enc_features = []
        for block in self.encoder_downblocks:
            x, y = block(x)
            enc_features.append(y)

        # 瓶颈层
        x = self.bottleneck(x)

        # 解码器
        for i, block in enumerate(self.upblocks):
            x = block(x, enc_features[self.n_layers - 1 - i])
        est_outputs = self.output(x)  # 形状：[batch_size, spectral_ch + 1, h_sz, w_sz]

        # 应用激活：高光谱用 sigmoid，深度用线性
        est_spectrals = torch.sigmoid(est_outputs[:, :self.spectral_ch, :, :])
        est_depthmaps = est_outputs[:, self.spectral_ch:, :, :]  # 线性输出

        return OutputsContainer(est_spectrals=est_spectrals, est_depthmaps=est_depthmaps)

# 测试代码
if __name__ == "__main__":
    model = SingleEncoderSingleDecoderResUNet(spectral_ch=25)
    input_tensor = torch.randn(4, 3, 256, 256)
    outputs = model(input_tensor)
    print(f"Output est_spectrals: {outputs.est_spectrals.shape}")  # 应为 [4, 25, 256, 256]
    print(f"Output est_depthmaps: {outputs.est_depthmaps.shape}")  # 应为 [4, 1, 256, 256]