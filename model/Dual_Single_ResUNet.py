import torch
import torch.nn as nn
from collections import namedtuple

# 输出容器定义（只包含高光谱图像）
OutputsContainer = namedtuple('OutputContainer', field_names=['est_spectrals'])

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
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_block(x)
        out = out + residual
        out = nn.ReLU()(out)
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

# 上采样块（解码器部分，接受两个编码器的跳跃连接）
class UpsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.block = ResidualConvBlock(in_ch, out_ch, norm_layer=norm_layer)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, y1, y2):
        x = self.upsample(x)
        x = torch.cat([x, y1, y2], dim=1)  # 连接两个编码器的跳跃特征
        x = self.block(x)
        return x

# 双编码器-单解码器 Res-UNet（两个不同的输入）
class DualEncoderSingleDecoderResUNet(nn.Module):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.color_ch = 3  # 输入颜色通道数（RGB）
        self.n_layers = 4  # 下采样/上采样层数
        self.spectral_ch = 31  # 高光谱输出通道数
        self.base_ch = hparams.model_base_ch  # 基础通道数，例如 32
        norm_layer = nn.BatchNorm2d

        # 通道数列表
        self.encoder_channels = [self.base_ch * (2 ** i) for i in range(self.n_layers)]  # [32, 64, 128, 256]
        self.bottleneck_ch = self.base_ch * (2 ** self.n_layers)  # 512

        # 第一个编码器（下采样路径）
        self.encoder1_input = nn.Sequential(
            nn.Conv2d(self.color_ch, self.base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.base_ch),
            nn.ReLU(),
        )
        self.encoder1_downblocks = nn.ModuleList()
        for i in range(self.n_layers):
            in_ch = self.base_ch if i == 0 else self.encoder_channels[i - 1]
            out_ch = self.encoder_channels[i]
            self.encoder1_downblocks.append(DownsampleBlock(in_ch, out_ch, norm_layer))

        # 第二个编码器（下采样路径）
        self.encoder2_input = nn.Sequential(
            nn.Conv2d(self.color_ch, self.base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.base_ch),
            nn.ReLU(),
        )
        self.encoder2_downblocks = nn.ModuleList()
        for i in range(self.n_layers):
            in_ch = self.base_ch if i == 0 else self.encoder_channels[i - 1]
            out_ch = self.encoder_channels[i]
            self.encoder2_downblocks.append(DownsampleBlock(in_ch, out_ch, norm_layer))

        # 瓶颈层（连接编码器和解码器）
        self.bottleneck = ResidualConvBlock(self.encoder_channels[-1] * 2, self.bottleneck_ch, norm_layer)

        # 高光谱解码器（上采样路径）
        self.spectral_upblocks = nn.ModuleList()
        for i in range(self.n_layers):
            in_ch = (self.bottleneck_ch if i == 0 else self.encoder_channels[self.n_layers - i]) + 2 * \
                    self.encoder_channels[self.n_layers - i - 1]
            out_ch = self.encoder_channels[self.n_layers - i - 1]
            self.spectral_upblocks.append(UpsampleBlock(in_ch, out_ch, norm_layer))
        self.spectral_output = nn.Conv2d(self.base_ch, self.spectral_ch, kernel_size=1, bias=True)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, captimgs1, captimgs2, *args, **kwargs):
        # 检查两个输入的形状
        b_sz1, c_sz1, h_sz1, w_sz1 = captimgs1.shape  # 第一个输入的形状
        b_sz2, c_sz2, h_sz2, w_sz2 = captimgs2.shape  # 第二个输入的形状
        assert (b_sz1 == b_sz2 and c_sz1 == c_sz2 and h_sz1 == h_sz2 and w_sz1 == w_sz2), \
            "两个输入的形状必须相同"

        # 第一个编码器（下采样）
        x1 = self.encoder1_input(captimgs1)
        enc1_features = []
        for block in self.encoder1_downblocks:
            x1, y1 = block(x1)
            enc1_features.append(y1)

        # 第二个编码器（下采样）
        x2 = self.encoder2_input(captimgs2)
        enc2_features = []
        for block in self.encoder2_downblocks:
            x2, y2 = block(x2)
            enc2_features.append(y2)

        # 特征融合（在最低分辨率处）
        combined_features = torch.cat([x1, x2], dim=1)  # 形状：[b_sz, base_ch * (2^(n_layers-1)) * 2, h_sz/16, w_sz/16]
        x = self.bottleneck(combined_features)  # 形状：[b_sz, base_ch * (2^n_layers), h_sz/16, w_sz/16]

        # 高光谱解码器（上采样）
        x_spectral = x
        for i, block in enumerate(self.spectral_upblocks):
            x_spectral = block(x_spectral, enc1_features[self.n_layers - 1 - i], enc2_features[self.n_layers - 1 - i])
        est_spectrals = torch.sigmoid(self.spectral_output(x_spectral))  # 形状：[b_sz, 31, h_sz, w_sz]

        # 返回输出
        outputs = OutputsContainer(est_spectrals=est_spectrals)
        return outputs

# 测试代码
hparams = type('HParams', (), {'model_base_ch': 32})()
model = DualEncoderSingleDecoderResUNet(hparams)

# 两个不同的输入张量
input_tensor1 = torch.randn(4, 3, 256, 256)  # 第一个输入
input_tensor2 = torch.randn(4, 3, 256, 256)  # 第二个输入
outputs = model(input_tensor1, input_tensor2)
print(outputs.est_spectrals.shape)  # [4, 31, 256, 256]