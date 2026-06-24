import torch
import torch.nn as nn

from model.unet import UNet


from collections import namedtuple

OutputsContainer = namedtuple('OutputContainer', field_names=['est_spectrals'])

class SpectralModel(nn.Module):

    def __init__(self, spectral_ch, *args, **kargs):
        super().__init__()
        color_ch = 3
        n_layers = 4
        spectral_ch = spectral_ch
        base_ch = 32  #model_base_ch=32，这是Unet的输入通道
        input_ch = color_ch  #输入的只有相机看到的图像
        base_input_layers = nn.Sequential(
            nn.Conv2d(input_ch, input_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(input_ch),
            nn.ReLU(),
            nn.Conv2d(input_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
        )  #这里相当于是设了两重卷积，这里的nn只定义计算

        # Without the preinverse input, it has ((color_ch * preinv_input_ch) + preinv_input_ch * 2) more parameters than
        # with the preinverse input. (255 params)

        input_layers = base_input_layers


        output_layers = nn.Sequential(
            nn.Conv2d(base_ch, spectral_ch, kernel_size=1, bias=True)
        )
        #这里的输出层相当于是从base_ch直接到color_ch和depth_ch，从32通道到4通道

        self.decoder = nn.Sequential(
            input_layers,
            UNet(
                channels=[base_ch, base_ch, 2 * base_ch, 2 * base_ch, 4 * base_ch, 4 * base_ch],
                n_layers=n_layers,
            ),
            output_layers,
        )
        #这个decoder其实就是定义了数据在里面是怎么算
        #输入和输出之间夹一个Unet

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, captimgs, *args, **kargs):
        b_sz, c_sz, h_sz, w_sz = captimgs.shape
        # 4, 3, 16, 1200, 1920, b_sz可以理解为batch_size
        # if self.preinverse:
        #     inputs = torch.cat([captimgs.unsqueeze(2), pinv_volumes], dim=2)
        # else:
        inputs = captimgs.unsqueeze(2)
        est = torch.sigmoid(self.decoder(inputs.reshape(b_sz, -1, h_sz, w_sz)))
        #这里的input是[4, 51, 1200, 1920]，第一个理解为batch_size,第二个为通道数

        #est是[4, 4, 1200, 1920]
        # est_images = est[:, :-1]
        # est_depthmaps = est[:, [-1]]

        outputs = OutputsContainer(
            est_spectrals = est
        )
        return outputs
