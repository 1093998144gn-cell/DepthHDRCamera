import torch
import torch.nn as nn
from model.unet import UNet

from collections import namedtuple
OutputsContainer = namedtuple('OutputContainer', field_names=['est_images', 'est_depthmaps'])

class SimpleModel(nn.Module):

    def __init__(self, hparams, *args, **kargs):
        super().__init__()
        depth_ch = 1
        color_ch = hparams.color_ch
        n_layers = 4
        base_ch = hparams.model_base_ch
        input_ch = color_ch  # 输入的只有相机看到的图像
        base_input_layers = nn.Sequential(
            nn.Conv2d(input_ch, input_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(input_ch),
            nn.ReLU(),
            nn.Conv2d(input_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
        )
        # Without the preinverse input, it has ((color_ch * preinv_input_ch) + preinv_input_ch * 2) more parameters than
        # with the preinverse input. (255 params)
        input_layers = base_input_layers


        output_layers = nn.Sequential(
            nn.Conv2d(base_ch, color_ch + depth_ch, kernel_size=1, bias=True)
        )
        self.decoder = nn.Sequential(
            input_layers,
            UNet(
                channels=[base_ch, base_ch, 2 * base_ch, 2 * base_ch, 4 * base_ch, 4 * base_ch],
                n_layers=n_layers,
            ),
            output_layers,
        )

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
        inputs = captimgs.unsqueeze(2)

        est = torch.sigmoid(self.decoder(inputs.reshape(b_sz, -1, h_sz, w_sz)))
        est_images = est[:, :-1]
        est_depthmaps = est[:, [-1]]
        outputs = OutputsContainer(
            est_images=est_images,
            est_depthmaps=est_depthmaps,
        )
        return outputs
