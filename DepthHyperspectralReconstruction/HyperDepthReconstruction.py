#####用于测试DOE在syn数据集上的效果

import copy
import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from torchvision.utils import save_image

from optics.DOECamera import DOECamera
from utils.loss import Vgg16PerceptualLoss, SpectralAngleMapperLoss, SpectralSmoothnessLoss
from utils.debayer import Debayer3x3, to_bayer
from utils.helper import *
from collections import namedtuple
from model.Single_ResUNet import SingleEncoderSingleDecoderResUNet



DepthHyperspectralOutputs = namedtuple('DepthHyperspectralOutputs',
                             field_names=['target_spectrals', 'est_spectrals', 'sensor_images', 'psf', 'target_rgb_images', 'est_rgb_images',
                                          'target_depthmaps', 'est_depthmaps'])


class DOEHyperDepthCamera(pl.LightningModule):
    def __init__(self, hparam, log_dir = None):
        super(DOEHyperDepthCamera, self).__init__()
        self.hparam = copy.deepcopy(hparam)
        self.save_hyperparameters(self.hparam)
        self.__build_model()
        self.log_dir = log_dir

    def __build_model(self):
        hparam = self.hparam
        optimize_optics = hparam.optimize_optics
        self.camera = DOECamera(optimize_optics)
        self.decoder = SingleEncoderSingleDecoderResUNet(spectral_ch=self.camera.num_wavelengths)
        self.debayer = Debayer3x3()
        self.image_lossfn = Vgg16PerceptualLoss()
        self.spectral_lossfn = SpectralAngleMapperLoss()
        self.spectralsmooth_lossfn = SpectralSmoothnessLoss()
        self.depth_lossfn = torch.nn.MSELoss()

    def configure_optimizers(self):
        params = [
            {'params': self.camera.parameters(), 'lr': self.hparams.optics_lr},
            {'params': self.decoder.parameters(), 'lr': self.hparams.cnn_lr},
        ]
        optimizer = torch.optim.Adam(params)
        return optimizer

    def training_step(self, samples, batch_idx):
        target_spectrals = samples['refimg']
        target_depthmaps = samples['disp']


        outputs = self.forward(target_spectrals, target_depthmaps)
        data_loss, loss_logs = self.__compute_loss(outputs)
        loss_logs = {f'train_loss/{key}': val for key, val in loss_logs.items()}

        misc_logs = {   #监控训练是否异常的统计量
            'train_misc/target_spectrals_max': target_spectrals.max(),
            'train_misc/target_spectrals_min': target_spectrals.min(),
            'train_misc/est_spectrals_max': outputs.est_spectrals.max(),
            'train_misc/est_spectrals_min': outputs.est_spectrals.min(),
            'train_misc/target_rgb_images_max': outputs.target_rgb_images.max(),
            'train_misc/target_rgb_images_min': outputs.target_rgb_images.min(),
            'train_misc/est_rgb_images_max': outputs.est_rgb_images.max(),
            'train_misc/est_rgb_images_min': outputs.est_rgb_images.min(),
            'train_misc/sensor_images_max': outputs.sensor_images.max(),
            'train_misc/sensor_images_min': outputs.sensor_images.min(),
        }

        if self.hparams.optimize_optics:
            misc_logs.update({
                'optics/heightmap_max': self.camera.param.max(),
                'optics/heightmap_min': self.camera.param.min(),
            })

        logs = {}
        logs.update(loss_logs)
        if batch_idx == 0:
             self.__log_images(outputs, 'train')
        self.log_dict(logs)


        print(f"Batch {batch_idx} - Loss: {data_loss.item():.6f}")
        return data_loss



    def validation_step(self, samples, batch_idx):
        target_spectrals = samples['refimg']
        target_depthmaps = samples['disp']

        # 前向传播
        outputs = self.forward(target_spectrals, target_depthmaps)

        # 计算验证损失，与训练阶段一致
        data_loss, loss_logs = self.__compute_loss(outputs)
        loss_logs = {f'val_loss/{key}': val for key, val in loss_logs.items()}

        # 记录损失
        self.log_dict(loss_logs, on_step=False, on_epoch=True)
        self.log('val_loss', data_loss, on_epoch=True, prog_bar=True)

        print(f"Batch {batch_idx} - Validation Loss: {data_loss.item():.6f}")
        if batch_idx == 0:
            self.__log_images(outputs, 'validation')
        return data_loss



    def forward(self, images, depthmaps = None):
        #目标RGB图像
        target_rgb_images = hyperspectral_to_rgb(self.camera.response_function, images, self.camera.num_wavelengths)

        target_spectrals = images
        target_depthmaps = depthmaps

        #生成传感器图像
        sensor_images, psf = self.camera(target_spectrals, target_depthmaps)

        # add some Gaussian noise
        dtype = images.dtype
        device = images.device
        noise_sigma_min = self.hparam.noise_sigma_min
        noise_sigma_max = self.hparam.noise_sigma_max
        noise_sigma = (noise_sigma_max - noise_sigma_min) * torch.rand((sensor_images.shape[0], 1, 1, 1), device=device,
                                                                       dtype=dtype) + noise_sigma_min

        # without Bayer
        if not torch.tensor(self.hparam.bayer):
            sensor_images = sensor_images + noise_sigma * torch.randn(sensor_images.shape, device=device, dtype=dtype)
        else:
            sensor_images_bayer = to_bayer(sensor_images)
            sensor_images_bayer = sensor_images_bayer + noise_sigma * torch.randn(sensor_images_bayer.shape, device=device,
                                                                        dtype=dtype)
            sensor_images = self.debayer(sensor_images_bayer)

        # Crop the boundary artifact of DFT-based convolution
        sensor_images = crop_boundary(sensor_images, self.camera.crop_width) #裁掉边缘的伪影

        sensor_images = sensor_images.type(torch.float32)
        # Feed the cropped images to CNN
        model_outputs = self.decoder(captimgs=sensor_images) #送入神经网络重建

        # Require twice cropping because the image formation also crops the boundary.
        target_spectrals = crop_boundary(images, 2 * self.camera.crop_width) #再次裁剪，保证尺寸一致
        target_rgb_images = crop_boundary(target_rgb_images, 2 * self.camera.crop_width)

        target_depthmaps = crop_boundary(target_depthmaps, 2 * self.camera.crop_width)

        sensor_images = crop_boundary(sensor_images, self.camera.crop_width)

        est_spectrals = crop_boundary(model_outputs.est_spectrals, self.camera.crop_width)
        est_rgb_images = hyperspectral_to_rgb(self.camera.response_function, est_spectrals, self.camera.num_wavelengths)

        est_depthmaps = crop_boundary(model_outputs.est_depthmaps, self.camera.crop_width)

        psf = crop_boundary(psf,2 * self.camera.crop_width)


        outputs = DepthHyperspectralOutputs(
            target_spectrals=target_spectrals,
            est_spectrals=est_spectrals,
            sensor_images=sensor_images,
            target_rgb_images=target_rgb_images,
            est_rgb_images=est_rgb_images,
            psf=psf,
            target_depthmaps=target_depthmaps,
            est_depthmaps=est_depthmaps
        )
        return outputs


    def __combine_loss(self, spectral_loss, image_loss, spectralsmooth_loss, depth_loss):

        return self.hparam.spectral_loss_weight * spectral_loss + \
            self.hparam.image_loss_weight * image_loss + \
            self.hparam.spectralsmooth_loss_weight * spectralsmooth_loss + \
            self.hparam.depth_loss_weight * depth_loss

    #不带PSF正则化
    def __compute_loss(self, outputs):

        spectral_loss = self.spectral_lossfn(outputs.est_spectrals, outputs.target_spectrals)

        rgb_loss = self.image_lossfn(outputs.est_rgb_images, outputs.target_rgb_images)

        spectralsmooth_loss = self.spectralsmooth_lossfn(outputs.est_spectrals)

        depth_loss = self.depth_lossfn(outputs.est_depthmaps, outputs.target_depthmaps)

        total_loss = self.__combine_loss(spectral_loss, rgb_loss, spectralsmooth_loss, depth_loss)

        logs = {
            'total_loss': total_loss,
            'spectral_loss': spectral_loss,
            'rgb_loss': rgb_loss,
            'spectralsmooth_loss': spectralsmooth_loss,
            'depth_loss': depth_loss
        }
        return total_loss, logs


    @torch.no_grad()
    def __log_images(self, outputs, stage: str):

        save_dir = 'F:\Result\HyperspectralDepth'
        stage_dir = os.path.join(save_dir, stage)
        os.makedirs(stage_dir, exist_ok=True)

        target_spectrals = outputs.target_spectrals
        est_spectrals = outputs.est_spectrals
        target_spectrals = save_wavelength_rgb_tensor(target_spectrals[0], wavelengths=self.camera.wavelengths)
        est_spectrals = save_wavelength_rgb_tensor(est_spectrals[0], wavelengths=self.camera.wavelengths)
        target_spectrals = target_spectrals.permute(0, 3, 1, 2).type(
            torch.float32)  # [31, 384, 384, 3] -> [31, 3, H, W]
        est_spectrals = est_spectrals.permute(0, 3, 1, 2).type(
            torch.float32)  # [31, 384, 384, 3] -> [31, 3, H, W]
        target_spectrals = target_spectrals / 255.0
        est_spectrals = est_spectrals / 255.0
        # Combine RGB images into a grid
        target_spectrals_grid = make_grid(target_spectrals, nrow=6, normalize=True)
        est_spectrals_grid = make_grid(est_spectrals, nrow=6, normalize=True)
        self.logger.experiment.add_image(f'{stage}/target_spectrals', target_spectrals_grid, self.current_epoch)
        self.logger.experiment.add_image(f'{stage}/est_spectrals', est_spectrals_grid, self.current_epoch)

        save_image(target_spectrals_grid, os.path.join(stage_dir, f'target_spectrals_epoch_{self.current_epoch}.png'))
        save_image(est_spectrals_grid, os.path.join(stage_dir, f'est_spectrals_epoch_{self.current_epoch}.png'))


        sensor_images = outputs.sensor_images
        sensor_images_grid = make_grid(sensor_images[0])
        self.logger.experiment.add_image(f"{stage}/sensor_images", sensor_images_grid, self.current_epoch)
        save_image(sensor_images_grid, os.path.join(stage_dir, f'sensor_images_epoch_{self.current_epoch}.png'))


        target_rgb_images = outputs.target_rgb_images
        est_rgb_images = outputs.est_rgb_images
        rgb_grid = make_grid([target_rgb_images[0], est_rgb_images[0]], nrow=2)
        self.logger.experiment.add_image(f"{stage}/RGB Comparison", rgb_grid, self.current_epoch)
        save_image(rgb_grid, os.path.join(stage_dir, f'RGB Comparison_epoch_{self.current_epoch}.png'))


        target_depthmaps = outputs.target_depthmaps
        est_depthmaps = outputs.est_depthmaps
        depth_grid = make_grid([target_depthmaps[0], est_depthmaps[0]], nrow=2)
        self.logger.experiment.add_image(f"{stage}/depthmaps", depth_grid, self.current_epoch)
        save_image(depth_grid, os.path.join(stage_dir, f'depthmaps_epoch_{self.current_epoch}.png'))



        psf = outputs.psf    #torch.Size([1, 31, 256, 256])
        print('psf', psf.size())
        psf_spectral_grid = make_grid(psf[5, :, :, :].unsqueeze(1), nrow=6, normalize=True)
        self.logger.experiment.add_image(f"{stage}optics/psf_spectral", psf_spectral_grid, self.current_epoch)
        psf_depth_grid = make_grid(psf[:, 12, :, :].unsqueeze(1), nrow=6, normalize=True)
        self.logger.experiment.add_image(f"{stage}optics/psf_depth", psf_depth_grid, self.current_epoch)

        save_image(psf_spectral_grid, os.path.join(stage_dir, f'psf_spectral_epoch_{self.current_epoch}.png'))
        save_image(psf_depth_grid, os.path.join(stage_dir, f'psf_depth_epoch_{self.current_epoch}.png'))


    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        #model training parameters
        parser.add_argument('--cnn_lr', type=float, default=1e-4)
        parser.add_argument('--optics_lr', type=float, default=1e-10)

        # loss parameters
        parser.add_argument('--spectral_loss_weight', type=float, default=1.0)
        parser.add_argument('--image_loss_weight', type=float, default=1.0)
        parser.add_argument('--psf_loss_weight', type=float, default=1.0)
        parser.add_argument('--spectralsmooth_loss_weight', type=float, default=1.0)
        parser.add_argument('--depth_loss_weight', type=float, default=1.0)
        parser.add_argument('--psf_size', type=int, default=64)

        # model parameters
        parser.add_argument('--model_base_ch', type=int, default=32)

        # optics parameters
        parser.add_argument('--noise_sigma_min', type=float, default=0.001)
        parser.add_argument('--noise_sigma_max', type=float, default=0.005)
        parser.add_argument('--diffraction_efficiency', type=float, default=0.7)
        parser.add_argument('--bayer', dest='bayer', action='store_true')
        parser.add_argument('--no-bayer', dest='bayer', action='store_false')
        parser.set_defaults(bayer=False)
        parser.add_argument('--optimize_optics', dest='optimize_optics', action='store_true')
        parser.add_argument('--no-optimize_optics', dest='optimize_optics', action='store_false')
        parser.set_defaults(optimize_optics=True)

        return parser

