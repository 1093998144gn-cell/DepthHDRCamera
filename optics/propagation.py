import torch
from matplotlib import pyplot as plt

from optics.light import LightWave
from utils.convfft import conv_fft
from utils.helper import *


def compute_pad_width(field, linear=True):
    """
    Compute the pad width of an array for FFT-based convolution
    Args:
        field: (B,Ch,R,C) complex tensor
        linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
    Returns:
        pad_width: pad-width tensor
    """

    if linear:
        R,C = field.shape[-2:]
        pad_width = (C-1, C-1, R-1, R-1)
    else:
        pad_width = (0,0,0,0)
    return pad_width

def unpad(field_padded, pad_width):
    """
    Unpad the already-padded complex tensor
    Args:
        field_padded: (B,Ch,R,C) padded complex tensor
        pad_width: pad-width tensor
    Returns:
        field: unpadded complex tensor
    """

    field = field_padded[...,pad_width[2]:-pad_width[3],pad_width[0]:-pad_width[1]]
    return field


class Propagation:
    def __init__(self, mode='S_FFT', device = 'cuda'):
        self.mode = mode
        self.device = device

    def forward(self, inputfield, z):
        if self.mode == 'D_FFT':
            return self.D_FFT(inputfield, z)
        elif self.mode == 'S_FFT':
            return self.S_FFT(inputfield, z)
        elif self.mode == 'T_FFT':
            return self.T_FFT(inputfield, z)
        else:
            raise NotImplementedError('Propagation mode not implemented')

    def D_FFT(self, light, z):
        M, N = light.resolution
        dx, dy = light.pitch

        fx = torch.fft.fftfreq(M, d=float(dx), device=self.device)
        fy = torch.fft.fftfreq(N, d=float(dy), device=self.device)

        Fx, Fy = torch.meshgrid(fx, fy, indexing='ij')



        wavelengths = light.wavelengths.view(1, -1, 1, 1).to(self.device)  # [1, num_wavelengths, 1, 1]
        k = 2 * torch.pi / wavelengths  # [1, num_wavelengths, 1, 1]

        # Compute propagation phase
        kx = 2 * torch.pi * Fx
        ky = 2 * torch.pi * Fy
        kz = torch.sqrt(k ** 2 - kx ** 2 - ky ** 2 + 0j)  # Add 0j for complex support
        H = torch.exp(1j * kz * z)

        lightwave = light.get_wave()

        lightwave_fft = torch.fft.fft2(lightwave, dim=(-2, -1))

        #角谱传播
        lightwave_fft = lightwave_fft * H

        lightwave = torch.fft.ifft2(lightwave_fft, dim=(-2, -1))

        return lightwave

    def S_FFT(self, light, z):
        lightwave = light.get_wave()

        M, N = light.resolution
        dx, dy = light.pitch

        x = torch.cat((torch.arange(-M // 2 + 1, 1, dtype=torch.float32, device=self.device) * dx,
                      torch.arange(0, M // 2, dtype=torch.float32, device=self.device) * dx))
        y = torch.cat((torch.arange(-N // 2 + 1, 1, dtype=torch.float32, device=self.device) * dy,
                      torch.arange(0, N // 2, dtype=torch.float32, device=self.device) * dy))
        x = x.double()
        y = y.double()
        X, Y = torch.meshgrid(x, y, indexing='ij')

        r_squared = X ** 2 + Y ** 2  # [M, N]

        wavelengths = light.wavelengths.view(1, -1, 1, 1).to(self.device)  # [1, num_wavelengths, 1, 1]
        k = 2 * torch.pi / wavelengths  # [1, num_wavelengths, 1, 1]

        fresnel_factor = torch.exp(1j * k * z) * torch.exp(1j * k * r_squared / (2 * z))

        lightwave = lightwave * fresnel_factor

        lightwave = torch.fft.fft2(lightwave, dim=(-2, -1))  # 在最后两个维度上计算 FFT
        lightwave = torch.fft.fftshift(lightwave, dim=(-2, -1))

        return lightwave


    def T_FFT(self, light, z):
        M, N = light.resolution
        dx, dy = light.pitch

        x = torch.linspace(-M // 2 * dx, M // 2 * dx, M)
        y = torch.linspace(-N // 2 * dy, N // 2 * dy, N)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        r_squared = X ** 2 + Y ** 2  # [M, N]

        wavelengths = light.wavelengths.view(1, -1, 1, 1).to(self.device)
        k = 2 * torch.pi / wavelengths

        fresnel_kernel = torch.exp(1j * k * z) * torch.exp(1j * k * r_squared / (2 * z))

        lightwave = light.get_wave()

        pad_width = compute_pad_width(lightwave, linear=True)

        fresnel_kernel = fresnel_kernel.expand(lightwave.shape)
        field_propagated = conv_fft(lightwave, fresnel_kernel, pad_width)

        return field_propagated




#
# def gaussian_beam(M, N, dx, dy, w0):
#     """
#     生成一个二维高斯光束 E(x,y) = exp(- (x² + y²) / w0² )
#     """
#     x = torch.linspace(-M//2, M//2-1, M) * dx
#     y = torch.linspace(-N//2, N//2-1, N) * dy
#     X, Y = torch.meshgrid(x, y, indexing='ij')
#     E = torch.exp(-(X**2 + Y**2) / w0**2)
#     return E
#
# def plot_field(field, title):
#     """
#     绘制光场强度
#     """
#     plt.figure(figsize=(5,5))
#     plt.imshow(torch.abs(field.cpu())**2, cmap='inferno', extent=(-5,5,-5,5))
#     plt.colorbar()
#     plt.title(title)
#     plt.show()
#
# # 参数
# M, N = 256, 256  # 空间分辨率
# dx, dy = 10e-6, 10e-6  # 像素间距 10 µm
# w0 = 100e-6  # 高斯光束腰宽 100 µm
# wavelength = torch.tensor([632.8e-9])  # 红光 632.8 nm
# z = 0.01  # 传播 10 mm
#
# input_field = gaussian_beam(M, N, dx, dy, w0).to('cuda')
# phase = input_field.angle().unsqueeze(0).unsqueeze(0).to('cuda')
# amplitude = torch.abs(input_field).unsqueeze(0).unsqueeze(0).to('cuda')
#
# light = LightWave(wavelength, (M,N), (dx,dy), phase=phase, amplitude=amplitude)
#
# # 生成高斯光束
#   # 传输到 GPU
#
# # 创建 Propagation 实例
# propagator = Propagation(mode='D_FFT', device='cuda')
# #
# # # 传播
# output_field = propagator.forward(light, z, wavelengths=torch.tensor([wavelength], device='cuda'))
# #
# # # 绘制结果
# plot_field(input_field, 'Initial Field')
# plot_field(output_field.squeeze(), 'Propagated Field (D_FFT)')

