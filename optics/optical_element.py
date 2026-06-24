import math

import torch
from abc import ABC, abstractmethod

from scipy.io import savemat

from metasurface.Pola_multi_rectangleatom import RectangleMLP
from metasurface.circleatom import CircleMLP


class OpticalElement(ABC):
    def __init__(self, resolution, pitch, wavelengths, device='cuda'):
        """
        初始化光学元件的基本属性

        :param resolution: 光学元件的分辨率 (例如：像素数或单位长度的空间分辨率)
        :param wavelengths: 使用的光波长 (单位: 米)
        :param unit_size: 光学元件的单元尺寸 (单位: 米)
        """
        self.resolution = resolution  # 分辨率
        self.pitch = pitch  # 像素间距
        self.wavelengths = wavelengths  # 波长
        self.device = device

    @abstractmethod
    def get_modulation(self):
        pass

    @abstractmethod
    def forward(self, inputfield):
        pass

class RectangleMetasurface(OpticalElement):
    def __init__(self, resolution, pitch, wavelengths, model_path, size_x=None, size_y=None):
        """
        初始化超材料光学元件

        :param resolution: 光学元件的分辨率 (例如：像素数或单位长度的空间分辨率)
        :param wavelength: 使用的光波长 (单位: 米)
        :param size: 超单元的尺寸分布 (单位: 米)
        :param model_path: 训练好的超表面模型路径
        """
        super(RectangleMetasurface, self).__init__(resolution, wavelengths, pitch)
        M, N = self.resolution
        self.size_x = torch.rand(M, N, device='cuda') if size_x is None else size_x
        self.size_y = torch.rand(M, N, device='cuda') if size_y is None else size_y

        self.atom_model = RectangleMLP()
        self.atom_model.load_state_dict(torch.load(model_path))
        self.atom_model.to(self.device)
        for param in self.atom_model.parameters():
            param.requires_grad = False

    def get_modulation(self):
        """
        获取超材料光学元件的相位调制矩阵
        :return: 相位调制矩阵 (shape: [resolution, resolution])
        """
        phasex_list, transx_list = [], []
        phasey_list, transy_list = [], []
        M, N = self.resolution
        # 如果自由，N*N; 如果对称， N//2
        for wavelength in self.wavelengths:
            w_norm = (wavelength - self.wavelengths[0]) / (self.wavelengths[-1] - self.wavelengths[0])
            w_norm = w_norm.repeat(M * N)
            flat_sizex = self.size_x.view(-1)
            flat_sizey = self.size_y.view(-1)

            inputsx = torch.stack([w_norm, flat_sizex, flat_sizey], dim=1)

            inputsy = torch.stack([w_norm, flat_sizey, flat_sizex], dim=1)

            outputsx = self.atom_model(inputsx)  # 通过超表面模型预测输出
            outputsy = self.atom_model(inputsy)

            phasex_w = torch.atan2(outputsx[:, 0], outputsx[:, 1]).view(1, M, N)  # 计算相位并重塑为 [1, M, N]
            transx_w = outputsx[:, 2].view(1, M, N)  # 获取透射率并重塑为 [1, M, N]

            phasey_w = torch.atan2(outputsy[:, 0], outputsy[:, 1]).view(1, M, N)  # 计算相位并重塑为 [1, M, N]
            transy_w = outputsy[:, 2].view(1, M, N)  # 获取透射率并重塑为 [1, M, N]

            phasex_list.append(phasex_w)
            transx_list.append(transx_w)

            phasey_list.append(phasey_w)
            transy_list.append(transy_w)

        phasex = torch.cat(phasex_list, dim=0)
        transx = torch.cat(transx_list, dim=0)

        phasey = torch.cat(phasey_list, dim=0)
        transy = torch.cat(transy_list, dim=0)

        complex_x = transx * torch.exp(1j * phasex)
        complex_y = transy * torch.exp(1j * phasey)

        complex_x = complex_x.unsqueeze(0)
        complex_y = complex_y.unsqueeze(0)

        return complex_x, complex_y


    def forward(self, inputfield):
        M, N = self.resolution
        assert(inputfield.shape[-1] == N and inputfield.shape[-2] == M)

        modulation_x, modulation_y = self.get_modulation()

        outputfieldx = inputfield * modulation_x
        outputfieldy = inputfield * modulation_y

        return outputfieldx, outputfieldy

class CircleMetasurface(OpticalElement):
    def __init__(self, resolution, pitch, wavelengths, model_path, size=None):
        """
        初始化超材料光学元件
        :param resolution: 光学元件的分辨率 (例如：像素数或单位长度的空间分辨率)
        :param wavelength: 使用的光波长 (单位: 米)
        :param size: 超单元的尺寸分布 (单位: 米)
        :param model_path: 训练好的超表面模型路径
        """
        super(CircleMetasurface, self).__init__(resolution, wavelengths, pitch)
        M, N = self.resolution
        self.size = torch.rand(M, N, device='cuda') if size is None else size

        self.atom_model = CircleMLP()
        self.atom_model.load_state_dict(torch.load(model_path))
        self.atom_model.to(self.device)
        for param in self.atom_model.parameters():
            param.requires_grad = False

    def get_modulation(self):
        """
        获取超材料光学元件的相位调制矩阵
        :return: 相位调制矩阵 (shape: [resolution, resolution])
        """
        phase_list, trans_list = [], []
        M, N = self.resolution
        # 如果自由，N*N; 如果对称， N//2
        for wavelength in self.wavelengths:
            w_norm = (wavelength - self.wavelengths[0]) / (self.wavelengths[-1] - self.wavelengths[0])
            w_norm = w_norm.repeat(M * N)
            flat_size = self.size.view(-1)

            inputs = torch.stack([w_norm, flat_size], dim=1)
            outputs = self.atom_model(inputs)  # 通过超表面模型预测输出

            phase_w = torch.atan2(outputs[:, 0], outputs[:, 1]).view(1, M, N)  # 计算相位并重塑为 [1, M, N]
            trans_w = outputs[:, 2].view(1, M, N)  # 获取透射率并重塑为 [1, M, N]

            phase_list.append(phase_w)
            trans_list.append(trans_w)

        phase = torch.cat(phase_list, dim=0)
        trans = torch.cat(trans_list, dim=0)

        modulation = trans * torch.exp(1j * phase)
        modulation = modulation.unsqueeze(0)
        return modulation

    def forward(self, inputfield):
        M, N = self.resolution
        assert (inputfield.shape[-1] == N and inputfield.shape[-2] == M)

        modulation= self.get_modulation()

        outputfield = inputfield * modulation

        return outputfield

class DOE(OpticalElement):
    def __init__(self, resolution, pitch, wavelengths, heightmap=None, material_refractive_index=None):
        """
        初始化DOE元件
        :param resolution: 光学元件的分辨率 (例如：像素数或单位长度的空间分辨率)
        :param wavelength: 使用的光波长 (单位: 米)
        :param heightmap: DOE的高度分布 (单位: 米)
        """
        super(DOE, self).__init__(resolution, pitch, wavelengths)
        M, N = self.resolution
        self.heightmap = torch.zeros(M, N, device='cuda') if heightmap is None else heightmap
        self.material_refractive_index = material_refractive_index


    def set_heightmap(self, heightmap):
        self.heightmap = heightmap

    def refractive_index(self, wavelength, a=1.5314, b=6.5707e-15, c=2.0328e-28):
        """Cauchy's equation - dispersion formula
        Default coefficients are for NOA61.
        https://refractiveindex.info/?shelf=other&book=Optical_adhesives&page=Norland_NOA61
        """
        return a + b / wavelength ** 2 + c / wavelength ** 4

    def heightmap_to_phase(self, height, wavelength, refractive_index):
        """将高度图转换为相位"""
        height = height.unsqueeze(0)
        wavelength = wavelength.view(-1, 1, 1)
        refractive_index = refractive_index.view(-1, 1, 1)
        return height * (2 * math.pi / wavelength) * (refractive_index - 1)

    def get_modulation(self):
        if self.material_refractive_index is None:
            refractive_index = self.refractive_index(self.wavelengths)
        else:
            refractive_index = torch.full_like(self.wavelengths, float(self.material_refractive_index))
        phase = self.heightmap_to_phase(self.heightmap, self.wavelengths, refractive_index)
        trans = torch.ones_like(phase)
        modulation = trans * torch.exp(1j*phase)
        modulation = modulation.unsqueeze(0)
        return modulation

    def forward(self, inputfield):
        M, N = self.resolution
        assert (inputfield.shape[-1] == N and inputfield.shape[-2] == M)
        modulation= self.get_modulation()
        outputfield = inputfield * modulation
        return outputfield

class lens(OpticalElement):
    def __init__(self, resolution, pitch, wavelengths, focal_length):
        """
        初始化lens元件
        :param resolution: 光学元件的分辨率 (例如：像素数或单位长度的空间分辨率)
        :param wavelength: 使用的光波长 (单位: 米)
        """
        super(lens, self).__init__(resolution, pitch, wavelengths)
        self.focal_length = focal_length

    def get_modulation(self):
        wavelengths = self.wavelengths.view(1, -1, 1, 1).to(self.device)  # [1, num_wavelengths, 1, 1]
        k=2 * math.pi / wavelengths  # [1, num_wavelengths, 1, 1]

        M, N = self.resolution
        dx, dy = self.pitch

        x = torch.arange(-M // 2, M // 2, dtype=torch.float32, device=self.device) * dx
        y = torch.arange(-N // 2, N // 2, dtype=torch.float32, device=self.device) * dy
        X, Y = torch.meshgrid(x, y, indexing='ij')

        print("focal_length:",self.focal_length)

        phase = -k * (X**2 + Y**2) / (2 * self.focal_length)
        trans = torch.ones_like(phase)

        modulation = trans * torch.exp(1j * phase)
        return modulation

    def forward(self, inputfield):
        M, N = self.resolution
        assert (inputfield.shape[-1] == N and inputfield.shape[-2] == M)
        modulation = self.get_modulation()
        outputfield = inputfield * modulation

        return outputfield

class aperture(OpticalElement):
    def __init__(self, resolution, pitch, wavelengths):
        """
        初始化aperture元件
        :param resolution: 光学元件的分辨率 (例如：像素数或单位长度的空间分辨率)
        :param wavelength: 使用的光波长 (单位: 米)
        """
        super(aperture, self).__init__(resolution, pitch, wavelengths)


    def get_modulation(self):
        M, N = self.resolution
        dx, dy = self.pitch
        R = min(M//2, N//2) * min(dx, dy)
        x = torch.arange(-M // 2, M // 2, dtype=torch.float32, device=self.device) * dx
        y = torch.arange(-N // 2, N // 2, dtype=torch.float32, device=self.device) * dy
        X, Y = torch.meshgrid(x, y, indexing='ij')
        r = torch.sqrt(X ** 2 + Y ** 2)
        aperture = (r <= R).float()
        phase = torch.zeros_like(aperture)
        trans = aperture

        modulation = trans * torch.exp(1j * phase)
        modulation = modulation.unsqueeze(0).unsqueeze(0)
        modulation = modulation.expand(1, self.wavelengths.shape[0], M, N)

        return modulation

    def forward(self, inputfield):
        M, N = self.resolution
        assert (inputfield.shape[-1] == N and inputfield.shape[-2] == M)

        modulation = self.get_modulation()
        outputfield = inputfield * modulation

        return outputfield
