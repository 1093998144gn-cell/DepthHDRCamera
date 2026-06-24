import logging
import math
import os
import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import optim
from enum import Enum

from torch.cuda import device
from torch.utils.data import DataLoader
from tqdm import tqdm



from dataset.syn_data import SynDataset
from model.simple_model import SpectralModel
from optics.light import LightWave
from optics.optical_element import aperture, DOE, lens
from optics.propagation import Propagation
from utils.helper import *
from dataset.hyperspectral import HSDB
import yaml
from scipy.io import savemat
from utils.loss import Vgg16PerceptualLoss, SpectralAngleMapperLoss, SpectralSmoothnessLoss, PSFHybridLoss


def heightmap_to_phase(heightmap, wavelength=546.1e-9, material_n=1.556):
    return heightmap * 2 * torch.pi / wavelength * (material_n - 1)  # 将高度图转换为相位图

def wrap_phase_to_2pi(phase):
    """
    Wrap phase values to the range [0, 2π] by adding integer multiples of 2π.

    Args:
        phase (torch.Tensor): Input phase tensor (in radians).

    Returns:
        torch.Tensor: Wrapped phase tensor in [0, 2π].
    """
    return phase - 2 * math.pi * torch.floor(phase / (2 * math.pi))

def phase_to_heightmap(phase, wavelength=546.1e-9, material_n=1.556):
    return phase * wavelength / (2 * math.pi * material_n)


def sin_grating(sf, ori, phase, R): #生成二维正弦光栅图案（用于衍射测试）
    """
    :param sf: spatial frequency (in pixels)
    :param ori: wave orientation (in degrees, [0-360])
    :param phase: wave phase (in degrees, [0-360])
    :param R: resolution (integer)
    :return: torch tensor of shape (R, R)
    """
    # 将 ori 和 phase 转换为张量
    ori = torch.tensor(ori, dtype=torch.float32)
    phase = torch.tensor(phase, dtype=torch.float32)

    # Get x and y coordinates
    x, y = torch.meshgrid(torch.arange(R), torch.arange(R), indexing='xy')

    # Get the appropriate gradient
    gradient = torch.sin(ori * torch.pi / 180) * x - torch.cos(ori * torch.pi / 180) * y

    # Plug gradient into wave function
    grating = torch.sin((2 * torch.pi * gradient) / sf + (phase * torch.pi) / 180)

    return grating



def linear_phase(R, x0, y0, pixel_size, wavelength=546.1e-9, material_n=1.556, focal_length=50e-3):
    """
    生成线性相位分布（模拟光束偏转），完全使用 PyTorch 实现
    :param R: 分辨率 (像素)
    :param wavelength: 波长 (米)
    :param focal_length: 焦距 (米)
    :param x0, y0: 焦点平移距离 (米)
    :param pixel_size: 像素物理尺寸 (米)
    :return: (R, R) 的 PyTorch 张量，表示线性相位分布
    """
    # 参数验证
    if wavelength <= 0 or focal_length <= 0 or pixel_size <= 0:
        raise ValueError("wavelength, focal_length, and pixel_size must be positive")

    # 生成像素坐标（以像素为单位，中心为原点）
    coords = torch.linspace(-R / 2, R / 2, R)
    x, y = torch.meshgrid(coords, coords, indexing='xy')

    # 转换为物理坐标 (米)
    x = x * pixel_size
    y = y * pixel_size

    # 线性相位（模拟光束偏转）
    tilt_phase = (2 * torch.pi / wavelength) * (x * (x0 / focal_length) + y * (y0 / focal_length))

    # 限制相位范围到 [0, 2π]
    tilt_phase  = tilt_phase % (2 * torch.pi)
    tilt_phase = mask_outside_radius(tilt_phase)

    n_minus_1 = material_n - 1
    tilt_heightmap = tilt_phase * wavelength / (2 * math.pi * n_minus_1)

    return tilt_phase, tilt_heightmap


def generate_quadratic_lens_heightmap(N, pitch, wavelength=546.1e-9, material_n=1.556, focal_length=50e-3):
    """
    Generate a [N, N] heightmap and [1, N/2] radial heightmap for a DOE with quadratic lens phase.

    Parameters:
    - N: Heightmap size (N x N)
    - wavelength: Wavelength in nanometers
    - focal_length: Focal length in meters
    - size: Physical size of DOE in meters (square, side length)

    Returns:
    - heightmap_2d: Torch tensor of shape [N, N] (height in meters)
    - heightmap_1d: Torch tensor of shape [1, N//2] (radial height in meters)
    """
    # Convert wavelength to meters
    wavelength_m = wavelength

    x = torch.cat((torch.arange(-N // 2 + 1, 1, dtype=torch.float32) * pitch,
                   torch.arange(0, N // 2, dtype=torch.float32) * pitch))
    y = torch.cat((torch.arange(-N // 2 + 1, 1, dtype=torch.float32) * pitch,
                   torch.arange(0, N // 2, dtype=torch.float32) * pitch))

    # Generate 2D coordinate grid
    X, Y = torch.meshgrid(x, y, indexing='ij')
    R2 = X ** 2 + Y ** 2  # Squared radial distance

    # Quadratic lens phase: phi = -pi * r^2 / (lambda * f)
    phase = -math.pi * R2 / (wavelength_m * focal_length)

    # Convert phase to height: h = phi * lambda / (2 * pi * (n - 1))
    n_minus_1 = material_n - 1
    heightmap_2d = phase * wavelength_m / (2 * math.pi * n_minus_1)

    # Generate 1D radial heightmap
    r = torch.arange(0, N // 2, dtype=torch.float32) * pitch # Radial distance from 0 to D/2
    r2 = r ** 2
    phase_1d = -math.pi * r2 / (wavelength_m * focal_length)
    heightmap_1d = phase_1d * wavelength_m / (2 * math.pi * n_minus_1)
    heightmap_1d = heightmap_1d.view(1, -1)  # Shape [1, N//2]

    return heightmap_2d, heightmap_1d

class DOECamera(nn.Module):
    def __init__(self, config_path, requires_grad: bool = False):
        super(DOECamera, self).__init__()
        # 读取 YAML 配置文件
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)


        self.aperture_label = PhaseType(config['camera']['aperture_label'])
        self.f = config['camera']['f']
        self.focal_distance = config['camera']['focal_distance']  # 焦距 (m)
        self.num_depths = config['camera']['num_depths']
        self.depth_range = config['camera']['depth_range']
        self.num_wavelengths = config['camera']['num_wavelengths']
        self.wavelengths_range = config['camera']['wavelengths_range']
        self.N = config['camera']['N']
        self.crop_width = config['camera']['crop_width']
        self.image_sz = self.N + 4 * self.crop_width
        self.pitch = float(config['camera']['pitch'])
        self.phase_only = config['camera']['phase_only']
        self.modulation_type = config['camera']['ModulationType']
        self.off_axis = config['camera']['off_axis']

        self.s = 1/(1/self.f - 1/self.focal_distance)  # 传感器距离 (m)
        self.D = self.image_sz * self.pitch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.wavelengths = torch.linspace(self.wavelengths_range[0], self.wavelengths_range[1], self.num_wavelengths, device=self.device)  # 波长 (m)


        if self.num_depths > 1:
            self.depths = torch.linspace(self.depth_range[0], self.depth_range[1], self.num_depths,
                                         device=self.device)  # 深度 (m)
        else:
            self.depths = torch.tensor(self.focal_distance, device=self.device)

        #--------------------------------------------------------------------
        self.use_litho = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 使用你提供的路径
        litho_project_path = r'E:\item\CODE\Neural-Lithography-main'
        ckpt_path = r'E:\item\CODE\NewDOECamera\model\ckpt\learned_litho_model_pbl3d.pt'
        
        # 局部临时切换路径，防止污染全局环境
        orig_sys_path = sys.path.copy()
        if litho_project_path not in sys.path:
            sys.path.insert(0, litho_project_path)
        
        try:
            # 强制刷新 utils 模块指向，防止导入到错误的工具包
            if 'utils' in sys.modules:
                backup_utils = sys.modules.pop('utils')
            
            from litho_simulator.learned_litho import model_selector
            self.litho_model = model_selector('pbl3d').to(self.device)
            
            if os.path.exists(ckpt_path):
                # 加载权重
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                self.litho_model.load_state_dict(state_dict)
                print(f"✅ 神经光刻模型加载成功: {ckpt_path}")
            
            # 核心：设为评估模式，冻结参数但不切断梯度路径
            self.litho_model.eval()
            for p in self.litho_model.parameters():
                p.requires_grad = False
        finally:
            # 还原系统路径和模块状态
            sys.path = orig_sys_path
            if 'utils' in sys.modules:
                del sys.modules['utils'] # 清理光刻工程的 utils
        #---------------------------------------------------------------------
        init_h = self.init_parameters() # 调用你定义的初始化函数
        # 初始化优化参数
        self.param = torch.nn.Parameter(init_h, requires_grad=requires_grad)
        print("param", self.param.size())

        #初始化相机响应函数
        self.response_function = self.init_response_function().to(self.device)  # [3, 31]

        #初始化光学元件
        self.doe = DOE((self.image_sz, self.image_sz), (self.pitch, self.pitch), self.wavelengths)
        print("doe:", self.doe.get_modulation().size())
        self.lens = lens((self.image_sz, self.image_sz), (self.pitch, self.pitch), self.wavelengths, self.f)
        print("lens:", self.lens.get_modulation().size())

        self.aperture = aperture((self.image_sz, self.image_sz), (self.pitch, self.pitch), self.wavelengths)
        print("aperture:", self.aperture.get_modulation().size())

        #初始化传播算子
        self.propagator = Propagation(mode='D_FFT')


    def init_response_function(self) -> torch.Tensor:
        response = torch.load(r'E:\item\CODE\NewDOECamera\optics\response420660.pt')
        response = response[:, :, :, :]
        response.to(self.device)
        return response


    def init_parameters(self):
        if self.aperture_label == PhaseType.FULL_FREEDOM:
            if self.modulation_type == ModulationType.LENS.value:
                print('在这')
                parameter, _ = generate_quadratic_lens_heightmap(self.image_sz, self.pitch, focal_length=self.f)
                # parameter = torch.zeros((self.image_sz, self.image_sz), device=self.device)
            elif self.modulation_type == ModulationType.MASK.value:
                parameter = torch.zeros((self.image_sz, self.image_sz), device=self.device)
            else:
                raise ValueError("Unknown modulation type")
            return parameter
        elif self.aperture_label == PhaseType.RADIAL_SYMMETRY:
            if self.modulation_type == ModulationType.LENS.value:
                _, parameter = generate_quadratic_lens_heightmap(self.image_sz, self.pitch, focal_length=self.f)
                parameter = parameter.squeeze()
            elif self.modulation_type == ModulationType.MASK.value:
                parameter = torch.zeros(self.image_sz, device=self.device)
            else:
                raise ValueError("Unknown modulation type")
            return parameter
        else:
            raise ValueError("Invalid aperture label")



    # ======================================================================
    # Method A: physically-calibrated neural lithography (1um/px)
    # ----------------------------------------------------------------------
    # 背景：Neural-Lithography-main 的标定是 1um/px，但本工程相机/DOE 网格是 8um/px（见 pitch）。
    # 如果直接把 8um/px 的高度图 resize 成 256x256 喂给 litho 模型，会造成“物理尺度错配”，
    # 加工扩散在像素域被放大成强低通，最终 PSF 扩散变大、成像更糊。
    #
    # 方案 A 做法：
    # 1) 相机网格(8um/px) -> litho 网格(1um/px) 重采样；
    # 2) 在高分辨率上用 256x256 滑窗（带 overlap 融合）跑 litho；
    # 3) litho 网格 -> 相机网格 用 area 下采样回 8um/px。
    #
    # 注意：这里按你的标定“1um/px”默认使用 litho_px_um=1.0（不改 YAML）。
    # ======================================================================
    def _apply_neural_litho_method_a(
        self,
        ideal_h_m: torch.Tensor,
        target_n: int,
        litho_px_um: float = 1.0,
        patch: int = 256,
        overlap: int = 32,
    ) -> torch.Tensor:
        """
        Apply neural lithography with physical calibration (Method A).

        Args:
            ideal_h_m: [H,W] ideal heightmap in meters on camera grid (pitch = self.pitch).
            target_n: output size on camera grid (usually self.image_sz).
            litho_px_um: litho model calibration in um/px (your case: 1.0).
            patch: litho model patch size (trained on 256x256).
            overlap: overlap pixels between patches for seam-free blending.

        Returns:
            printed_h_m: [target_n,target_n] printed heightmap in meters on camera grid.
        """
        if ideal_h_m.dim() != 2:
            raise ValueError(f"ideal_h_m must be 2D [H,W], got {ideal_h_m.shape}")
        if litho_px_um <= 0:
            raise ValueError("litho_px_um must be > 0")
        if patch <= 0:
            raise ValueError("patch must be > 0")
        if overlap < 0 or overlap >= patch:
            raise ValueError("overlap must be in [0, patch)")

        # 0) Ensure camera-grid size first (avoid accumulating interpolation errors later)
        if ideal_h_m.shape[-2] != target_n or ideal_h_m.shape[-1] != target_n:
            ideal_h_m = F.interpolate(
                ideal_h_m.unsqueeze(0).unsqueeze(0),
                size=(target_n, target_n),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        # 1) Camera grid (um/px) -> Litho grid (um/px)
        pitch_um = float(self.pitch) * 1e6  # camera pixel pitch in um/px (here: 8.0)
        scale = pitch_um / float(litho_px_um)  # e.g. 8um/px -> 1um/px => scale=8
        hi_h = max(1, int(round(target_n * scale)))
        hi_w = max(1, int(round(target_n * scale)))

        # Bicubic upsample keeps surfaces smoother than bilinear; clamp keeps non-negative heights.
        x_hi_um = ideal_h_m.unsqueeze(0).unsqueeze(0) * 1e6  # [1,1,H,W] in um
        if (hi_h, hi_w) != (target_n, target_n):
            x_hi_um = F.interpolate(x_hi_um, size=(hi_h, hi_w), mode="bicubic", align_corners=False)
        x_hi_um = torch.clamp(x_hi_um, min=0.0)

        # 2) Sliding-window inference on high-res grid (keep gradient path; do NOT wrap in no_grad)
        def _blend_window_2d(patch_sz: int, ov: int, device, dtype) -> torch.Tensor:
            if ov <= 0:
                return torch.ones((patch_sz, patch_sz), device=device, dtype=dtype)
            # Linear ramps on borders to reduce seams; never exactly zero to avoid boundary division issues.
            w = torch.ones((patch_sz,), device=device, dtype=dtype)
            ramp = torch.linspace(0.0, 1.0, steps=ov + 2, device=device, dtype=dtype)[1:-1]  # (0,1)
            w[:ov] = ramp
            w[-ov:] = torch.flip(ramp, dims=(0,))
            return (w[:, None] * w[None, :]).clamp_min(1e-3)

        def _run_sliding_window(model, x: torch.Tensor) -> torch.Tensor:
            # x: [1,1,H,W] in um
            _, _, H, W = x.shape
            step = patch - overlap
            if step <= 0:
                raise ValueError("patch - overlap must be > 0")

            num_steps_h = 1 if H <= patch else (int(math.ceil((H - patch) / step)) + 1)
            num_steps_w = 1 if W <= patch else (int(math.ceil((W - patch) / step)) + 1)

            H_pad = (num_steps_h - 1) * step + patch
            W_pad = (num_steps_w - 1) * step + patch
            pad_h = max(0, H_pad - H)
            pad_w = max(0, W_pad - W)

            if pad_h > 0 or pad_w > 0:
                # reflect padding has constraints: pad < input_size; otherwise fallback to replicate
                can_reflect = (H > 1 and W > 1 and pad_h < H and pad_w < W)
                pad_mode = "reflect" if can_reflect else "replicate"
                x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
            else:
                x_pad = x

            window = _blend_window_2d(patch, overlap, device=x.device, dtype=x.dtype)[None, None, ...]  # [1,1,P,P]
            out_sum = torch.zeros_like(x_pad)
            w_sum = torch.zeros_like(x_pad)

            for iy in range(num_steps_h):
                y0 = iy * step
                y1 = y0 + patch
                for ix in range(num_steps_w):
                    x0 = ix * step
                    x1 = x0 + patch
                    tile = x_pad[:, :, y0:y1, x0:x1]
                    pred = model(tile)
                    if pred.shape != tile.shape:
                        raise ValueError(f"litho model output shape {pred.shape} != input tile shape {tile.shape}")
                    out_sum[:, :, y0:y1, x0:x1] += pred * window
                    w_sum[:, :, y0:y1, x0:x1] += window

            out = out_sum / torch.clamp(w_sum, min=1e-12)
            return out[:, :, :H, :W]

        printed_hi_um = _run_sliding_window(self.litho_model, x_hi_um)

        # 3) Litho grid -> Camera grid
        if printed_hi_um.shape[-2] != target_n or printed_hi_um.shape[-1] != target_n:
            # Downsample uses area to preserve average height; upsample uses bicubic.
            if printed_hi_um.shape[-2] > target_n or printed_hi_um.shape[-1] > target_n:
                printed_cam_um = F.interpolate(printed_hi_um, size=(target_n, target_n), mode="area")
            else:
                printed_cam_um = F.interpolate(printed_hi_um, size=(target_n, target_n), mode="bicubic", align_corners=False)
        else:
            printed_cam_um = printed_hi_um

        printed_cam_um = torch.clamp(printed_cam_um, min=0.0)
        return printed_cam_um.squeeze(0).squeeze(0) / 1e6


    def get_heightmap(self):
        """
        返回用于 DOE 的 2D 高度图（单位：米），尺寸必须为 [self.image_sz, self.image_sz]。

        关键点：
        - `aperture_label=radial_symmetry` 时，`self.param` 是 1D 半径剖面（长度 self.image_sz/2），必须先经
          `radial_symmetry()` 展成 2D。
        - “设计高度图”应做 2π 相位包裹，使高度落在约 `λ/(n-1) ~ 1um` 的物理范围内，避免出现几十微米的负厚度。
        - `use_litho=True` 时，用神经光刻模型对“包裹后的设计高度图”做加工响应预测，并再插值回相机分辨率。
        """
        target_n = int(self.image_sz)

        # 0) 先把参数统一成 2D 设计高度图
        if self.aperture_label == PhaseType.RADIAL_SYMMETRY:
            # self.param: [N] where 2*N == image_sz
            ideal_h = radial_symmetry(self.param.unsqueeze(0).to(self.device)).squeeze(0)
        else:
            ideal_h = self.param.to(self.device)
            if ideal_h.dim() == 1:
                # 容错：若出现 1D 参数，至少保证输出是 2D
                ideal_h = ideal_h.view(1, -1)
                ideal_h = radial_symmetry(ideal_h).squeeze(0)

        # 1) 做 2π 相位包裹：高度范围约为 [0, λ/(n-1))
        # 使用设计波长的中值（若有多波长），避免硬编码。
        lambda_d = float(self.wavelengths.detach().mean().item()) if hasattr(self, "wavelengths") else 550e-9
        ideal_h = transform_heightmap_to_positive(ideal_h, lambda_val=lambda_d, n=1.556)

        if hasattr(self, 'use_litho') and self.use_litho:
            # 若光刻模型未成功加载，退化为理想高度图
            if not hasattr(self, "litho_model") or self.litho_model is None:
                return ideal_h

            # === 方案 A（物理标定一致）：8um/px -> 1um/px（滑窗）-> 8um/px ===
            # litho 标定：1um/px（按你的确认）
            return self._apply_neural_litho_method_a(
                ideal_h_m=ideal_h,
                target_n=target_n,
                litho_px_um=1.0,
                patch=256,
                overlap=32,
            )

        # 非光刻模式：也必须确保输出尺寸为 target_n
        if ideal_h.dim() == 2 and (ideal_h.shape[-2] != target_n or ideal_h.shape[-1] != target_n):
            ideal_h = F.interpolate(ideal_h.unsqueeze(0).unsqueeze(0), size=(target_n, target_n), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        return ideal_h
    # #原始get_heightmap
    # def get_heightmap(self):
    #     if self.aperture_label == PhaseType.FULL_FREEDOM:
    #         heightmap = self.param.to(self.device)
    #         heightmap = transform_heightmap_to_positive(heightmap)
    #         return heightmap
    #     elif self.aperture_label == PhaseType.RADIAL_SYMMETRY:
    #         heightmap = radial_symmetry(self.param.unsqueeze(0).to(self.device))
    #         heightmap = heightmap.squeeze()
    #         heightmap = transform_heightmap_to_positive(heightmap)
    #         return heightmap
    #     else:
    #         raise ValueError("Invalid aperture label")


    def compute_psf(self):
        # 初始化光波场
        light = LightWave((self.image_sz, self.image_sz), (self.pitch, self.pitch), self.wavelengths)
        light.init_wave(self.depths, wave_type='spherical')

        #设置DOE参数
        heightmap = self.get_heightmap()

        self.doe.set_heightmap(heightmap)

        doe_field = self.doe.forward(light.get_wave())
        # lens_field = self.lens.forward(doe_field)
        aperture_field = self.aperture.forward(doe_field)

        light.set_complex(aperture_field)
        sensor_field = self.propagator.forward(light, self.s)

        # 计算 PSF（光强）
        PSF = torch.abs(sensor_field) ** 2  # [num_depths, num_wavelengths, N, N]
        PSF = PSF / torch.sum(PSF, dim=(-2, -1), keepdim=True)  # 归一化
        return PSF  # 返回实数值 PSF

    def simulate_imaging(self, hyperspectral: torch.Tensor, depthmap=None):
        """模拟成像过程，使用复数运算"""
        psf = self.compute_psf()

        batch_size = hyperspectral.shape[0]

        if depthmap is None:
            if self.num_depths != 1:
                raise ValueError("Depth map must be provided if num_depths > 1")
            else:
                depthmap = torch.ones(batch_size, 1, self.image_sz, self.image_sz,
                                      device=self.device) * self.focal_distance
        else:
            depthmap = stretch_depthmap(depthmap, depth_min=self.depth_range[0], depth_max=self.depth_range[1])

        depth_step = (self.depth_range[1] - self.depth_range[0]) / ((self.num_depths - 1) if self.num_depths > 1 else 1)
        depth_bins = self.depths.view(1, self.num_depths, 1, 1)
        # 计算二值化占用图（原始掩码）
        mask = torch.logical_and(
            depthmap >= (depth_bins - depth_step / 2),
            depthmap < (depth_bins + depth_step / 2)
        ).float()
        # 添加高斯滤波平滑掩码
        # 将 mask 的形状从 [batch_size, num_depths, N, N] 处理为适合卷积的格式
        mask = mask.view(batch_size * self.num_depths, 1, self.image_sz, self.image_sz)  # [B*D, 1, H, W]
        mask = gaussian_blur(mask, kernel_size=3, sigma=2.0, device=self.device)  # 应用高斯滤波
        mask = mask.view(batch_size, self.num_depths, self.image_sz, self.image_sz)  # 恢复形状
        # ================== 关键补丁位置 ==================
        # 如果 depthmap 的值全都不在 depth_range 内，mask 会全为 0，导致卷积结果全黑。
        # 这里强制检查：如果全为 0，则让所有深度层平均分担（或设为 1），保证至少有图像输出。
        if mask.max() == 0:
            print("WARNING: Depthmap values out of range! Mask is all zeros. Patching to ones.")
            mask = torch.ones_like(mask) / self.num_depths 
        # =================================================
        hyperspectral_exp = hyperspectral.unsqueeze(1) * mask.unsqueeze(2)

        fft_psf = torch.fft.fft2(psf, dim=(-2, -1)).unsqueeze(0)
        fft_hyperspectral = torch.fft.fft2(hyperspectral_exp, dim=(-2, -1))

        fft_product = fft_psf * fft_hyperspectral

        conv_result = torch.fft.ifft2(fft_product, dim=(-2, -1))

        conv_result = torch.real(conv_result)

        conv_result = torch.fft.fftshift(conv_result, dim=(-2, -1))

        hyperspectral_img = torch.sum(conv_result, dim=1)

        if self.num_wavelengths > 3:
            sensor_img = hyperspectral_to_rgb(self.response_function, hyperspectral_img, self.num_wavelengths)
            sensor_img = normalize_rgb(sensor_img)
        else:
            sensor_img = hyperspectral_img

        sensor_img = normalize_rgb(sensor_img)

        return sensor_img, psf


    def visualize_psf(self, pupil_function, PSF: torch.Tensor = None):
        """可视化所有深度和波长的PSF"""
        if PSF is None:
            PSF = self.compute_psf(pupil_function)

        num_depths, num_wavelengths = self.depths.shape[0], self.wavelengths.shape[0]
        fig, axes = plt.subplots(num_depths, num_wavelengths, figsize=(num_wavelengths * 4, num_depths * 4))
        x_max_theory_val = self.x_max_theory.item()

        if num_depths == 1:
            axes = axes[np.newaxis, :]
        elif num_wavelengths == 1:
            axes = axes[:, np.newaxis]

        for d in range(num_depths):
            for w in range(num_wavelengths):
                psf_channel = PSF[d, w].cpu().numpy()
                ax = axes[d, w]
                ax.imshow(psf_channel,
                          extent=[-x_max_theory_val, x_max_theory_val, -x_max_theory_val, x_max_theory_val], cmap='hot')
                ax.set_title(f'Depth {self.depths[d].item():.1f}m, λ={self.wavelengths[w].item() * 1e9:.0f}nm')
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
                plt.colorbar(ax.images[0], ax=ax, label='Normalized Intensity')

        plt.tight_layout()
        plt.show()

    def print_scales(self):
        """打印尺度信息"""
        print(f"Theoretical Δx (middle wavelength): {self.Delta_x_theory.item():.2e} m")
        print(f"Sensor Δx: {self.sensor_pixel_size:.2e} m")
        print("sensor distance: ", self.s)
        print("D:", self.D)
        print("response function:", self.response_function.size())

    def forward(self, img, depthmap=None):
         # 动态计算
        sensor_img, psf = self.simulate_imaging(img, depthmap)
        return sensor_img, psf

    def train_doe(self, num_epochs=2000, lr=1e-5, psf_size=70, save_dir="./checkpoints"):
        """
        训练函数，用于优化 DOE 高度图以最小化 PSF 损失，显示精度并保存最优模型。

        Args:
            target_psf (torch.Tensor): 目标 PSF，形状与 compute_psf 输出一致。
            num_epochs (int): 训练轮数。
            lr (float): 初始学习率。
            save_dir (str): 模型检查点和日志保存路径。

        Returns:
            None
        """
        # 设置日志
        os.makedirs(save_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(save_dir, "training.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logger = logging.getLogger()

        # 确保 heightmap1d_ 可训练
        if not self.param.requires_grad:
            self.param.requires_grad_(True)

        # 初始化优化器和调度器
        optimizer = optim.Adam([self.param], lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        # 跟踪最优模型
        best_loss = float('inf')
        best_checkpoint_path = os.path.join(save_dir, "doe_best_checkpoint.pth")

        # 训练循环
        self.train()  # 设置模型为训练模式
        for epoch in range(num_epochs):
            # 清零梯度
            optimizer.zero_grad()

            # 计算 PSF
            psf = self.compute_psf()

            # 计算 PSF 损失
            # psf_lossfn1 = PSFLoss(psf_size)  # 假设 PSFloss 接受 psf_shift 和目标 PSF
            #
            # psf_lossfn2 = PSFLoss(psf_size/2)

            psf_lossfn = PSFHybridLoss(psf.shape[-1])

            # loss = psf_lossfn1(psf) + PSFConsistencyloss(psf)
            # loss = PSFConsistencyloss(psf)
            loss = psf_lossfn(psf)

            # 反向传播
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_([self.param], max_norm=1.0)

            # 更新参数
            optimizer.step()

            # 确保高度图正值
            # with torch.no_grad():
            #     self.heightmap1d_.data = torch.clamp(self.heightmap1d_.data, min=0)

            # 打印和记录损失
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")

            # 更新学习率
            scheduler.step()

            # 保存最优模型
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.state_dict(),  # 保存整个模型的 state_dict
                    'optimizer': optimizer.state_dict(),
                    'loss': loss.item()
                }, best_checkpoint_path)
                logger.info(f"Saved best checkpoint (Loss: {best_loss:.6f}) at: {best_checkpoint_path}")

        logger.info(f"Training completed. Best Loss: {best_loss:.6f}, Best Checkpoint: {best_checkpoint_path}")
        print(f"Training completed. Best Loss: {best_loss:.6f}, Best Checkpoint: {best_checkpoint_path}")


def DoeTrainer():
    config_path = 'E:\PycharmProject/NewDOECamera\optics\DOECamera.yaml'
    camera = DOECamera(config_path, True)
    camera = camera.to('cuda')
    camera.train_doe()


def load_and_visualize_psf(checkpoint_path):
    """
        读取训练好的检查点文件，计算 PSF 并显示每个通道的 PSF 图像。

        Args:
            camera (DOECamera): 已初始化的 DOECamera 实例。
            checkpoint_path (str): 检查点文件路径（如 "doe_best_checkpoint.pth"）。
            save_dir (str): 保存 PSF 图像的目录（如果 save_image=True）。
            save_image (bool): 是否保存 PSF 图像到文件。

        Returns:
            tuple: (psf, psf_shift) - 计算得到的 PSF 张量，形状 [b, c, h, w]。
        """
    config_path = 'E:\PycharmProject/NewDOECamera\optics\DOECamera.yaml'
    camera = DOECamera(config_path, True)
    camera = camera.to('cuda')

    # 确保检查点文件存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=camera.device)
    camera.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, Loss: {checkpoint['loss']:.6f}")

    # 设置模型为评估模式
    camera.eval()

    # 计算 PSF
    with torch.no_grad():
        psf = camera.compute_psf()

    # 确保 PSF 形状正确
    assert psf.dim() == 4, f"Expected psf_shift to have 4 dimensions [b, c, h, w], got {psf.shape}"
    b, c, h, w = psf.shape
    assert b == 1, f"Expected batch size 1, got {b}"

    # 获取波长信息（如果可用）
    wavelengths = camera.wavelengths.cpu().numpy() if hasattr(camera, 'wavelengths') else None
    if wavelengths is None or len(wavelengths) != c:
        wavelengths = [f"Channel {i}" for i in range(c)]

    # 绘制 PSF 图像
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    axes = axes.flatten()
    for i in range(25):
        axes[i].imshow(psf[0, i].detach().cpu().numpy(), cmap='gray')
        axes[i].set_title(f'psf {wavelengths[i]} nm')
        axes[i].axis('off')  # Hide axes for cleaner visualization
    plt.tight_layout()
    plt.show()

    # savemat('psf.mat', {'psf': psf.detach().cpu().numpy()})  # 保存 PSF 数据到 .mat 文件

    return psf


def spectral_psf():
    config_path = 'E:\PycharmProject/NewDOECamera\optics\DOECamera.yaml'
    camera = DOECamera(config_path=config_path, requires_grad=True)
    camera = camera.to('cuda')
    psf = camera.compute_psf()

    savemat('psf.mat', {'psf': psf.detach().cpu().numpy()})

    wavelengths = camera.wavelengths

    psf = psf.detach().cpu().numpy()
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Plot each of the 25 images
    for i in range(25):
        axes[i].imshow(psf[0, i], cmap='viridis')
        axes[i].set_title(f'psf {wavelengths[i]} nm')
        axes[i].axis('off')  # Hide axes for cleaner visualization

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

    # savemat('psf1024.mat', {'psf1024': psf})


def test_imaging():
    camera = DOECamera(True)
    camera = camera.to('cuda')
    dataset = HSDB((camera.image_sz, camera.image_sz))
    sample = dataset.__getitem__(0)
    image = sample['image']

    image = image.unsqueeze(0)
    image = image.to('cuda')

    sensor_img, psf = camera.simulate_imaging(image)

    print('sensor_img', sensor_img.size())

    sensor_img_np = sensor_img[0].detach().cpu().numpy()  # [3, 640, 640]

    # 1. 显示 sensor_img（RGB图像）
    plt.figure(figsize=(8, 8))
    plt.imshow(sensor_img_np.transpose(1, 2, 0))
    plt.title("Sensor Image (RGB)")
    plt.axis('off')
    plt.show()

    target_rgb = hyperspectral_to_rgb(camera.response_function, image, camera.num_wavelengths)
    target_rgb = normalize_rgb(target_rgb)
    plt.figure(figsize=(8, 8))
    plt.imshow(target_rgb[0].detach().cpu().numpy().transpose(1, 2, 0))
    plt.title("Target RGB")
    plt.axis('off')
    plt.show()



    selected_psf = psf[0, :, :, :]  # 形状变为 (10, 3, 384, 384)
    # 提取中间 128x128 并归一化
    cropped_psf = selected_psf
    normalized_psf = cropped_psf.detach().cpu().numpy()

    savemat('psf.mat', {'psf': normalized_psf})

    # 设置画布，显示 30 张图片（5 行 × 6 列）
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 12))  # 调整画布大小，每行 6 张

    # 遍历并显示图片
    for i in range(25):
        channel = i  # 通道索引
        row = i // 5  # 计算行号
        col = i % 5  # 计算列号
        ax = axes[row, col]

        # 显示图片
        ax.imshow(normalized_psf[channel], cmap='gray')

        # 添加十字架（横线和竖线）
        # ax.axhline(y=64, color='red', linewidth=1)  # 横线，128/2=64
        # ax.axvline(x=64, color='red', linewidth=1)  # 竖线，128/2=64

        # 设置标题和隐藏坐标轴
        ax.set_title(f'{i} nm', fontsize=8)
        ax.axis('off')  # 隐藏坐标轴

    # 调整布局并显示
    plt.tight_layout()
    plt.show()


    heightmap = camera.get_heightmap()
    plt.figure(figsize=(8, 8))
    plt.imshow(heightmap.detach().cpu().numpy())
    plt.title("Heightmap")
    plt.colorbar()
    plt.show()

    print('heightmap_max', heightmap.max())




def test_DOE():
    wavelengths = torch.linspace(4.20e-7, 6.60e-7, 25)
    doe = DOE((1024, 1024), (1e-6, 1e-6), wavelengths)

    tilt_phase, tilt_heightmap = linear_phase(1024, 1.3e-3, 0, 1e-6, focal_length=30e-3)

    doe.set_heightmap(tilt_heightmap)

    phase = doe.heightmap_to_phase(doe.heightmap, doe.wavelengths, doe.refractive_index(doe.wavelengths))
    print('phase:', phase.shape)
    savemat('phase.mat', {'phase': phase})





if __name__ == "__main__":

    spectral_psf()

    # DoeTrainer()
    # load_and_visualize_psf('E:\PycharmProject/NewDOECamera\optics\checkpoints\doe_best_checkpoint.pth')


