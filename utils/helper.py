import math
from enum import Enum

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.io import savemat
from torchvision.utils import make_grid
import torch.nn.functional as F


class PhaseType(Enum):
    RADIAL_SYMMETRY = "radial_symmetry"
    RADIAL_SPIRAL = "radial_symmetry_spiral"
    FULL_FREEDOM = "full_freedom"


class ModulationType(Enum):
    LENS = "lens"
    MASK = "mask"


def crop_boundary(x, w):
    if w == 0:
        return x
    else:
        return x[..., w:-w, w:-w]

def pad_boundary(x, w):
    if w == 0:
        return x
    else:
        return F.pad(x, (w, w, w, w))


def hyperspectral_to_rgb(response_function, hyperspectral_img, num_wavelengths):
    if response_function.shape[1] != hyperspectral_img.shape[1]:
        print("response_function shape: ", response_function.shape)
        print("hyperspectral_img shape: ", hyperspectral_img.shape)
        raise ValueError("response_function and hyperspectral_img must have the same number of channels.")
    response = response_function.view(1, 3, num_wavelengths, 1, 1)
    hyperspectral_img = hyperspectral_img.unsqueeze(1)
    sensor_img = torch.sum(hyperspectral_img * response, dim=2)

    sensor_img = sensor_img[:, [2, 1, 0], :, :]

    return sensor_img



# 颜色映射函数：将波长转换为RGB
def wavelength_to_rgb(wavelength, gamma=0.8):
    """
    Convert wavelength in nanometers to RGB color.
    Args:
        wavelength (float): Wavelength in meters (e.g., 420e-9 for 420 nm).
        gamma (float): Gamma correction factor. Default is 0.8.
    Returns:
        tuple: RGB color as integers (R, G, B).
    """
    intensity_max = 255
    R = G = B = 0

    if 380e-9 <= wavelength <= 440e-9:
        R = -(wavelength - 440e-9) / (440e-9 - 380e-9)
        G = 0.0
        B = 1.0
    elif 440e-9 < wavelength <= 490e-9:
        R = 0.0
        G = (wavelength - 440e-9) / (490e-9 - 440e-9)
        B = 1.0
    elif 490e-9 < wavelength <= 510e-9:
        R = 0.0
        G = 1.0
        B = -(wavelength - 510e-9) / (510e-9 - 490e-9)
    elif 510e-9 < wavelength <= 580e-9:
        R = (wavelength - 510e-9) / (580e-9 - 510e-9)
        G = 1.0
        B = 0.0
    elif 580e-9 < wavelength <= 645e-9:
        R = 1.0
        G = -(wavelength - 645e-9) / (645e-9 - 580e-9)
        B = 0.0
    elif 645e-9 < wavelength <= 780e-9:
        R = 1.0
        G = 0.0
        B = 0.0

    R = int((R ** gamma) * intensity_max)
    G = int((G ** gamma) * intensity_max)
    B = int((B ** gamma) * intensity_max)
    return (R, G, B)


# 将高光谱图像转换为RGB图像
def save_wavelength_rgb_tensor(spectral_data, wavelengths = None):
    """
    将高光谱图像的每个通道转换为RGB图像，每个通道根据波长映射到相应的RGB颜色。

    Args:
        spectral_image (torch.Tensor): 高光谱图像，形状为 [C, H, W]。
        wavelengths (list): 每个通道对应的波长值（单位：米）。

    Returns:
        torch.Tensor: 转换后的RGB图像，形状为 [C, H, W, 3]。
    """
    if wavelengths == None:
        wavelengths = list(np.arange(420, 721, 10) * 1e-9)

    rgb_images = []
    for i, wavelength in enumerate(wavelengths):
        rgb_color = torch.tensor(wavelength_to_rgb(wavelength), dtype=torch.uint8)
        height, width = spectral_data[i].shape
        rgb_image = torch.zeros((height, width, 3), dtype=torch.uint8)
        for j in range(3):
            rgb_image[..., j] = (spectral_data[i] * rgb_color[j]).to(torch.uint8)
        rgb_images.append(rgb_image)

    # 将所有通道的RGB图像合并为一个张量
    rgb_images_tensor = torch.stack(rgb_images, dim=0)
    return rgb_images_tensor


# 可视化RGB图像
def visualize_rgb_images(spectrals_rgb, path = None):
    """
    可视化每个波长通道对应的RGB图像。

    Args:
        spectral_data (torch.Tensor): RGB图像张量，形状为 [C, H, W, 3]。
        wavelengths (list): 每个通道对应的波长值（单位：米）。
    """
    # 1. 调整维度顺序，从 [31, 256, 256, 3] 到 [31, 3, 256, 256]
    spectrals_rgb = spectrals_rgb.permute(0, 3, 1, 2)  # [N, C, H, W]

    spectrals_rgb = (spectrals_rgb - spectrals_rgb.min()) / (
            spectrals_rgb.max() - spectrals_rgb.min() + 1e-8)

    # 2. 使用 make_grid 将 31 张图像排列成网格
    # nrow 参数控制每行显示的图像数量，例如 6 表示每行 6 张图
    grid = make_grid(spectrals_rgb, nrow=6, padding=2, normalize=True)

    # 3. 将网格张量转换为 numpy 数组，用于 matplotlib 显示
    grid_np = grid.permute(1, 2, 0).detach().cpu().numpy()  # 从 [C, H, W] 转换为 [H, W, C]

    # 4. 使用 matplotlib 可视化
    plt.figure(figsize=(15, 10))  # 设置画布大小
    plt.imshow(grid_np)
    plt.axis('off')  # 关闭坐标轴
    plt.title("31 RGB Images in Grid")
    if path:
        plt.savefig(path , dpi=300, bbox_inches='tight')  # 保存图像
    plt.show()


def differentiable_interp(x, xp, fp):
    """
    可微的线性插值函数。

    参数:
        x (torch.Tensor): 需要插值的点，任意形状
        xp (torch.Tensor): 原始索引，形状 [N]
        fp (torch.Tensor): 原始值，形状 [num, N]

    返回:
        torch.Tensor: 插值结果，形状 [num, ...]（与 x 的形状匹配）
    """
    max_radius = xp[-1]  # 最大索引
    x = x.clamp(min=0, max=max_radius)  # 限制范围，可微
    idx = (x / max_radius) * (len(xp) - 1)  # 映射到 0-(N-1)
    idx_floor = idx.floor().long()  # 下界索引
    idx_ceil = (idx_floor + 1).clamp(max=len(xp) - 1)  # 上界索引
    weight = idx - idx_floor.float()  # 插值权重
    y0 = fp[:, idx_floor]  # [num, ...]
    y1 = fp[:, idx_ceil]  # [num, ...]
    return y0 + (y1 - y0) * weight  # [num, ...]


def radial_symmetry(data):
    """
    将 [num, N] 的半径数据转换为 [num, 2*N, 2*N] 的径向对称数据，整个过程可微。
    输入数据对应半径 [0, N-1]，输出图像在半径 R <= N-1 范围内插值，超出部分设为 0。
    使用 1/4 扇形插值 + 翻转补全。

    参数:
        data (torch.Tensor): 输入张量，形状 [num, N]，对应半径 [0, N-1]

    返回:
        torch.Tensor: 输出张量，形状 [num, 2*N, 2*N]，径向对称数据
    """
    # 获取输入维度
    num, N = data.shape

    # 确保 data 在正确设备上并支持梯度
    device = data.device
    data = data.requires_grad_(True) if not data.requires_grad else data

    # 创建第一象限网格（x, y >= 0）
    x_1q = torch.arange(0, N, dtype=torch.float32, device=device)  # [0, 1, ..., N-1]
    y_1q = x_1q.clone()
    X_1q, Y_1q = torch.meshgrid(x_1q, y_1q, indexing='ij')  # [N, N]

    # 计算第一象限径向距离
    R_1q = torch.sqrt(X_1q ** 2 + Y_1q ** 2)  # [N, N]

    # 最大半径
    max_radius = float(N - 1)

    # 参考点（相对半径）
    xp = torch.linspace(0, max_radius, N, device=device)  # [0, N-1]

    # 插值（仅在 R <= max_radius 范围内）
    R_1q_flat = R_1q.flatten()  # [N*N]
    result_1q_flat = differentiable_interp(R_1q_flat, xp, data)  # [num, N*N]

    # 显式处理 R > N-1
    mask = (R_1q_flat <= max_radius).float()  # [N*N]，R <= N-1 时为 1，否则为 0
    result_1q_flat = result_1q_flat * mask[None, :]  # 广播掩码，[num, N*N]

    result_1q = result_1q_flat.reshape(num, N, N)  # [num, N, N]

    result_2q = torch.flip(result_1q, dims=(-2,))

    result_d = torch.cat([result_2q, result_1q], dim=-2)

    result_u = torch.flip(result_d, dims=(-1,))

    output = torch.cat([result_u, result_d], dim=-1)

    return output


def generate_fresnel_zone_phase(N, L, epsilon, device='cuda'):
    """
    生成菲涅尔区相位分布，并归一化到 [-π, π]，圆外区域填充 0。

    :param N: 图像大小 (NxN)
    :param L: 环数
    :param epsilon: 控制环宽度的参数
    :param device: 计算设备 (cpu 或 cuda)
    :return: 生成的相位分布张量
    """
    x = torch.linspace(-1, 1, N, device=device)
    y = torch.linspace(-1, 1, N, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    r = torch.sqrt(X ** 2 + Y ** 2)  # 计算归一化半径
    phi = torch.atan2(Y, X)  # 计算方位角

    phase = torch.zeros_like(r, device=device)  # 初始化相位为 0
    r_powers = torch.linspace(0, 1, L + 1, device=device) ** epsilon  # 预计算 r_min, r_max

    for l in range(1, L + 1):
        mask = (r >= r_powers[l - 1]) & (r < r_powers[l])
        phase = torch.where(mask, l * phi, phase)

    # 归一化相位到 [-π, π]，不影响相位的整体定义
    phase = torch.remainder(phase + torch.pi, 2 * torch.pi) - torch.pi

    # 圆外区域填充为 0
    phase[r > 1] = 0

    return phase


def display_phase(phase, path = None):
    """
    显示菲涅尔区相位分布。
    :param phase: 相位分布张量
    """
    # 将 PyTorch 张量转换为 NumPy 数组以使用 matplotlib
    phase_np = phase.detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    cmap = plt.cm.twilight  # 使用淡雅配色
    im = plt.imshow(phase_np, cmap=cmap, extent=(-1, 1, -1, 1))
    plt.colorbar(im, label='Phase (radians)', fraction=0.046, pad=0.04)
    plt.title("Phase")
    plt.xlabel("x")
    plt.ylabel("y")
    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_1d_random_phase(N, device='cpu'):
    """
    生成一维相位数据，范围 0 到 2pi 之间的随机数。
    :param N: 数据点数
    :param device: 计算设备 (cpu 或 cuda)
    :return: 生成的相位张量
    """
    return torch.rand(N, device=device) * 2 * torch.pi


def gaussian_kernel(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """生成二维高斯核"""
    # 确保 kernel_size 是奇数
    assert kernel_size % 2 == 1, "Kernel size must be odd"

    # 创建一维高斯核
    ax = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32, device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()  # 归一化
    return kernel.view(1, 1, kernel_size, kernel_size)


def gaussian_blur(input: torch.Tensor, kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """手动实现高斯模糊"""
    # 输入形状应为 [B, C, H, W]
    batch_size, channels, height, width = input.shape

    # 生成高斯核
    kernel = gaussian_kernel(kernel_size, sigma, device)

    # 对每个通道应用卷积
    padded_input = F.pad(input, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2),
                         mode='reflect')
    blurred = F.conv2d(padded_input, kernel.expand(channels, 1, kernel_size, kernel_size), groups=channels)
    return blurred

def normalize_rgb(image):
    normalize_image = (image - image.min()) / (image.max() - image.min() + 1e-6)
    return normalize_image


def stretch_depthmap(depth_map, depth_min, depth_max):
    """
    将 [0, 1] 范围的深度图拉伸到 [depth_min, depth_max]，全程使用 torch.Tensor。

    Args:
        depth_map: 输入深度图，范围 [0, 1]（torch.Tensor）
        depth_min: 目标范围的最小值（float 或 torch.Tensor）
        depth_max: 目标范围的最大值（float 或 torch.Tensor）

    Returns:
        拉伸后的深度图（torch.Tensor）
    """
    # 确保输入是 torch.Tensor
    if not isinstance(depth_map, torch.Tensor):
        raise TypeError("depth_map must be a torch.Tensor")

    # 转换为浮点型张量
    depth_map = depth_map.float()

    # 将 depth_min 和 depth_max 转换为张量并确保与 depth_map 在同一设备上
    depth_min = torch.tensor(depth_min, dtype=torch.float32, device=depth_map.device)
    depth_max = torch.tensor(depth_max, dtype=torch.float32, device=depth_map.device)


    # 线性拉伸到 [depth_min, depth_max]
    stretched_depth = depth_min + (depth_max - depth_min) * depth_map

    return stretched_depth


def torch_floormod(x, y):
    return x - y * torch.floor(torch.div(x, y))


def transform_heightmap_to_positive(data, lambda_val=546.1e-9, n=1.556):
    """
    Transform heightmap to ensure all values are positive while preserving differentiability.

    Args:
        data (torch.Tensor): Input heightmap tensor, requires_grad=True.
        lambda_val (float): Wavelength, default 546.1e-9.
        n (float): Refractive index, default 1.556.

    Returns:
        torch.Tensor: Transformed heightmap with all positive values.
    """
    coeff = lambda_val / (n - 1)

    # Create a boolean mask for negative values
    mask = data < 0

    # Initialize m with zeros, same shape as data
    m = torch.zeros_like(data, dtype=torch.int)

    # Compute m_float only for negative values
    if mask.any():  # Only compute if there are negative values
        m_float = -data[mask] * (n - 1) / lambda_val
        m_values = torch.ceil(m_float).to(torch.int)

        # Check temp_result for negative values
        temp_result = data[mask] + coeff * m_values.float()
        m_values = torch.where(temp_result <= 0, m_values + 1, m_values)

        # Assign m_values to corresponding positions in m
        m[mask] = m_values

    # Update data using non-in-place operation
    data = torch.where(mask, data + coeff * m.float(), data)

    phase = heightmap_to_phase(data)
    phase = wrap_phase_to_2pi(phase)

    data = phase_to_heightmap(phase)

    return data




def phase_to_heightmap(phase, lamda = 546.1e-9, material_n=1.556):
    """
    Convert phase profiles psi_xf and psi_yf to 2D heightmaps for a DOE.

    Parameters:
    - psi_xf: 2D torch tensor, phase profile for x-polarized light in radians
    - psi_yf: 2D torch tensor, phase profile for y-polarized light in radians
    - lamda: float, wavelength in nm
    - material_n: float, refractive index of the material (default: 1.556)

    Returns:
    - heightmap_x: 2D torch tensor, heightmap for x-polarized light in mm
    - heightmap_y: 2D torch tensor, heightmap for y-polarized light in mm
    """

    # Compute refractive index difference
    n_minus_1 = material_n - 1

    # Convert phase to height: h = phi * lambda / (2 * pi * (n - 1))
    heightmap = phase * lamda / (2 * math.pi * n_minus_1)

    return heightmap

def heightmap_to_phase(heightmap, lamda = 546.1e-9, material_n=1.556):
    """
    Convert phase profiles psi_xf and psi_yf to 2D heightmaps for a DOE.

    Parameters:
    - psi_xf: 2D torch tensor, phase profile for x-polarized light in radians
    - psi_yf: 2D torch tensor, phase profile for y-polarized light in radians
    - lamda: float, wavelength in nm
    - material_n: float, refractive index of the material (default: 1.556)

    Returns:
    - heightmap_x: 2D torch tensor, heightmap for x-polarized light in mm
    - heightmap_y: 2D torch tensor, heightmap for y-polarized light in mm
    """

    # Compute refractive index difference
    n_minus_1 = material_n - 1

    # Convert phase to height: h = phi * lambda / (2 * pi * (n - 1))
    phase = heightmap * (2 * math.pi / lamda) * n_minus_1

    return phase

def wrap_phase_to_2pi(phase):
    """
    Wrap phase values to the range [0, 2π] by adding integer multiples of 2π.

    Args:
        phase (torch.Tensor): Input phase tensor (in radians).

    Returns:
        torch.Tensor: Wrapped phase tensor in [0, 2π].
    """
    return phase - 2 * math.pi * torch.floor(phase / (2 * math.pi))

def mask_outside_radius(tensor, invalid_value=float('nan')):
    """
    将tensor中半径大于N/2的区域设置为无效值
    参数：
        tensor: 输入的[N,N] PyTorch Tensor
        invalid_value: 要设置的无效值，默认为NaN
    返回：
        处理后的tensor
    """
    # 获取tensor的尺寸
    N = tensor.shape[0]
    if tensor.shape[1] != N:
        raise ValueError("Input tensor must be square [N,N]")

    # 创建坐标网格
    x = torch.arange(N, dtype=torch.float32) - (N - 1) / 2  # 从中心点偏移
    y = torch.arange(N, dtype=torch.float32) - (N - 1) / 2
    X, Y = torch.meshgrid(x, y, indexing='ij')  # 生成二维坐标网格

    # 计算每个点到中心的距离
    distances = torch.sqrt(X ** 2 + Y ** 2)

    # 创建掩码：半径大于N/2的区域
    mask = distances > N / 2

    # 复制输入tensor并应用掩码
    result = tensor
    result[mask] = 0

    return result


def normalize_channels(tensor):
    """
    对[b, c, h, w]格式的PyTorch Tensor进行逐通道归一化，使用矩阵运算
    Args:
        tensor: PyTorch Tensor，形状为[b, c, h, w]
    Returns:
        normalized_tensor: 归一化后的Tensor，值在[0,1]范围内
    """
    # 确保输入是4维Tensor
    if len(tensor.shape) != 4:
        raise ValueError("Input tensor must be 4D with shape [batch, channels, height, width]")

    # 获取每个通道的最小值和最大值，沿着h和w维度计算
    min_vals = tensor.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    max_vals = tensor.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

    # 避免除以零，通过设置一个掩码
    range_vals = max_vals - min_vals
    range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)

    # 进行min-max归一化
    normalized_tensor = (tensor - min_vals) / range_vals

    # 将值限制在[0,1]范围内（处理可能的数值误差）
    normalized_tensor = torch.clamp(normalized_tensor, 0.0, 1.0)

    return normalized_tensor


def create_target_psf(psf):
    """
    根据输入PSF形状生成target_psf，所有通道中心像素为1，其余为0
    参数:
        psf: 输入PSF tensor，形状为(B, C, H, W)
    返回:
        target_psf: 与psf同形状的tensor，每个通道中心像素为1，其余为0
    """
    # 验证输入tensor维度
    if psf.dim() != 4:
        raise ValueError("Input PSF must have 4 dimensions (B, C, H, W)")

    # 获取输入tensor的形状
    B, C, H, W = psf.shape

    # 创建全零tensor
    target_psf = torch.zeros_like(psf)

    # 计算中心像素坐标（考虑奇偶情况）
    center_h = H // 2
    center_w = W // 2

    # 将每个通道的中心像素设为1
    target_psf[:, :, center_h, center_w] = 1.0

    return target_psf
