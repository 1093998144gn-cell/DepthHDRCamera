import torch
import matplotlib.pyplot as plt
from torch.cuda import device


class LightWave:
    def __init__(self, resolution, pitch, wavelengths, phase=None, amplitude=1.0, device='cuda'):
        """
        初始化光波类
        :param wavelengths: 波长列表 [num_wavelengths]
        :param resolution: 波的分辨率 (M, N)
        :param pitch: 采样间隔 (dx, dy)
        :param amplitude: 振幅
        :param wave_type: 波的类型 ('plane' 或 'spherical')
        :param depths: 仅用于球面波，表示传播深度列表 [num_depths]
        """
        self.wavelengths = wavelengths  # [1, num_wavelengths, 1, 1]
        self.resolution = resolution
        self.pitch = pitch

        self.amplitude = amplitude
        self.phase = phase

        if self.phase is not None and self.amplitude is not None:
            self.complex = self.amplitude * torch.exp(1j * self.phase)
        else:
            self.complex = None

        self.device = device

    def init_wave(self, depths=None, wave_type='plane'):
        """
        计算整个波场的波函数值
        :return: 复振幅值 [num_depths, num_wavelengths, M, N]
        """
        if wave_type == 'spherical' and depths is None:
            raise ValueError('depths must not be None when wave_type is spherical')

        M, N = self.resolution
        dx, dy = self.pitch
        x = torch.linspace(-int(M) // 2 , int(M) // 2, steps=int(M)) * dx
        y = torch.linspace(-int(N) // 2 , int(N) // 2, steps=int(N)) * dy
        X, Y = torch.meshgrid(x, y, indexing='ij')  # [M, N]
        X, Y = X.unsqueeze(0).unsqueeze(0).to(self.device), Y.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, M, N]
        k = 2 * torch.pi / self.wavelengths # [1, num_wavelengths, 1, 1]
        k = k.view(1, self.wavelengths.shape[0], 1, 1)

        if wave_type == 'plane':
            phase = torch.zeros(1, self.wavelengths.shape[0], M, N)
            self.complex = self.amplitude * torch.exp(1j * phase)
            return self.complex # [1, num_wavelengths, M, N]
        elif wave_type == 'spherical':
            depths = depths.view(-1, 1, 1, 1).to(self.device)  # [num_depths, 1, 1, 1]
            phase = k * (X **2 + Y **2) / (2 * depths)  # [num_depths, num_wavelengths, M, N]
            self.complex = self.amplitude * torch.exp(1j * phase)
            return self.complex  # [num_depths, num_wavelengths, M, N]
        else:
            raise ValueError("未知的波类型，应为 'plane' 或 'spherical'")

    def set_complex(self, comp):
        self.complex = comp

    def get_wave(self):
        return self.complex


# 测试函数
def test_lightwave():
    # 设置参数
    wavelengths = torch.tensor([500e-9, 600e-9])  # 两个波长: 500nm 和 600nm
    resolution = (128, 128)  # 分辨率
    pitch = (1e-6, 1e-6)  # 采样间隔 1微米
    num_depths = 2  # 两个深度
    phase = torch.zeros(num_depths, 2, 128, 128)  # 初始相位为零
    amplitude = 1.0
    device = 'cuda'  # 如果没有 GPU，使用 CPU

    # 测试 1: 初始化
    print("测试 1: 初始化 LightWave 类")
    light = LightWave(wavelengths, resolution, pitch, num_depths, phase, amplitude, device)
    print(f"初始复振幅形状: {light.complex.shape}")
    assert light.complex.shape == phase.shape, "初始复振幅形状错误"

    # 测试 2: 平面波
    print("\n测试 2: 生成平面波")
    try:
        plane_wave = light.init_wave(wave_type='plane')
        print(f"平面波形状: {plane_wave.shape}")
        assert plane_wave.shape == (1, 2, 128, 128), "平面波形状错误"

        # 可视化第一个波长的实部
        plt.figure(figsize=(8, 6))
        plt.imshow(plane_wave[0, 0].real.detach().cpu().numpy(), cmap='viridis')
        plt.title("Plane Wave Real Part (500nm)")
        plt.colorbar()
        plt.show()
    except ValueError as e:
        print(f"平面波生成失败: {e}")
    #
    # 测试 3: 球面波
    print("\n测试 3: 生成球面波")
    depths = torch.tensor([1e-3, 2e-3]) # 深度 1mm 和 2mm
    spherical_wave = light.init_wave(depths=depths, wave_type='spherical')
    print(f"球面波形状: {spherical_wave.shape}")
    assert spherical_wave.shape == (2, 2, 128, 128), "球面波形状错误"

    # 可视化第一个波长在第一个深度的实部
    plt.figure(figsize=(8, 6))
    plt.imshow(spherical_wave[0, 0].real.detach().cpu().numpy(), cmap='viridis')
    plt.title("Spherical Wave Real Part (500nm, 1mm)")
    plt.colorbar()
    plt.show()
    #
    # # 测试 4: get_wave 方法
    # print("\n测试 4: 检查 get_wave 方法")
    # current_wave = light.get_wave()
    # print(f"当前波形状: {current_wave.shape}")
    # assert current_wave.shape == (2, 2, 128, 128), "get_wave 返回形状错误"
    # print("当前波与球面波相同:", torch.allclose(current_wave, spherical_wave))



if __name__ == "__main__":
    test_lightwave()



