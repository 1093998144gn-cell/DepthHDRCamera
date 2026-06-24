import torch
import numpy as np
from matplotlib import pyplot as plt

from metasurface.circleatom import CircleMLP


def optimize_size_distribution(model, target_phase, wavelengths, norm_params,
                                 N, max_steps=1000, lr=1e-9, trans_weight=0.0,
                                 verbose=True):
    """
        优化超表面直径分布以匹配目标相位分布。

        参数:
            model (nn.Module): 已训练的 CircleMLP 模型
            target_phase (torch.Tensor): 目标相位分布，形状 [num_wavelengths, N, N]，单位：弧度
            wavelengths (torch.Tensor): 波长数组，形状 [num_wavelengths]
            norm_params (list): 归一化参数 [w_min, w_max, d_min, d_max]
            N (int): 超表面网格大小（N x N）
            max_steps (int): 最大优化步数，默认 10004
            lr (float): 学习率，默认 1e-9（因直径单位为纳米）
            trans_weight (float): 透射率损失的权重，默认 0.0（不考虑透射率约束）
            verbose (bool): 是否打印优化过程，默认 True

        返回:
            diameters (torch.Tensor): 优化后的直径分布，形状 [N, N]，单位：米
            final_loss (float): 最终损失值
    """
    device = 'cuda'
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    # 提取归一化参数
    w_min, w_max, d_min, d_max = norm_params

    # 检查输入形状
    num_wavelengths = wavelengths.shape[0]
    if target_phase.shape != (num_wavelengths, N, N):
        raise ValueError(f"目标相位形状应为 [{num_wavelengths}, {N}, {N}]，但得到 {target_phase.shape}")

    # 初始化直径分布
    diameters = torch.tensor(np.random.uniform(d_min, d_max, (N, N)),
                            dtype=torch.float32, requires_grad=True, device=device)



    # 优化器
    optimizer = torch.optim.Adam([diameters], lr=lr)

    # 优化循环
    for step in range(max_steps):
        optimizer.zero_grad()

        # 构造输入
        W = wavelengths.view(num_wavelengths, 1, 1).expand(num_wavelengths, N, N)
        D = diameters.expand(num_wavelengths, N, N)
        X = torch.stack([(W - w_min) / (w_max - w_min),
                         (D - d_min) / (d_max - d_min)], dim=-1).view(-1, 2)

        # 前向传播，模型参数不会更新
        pred = model(X)
        pred_phase = torch.atan2(pred[:, 0], pred[:, 1]).view(num_wavelengths, N, N)
        pred_trans = pred[:, 2].view(num_wavelengths, N, N)

        # 计算损失
        loss = phase_loss(pred_phase, target_phase)
        if trans_weight > 0:
            trans_loss = torch.mean((1.0 - pred_trans) ** 2)
            loss += trans_weight * trans_loss

        # 反向传播，只更新 diameters 的梯度
        loss.backward()
        optimizer.step()

        # 约束直径范围
        with torch.no_grad():
            diameters.clamp_(min=d_min, max=d_max)

        # 打印进度
        if verbose and step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}")

    # 最终损失
    final_loss = loss.item()
    if verbose:
        print(f"优化完成，最终损失: {final_loss:.6f}")

    with torch.no_grad():
        # 构造输入
        W = wavelengths.view(num_wavelengths, 1, 1).expand(num_wavelengths, N, N)
        D = diameters.expand(num_wavelengths, N, N)
        X = torch.stack([(W - w_min) / (w_max - w_min),
                         (D - d_min) / (d_max - d_min)], dim=-1).view(-1, 2)

        # 前向传播，模型参数不会更新
        pred = model(X)
        pred_phase = torch.atan2(pred[:, 0], pred[:, 1]).view(num_wavelengths, N, N)
        pred_trans = pred[:, 2].view(num_wavelengths, N, N)

        # 可视化
        plt.imshow(pred_phase.squeeze().cpu().numpy(), cmap='viridis', extent=[0, 1, 0, 1])
        plt.colorbar(label='Phase (rad)')
        plt.title(f'Predict Sawtooth Phase Distribution')
        plt.xlabel('x (normalized)')
        plt.ylabel('y (normalized)')
        plt.show()


    return diameters.detach(), final_loss

# 相位损失函数
def phase_loss(pred_phase, target_phase):
    diff = torch.remainder(pred_phase - target_phase + np.pi, 2 * np.pi) - np.pi
    return torch.mean(diff ** 2)


def generate_sawtooth_phase(N, periods=1, direction='x', phase_range=(-np.pi, np.pi)):
    """
    生成 N x N 的锯齿状梯度相位分布。

    参数:
        N (int): 网格大小 (N x N)
        periods (float): 相位周期数（沿指定方向的锯齿重复次数），默认 1
        direction (str): 梯度方向，'x'（水平）或 'y'（垂直），默认 'x'
        phase_range (tuple): 相位范围，例如 (-π, π)，默认 (-π, π)

    返回:
        phase (torch.Tensor): 锯齿状相位分布，形状 [N, N]，单位：弧度
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 生成网格坐标
    coords = torch.linspace(0, 1, N, device=device)  # 归一化坐标 [0, 1]

    # 根据方向生成梯度
    if direction == 'x':
        gradient = coords.view(1, N).expand(N, N)  # 沿 x 方向变化，y 方向复制
    elif direction == 'y':
        gradient = coords.view(N, 1).expand(N, N)  # 沿 y 方向变化，x 方向复制
    else:
        raise ValueError("direction 必须是 'x' 或 'y'")

    # 计算锯齿波相位
    min_phase, max_phase = phase_range
    phase_amplitude = max_phase - min_phase  # 相位范围幅度
    phase = min_phase + phase_amplitude * (gradient * periods % 1.0)

    return phase


# 示例使用
if __name__ == "__main__":
    num_wavelengths = 10
    N = 32
    focal_length = 1e-3  # 焦距 1mm

    # 生成相位分布
    target_phase = generate_sawtooth_phase(N, periods=6)

    # 可视化
    plt.imshow(target_phase.cpu().numpy(), cmap='viridis', extent=[0, 1, 0, 1])
    plt.colorbar(label='Phase (rad)')
    plt.title(f'Sawtooth Phase Distribution')
    plt.xlabel('x (normalized)')
    plt.ylabel('y (normalized)')
    plt.show()

    # 打印形状和范围
    print(f"相位分布形状: {target_phase.shape}")
    print(f"相位范围: [{target_phase.min().item():.3f}, {target_phase.max().item():.3f}] rad")





    # 假设已有模型

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CircleMLP().to(device)
    model.load_state_dict(torch.load('circle_atom.pth'))  # 加载训练好的模型

    # 构造示例输入
    num_wavelengths = 1
    wavelengths = torch.tensor([650e-9], device=device)
    norm_params = [400e-9, 720e-9, 100e-9, 300e-9]  # 从训练数据中获取

    target_phase = target_phase.unsqueeze(0).expand(num_wavelengths, -1, -1)

    # 调用函数
    diameters, final_loss = optimize_size_distribution(
        model, target_phase, wavelengths, norm_params, N,
        max_steps=1000, lr=1e-8, trans_weight=0, verbose=True
    )

    # 输出结果
    print("优化后的直径分布 (单位: nm):")
    print(diameters.cpu().numpy() * 1e9)  # 转换为纳米单位显示
    print(f"最终损失: {final_loss:.6f}")


