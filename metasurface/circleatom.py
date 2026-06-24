import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义增强的MLP模型
class CircleMLP(nn.Module):
    def __init__(self, input_size=2, hidden_size=256, output_size=3):  # 输出3维：sin(phase), cos(phase), trans
        super(CircleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1),
            # nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(0.1),
            # nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, output_size)
        )
        self.trans_max = 1.0

    def forward(self, x):
        output = self.network(x)
        sin_phase = output[:, 0]  # sin(phase) 分量，范围 [-1, 1]
        cos_phase = output[:, 1]  # cos(phase) 分量，范围 [-1, 1]
        trans = torch.sigmoid(output[:, 2]) * self.trans_max  # 透射率 [0, 1]
        return torch.stack([sin_phase, cos_phase, trans], dim=1)


# 数据预处理
def preprocess_data(data):
    wavelength = data[0, :]
    diameter = data[1, :]
    phase = data[2, :]  # 假设相位在 -π 到 π
    trans = data[3, :]

    w_min, w_max = np.min(wavelength), np.max(wavelength)
    d_min, d_max = np.min(diameter), np.max(diameter)
    X = np.vstack([(wavelength - w_min) / (w_max - w_min),
                   (diameter - d_min) / (d_max - d_min)]).T
    # 相位编码为 sin 和 cos
    y = np.vstack([np.sin(phase), np.cos(phase), trans]).T

    return (torch.tensor(X, dtype=torch.float32).to(device),
            torch.tensor(y, dtype=torch.float32).to(device),
            [w_min, w_max, d_min, d_max])


# 训练函数并保存模型
def train_model(model, X_train, y_train, epochs=2000, lr=0.001, save_path='circle_atom.pth'):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss_sin = criterion(outputs[:, 0], y_train[:, 0])
        loss_cos = criterion(outputs[:, 1], y_train[:, 1])
        loss_trans = criterion(outputs[:, 2], y_train[:, 2])
        loss = loss_sin + loss_cos + loss_trans
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Total Loss: {loss.item():.6f}, '
                  f'Sin Loss: {loss_sin.item():.6f}, Cos Loss: {loss_cos.item():.6f}, '
                  f'Trans Loss: {loss_trans.item():.6f}')

    # 计算MAE
    with torch.no_grad():
        predictions = model(X_train)
        pred_phase = torch.atan2(predictions[:, 0], predictions[:, 1])
        true_phase = torch.atan2(y_train[:, 0], y_train[:, 1])
        mae_phase = torch.mean(torch.abs(pred_phase - true_phase)).item()
        mae_trans = torch.mean(torch.abs(predictions[:, 2] - y_train[:, 2])).item()
    print(f"\n训练完成:")
    print(f"相位平均绝对误差 (MAE): {mae_phase:.6f} rad")
    print(f"透射率平均绝对误差 (MAE): {mae_trans:.6f}")

    # 保存模型
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至: {save_path}")

    return mae_phase, mae_trans


# 预测函数
def predict(model, diameter, wavelength, norm_params):
    w_min, w_max, d_min, d_max = norm_params
    input_data = torch.tensor([[(wavelength - w_min) / (w_max - w_min),
                                (diameter - d_min) / (d_max - d_min)]],
                              dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(input_data)
        phase = torch.atan2(pred[:, 0], pred[:, 1]).item()  # 恢复相位
        trans = pred[:, 2].item()
    return phase, trans


# 绘制相位谱和透射谱
def plot_spectra(model, norm_params, save_fig_path='circle_atom_spectra.png'):
    diameters = np.linspace(100e-9, 300e-9, 100)
    wavelengths = np.linspace(400e-9, 720e-9, 100)
    D, W = np.meshgrid(diameters, wavelengths)
    phase_map = np.zeros_like(D)
    trans_map = np.zeros_like(D)

    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            phase, trans = predict(model, D[i, j], W[i, j], norm_params)
            phase_map[i, j] = phase
            trans_map[i, j] = trans

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(phase_map, extent=[100, 300, 400, 720],
                     aspect='auto', cmap='viridis', origin='lower', vmin=-np.pi, vmax=np.pi)
    ax1.set_xlabel('diameter (nm)')
    ax1.set_ylabel('wavelength (nm)')
    ax1.set_title('phase (rad)')
    plt.colorbar(im1, ax=ax1, label='phase (rad)')

    im2 = ax2.imshow(trans_map, extent=[100, 300, 400, 720],
                     aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=1)
    ax2.set_xlabel('diameter (nm)')
    ax2.set_ylabel('wavelength (nm)')
    ax2.set_title('transmission')
    plt.colorbar(im2, ax=ax2, label='transmission')

    plt.tight_layout()

    # 保存图像
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存至: {save_fig_path}")

    plt.show()


# 主程序
# def main():
#     mat_data = sio.loadmat("D:\Data\Dataset/atomlibrary\circle.mat")  # 替换为你的文件名
#     data = mat_data['circle']
#
#     if data.shape != (4, 1683):
#         raise ValueError("数据形状应为 4×1683")
#
#     X_train, y_train, norm_params = preprocess_data(data)
#     model = CircleMLP(hidden_size=256).to(device)
#     mae_phase, mae_trans = train_model(model, X_train, y_train, epochs=2000, lr=0.001,
#                                        save_path='circle_atom.pth')
#
#     test_diameter = data[1, 0]
#     test_wavelength = data[0, 0]
#     phase, trans = predict(model, test_diameter, test_wavelength, norm_params)
#
#     print(f"\n测试输入: 直径 = {test_diameter:.3f}, 波长 = {test_wavelength:.3f}")
#     print(f"预测相位响应: {phase:.3f} rad ({phase * 180 / np.pi:.1f}°)")
#     print(f"预测透射率: {trans:.3f}")
#     print(f"真实相位: {data[2, 0]:.3f} rad")
#     print(f"真实透射率: {data[3, 0]:.3f}")
#
#     plot_spectra(model, norm_params)
#
#
# if __name__ == "__main__":
#     main()