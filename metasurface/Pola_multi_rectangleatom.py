import os
import h5py
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import nn


def load_mat_to_tensor(folder_path, start_nm=400, end_nm=720, step_nm=10, file_name = 'xyT_', struct_name='lum', field_name='z'):
    """
    读取指定文件夹中的MAT v7.3文件中结构体的指定字段并转换为PyTorch Tensor。

    参数:
        folder_path (str): MAT文件所在的文件夹路径
        start_nm (int): 起始波长（默认400）
        end_nm (int): 结束波长（默认720）
        step_nm (int): 波长步长（默认10）
        struct_name (str): 结构体名称（默认'lum'）
        field_name (str): 结构体中的字段名称（默认'z'）

    返回:
        torch.Tensor: 堆叠后的Tensor，若无数据则返回None
    """
    # 生成文件名列表
    wavelengths = range(start_nm, end_nm + 1, step_nm)
    file_list = [file_name + f'{w}nm.mat' for w in wavelengths]

    # 初始化一个列表来存储所有数据
    tensor_list = []

    # 循环读取MAT文件并转换为Tensor
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            # 使用h5py加载MAT文件
            with h5py.File(file_path, 'r') as f:
                # 检查结构体是否存在
                if struct_name in f:
                    lum_group = f[struct_name]
                    if isinstance(lum_group, h5py.Group):
                        # 检查字段是否存在
                        if field_name in lum_group:
                            data = lum_group[field_name][()]  # 读取lum.z数据
                            data = np.array(data)  # 转换为NumPy数组
                            tensor = torch.tensor(data, dtype=torch.float32)
                            tensor_list.append(tensor)
                            print(f'成功加载并转换为Tensor: {file_name}, 形状: {tensor.shape}')
                        else:
                            print(f'文件 {file_name} 中的 {struct_name} 不包含字段 {field_name}')
                    else:
                        print(f'文件 {file_name} 中的 {struct_name} 不是一个组')
                else:
                    print(f'文件 {file_name} 中不存在结构体 {struct_name}')
        else:
            print(f'文件不存在: {file_name}')

    # 将所有Tensor堆叠成一个大Tensor
    if tensor_list:
        try:
            final_tensor = torch.stack(tensor_list)
            print(f'最终Tensor形状: {final_tensor.shape}')
            return final_tensor
        except RuntimeError as e:
            print(f'无法堆叠Tensor，可能形状不一致: {e}')
            return tensor_list  # 返回列表形式，避免堆叠错误
    else:
        print('没有成功加载任何文件')
        return None

class RectangleMLP(nn.Module):
    def __init__(self, input_size=3, hidden_size=256, output_size=3):  # 输入3维：wavelength, l, w
        super(RectangleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1),
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

def preprocess_data(phasedata, transdata):
    wavelengths = np.linspace(400e-9, 720e-9, 33)  # 波长范围
    l_coords = np.linspace(100e-9, 300e-9, 51)  # l 坐标
    w_coords = np.linspace(100e-9, 300e-9, 51)  # w 坐标

    # 归一化
    w_min, w_max = wavelengths.min(), wavelengths.max()
    l_min, l_max = l_coords.min(), l_coords.max()
    w_coord_min, w_coord_max = w_coords.min(), w_coords.max()

    # 生成输入数据 [wavelength, l, w]
    X = []
    y = []
    for i, wave in enumerate(wavelengths):
        for l in range(len(l_coords)):
            for w in range(len(w_coords)):
                X.append([(wave - w_min) / (w_max - w_min),
                          (l_coords[l] - l_min) / (l_max - l_min),
                          (w_coords[w] - w_coord_min) / (w_coord_max - w_coord_min)])
                phase = phasedata[i,l,w]  # 相位
                trans = transdata[i,l,w] # 幅度作为透射率
                y.append([np.sin(phase), np.cos(phase), trans])

    X = torch.tensor(X, dtype=torch.float32).to('cuda')
    y = torch.tensor(y, dtype=torch.float32).to('cuda')
    norm_params = [w_min, w_max, l_min, l_max, w_coord_min, w_coord_max]
    return X, y, norm_params


# 训练函数并保存模型
def train_model(model, X_train, y_train, epochs=3000, lr=0.001, save_path='PM_rectangle_atom.pth'):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))


    # 用于记录损失
    loss_history = []
    sin_loss_history = []
    cos_loss_history = []
    trans_loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss_sin = criterion(outputs[:, 0], y_train[:, 0])
        loss_cos = criterion(outputs[:, 1], y_train[:, 1])
        loss_trans = criterion(outputs[:, 2], y_train[:, 2])
        loss = 10 * loss_sin + 10 * loss_cos + loss_trans
        loss.backward()
        optimizer.step()

        # 记录损失
        loss_history.append(loss.item())
        sin_loss_history.append(loss_sin.item())
        cos_loss_history.append(loss_cos.item())
        trans_loss_history.append(loss_trans.item())

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

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), loss_history, label='Total Loss', color='blue')
    plt.plot(range(1, epochs + 1), sin_loss_history, label='Sin Loss', color='orange', linestyle='--')
    plt.plot(range(1, epochs + 1), cos_loss_history, label='Cos Loss', color='green', linestyle='--')
    plt.plot(range(1, epochs + 1), trans_loss_history, label='Trans Loss', color='red', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    print("损失曲线已保存至: training_loss.png")
    plt.show()

    return mae_phase, mae_trans

# 预测函数
def predict(model, wavelength, l, w, norm_params):
    w_min, w_max, l_min, l_max, w_coord_min, w_coord_max = norm_params
    input_data = torch.tensor([[(wavelength - w_min) / (w_max - w_min),
                                (l - l_min) / (l_max - l_min),
                                (w - w_coord_min) / (w_coord_max - w_coord_min)]],
                              dtype=torch.float32).to('cuda')
    with torch.no_grad():
        pred = model(input_data)
        phase = torch.atan2(pred[:, 0], pred[:, 1]).item()
        trans = pred[:, 2].item()
    return phase, trans



# 绘制谱图（预测和原始数据）
def plot_spectra(model, phase_data, trans_data, norm_params,
                 pred_phasex_path='pred_phasex_spectra.png', pred_transx_path='pred_transx_spectra.png',
                 raw_phasex_path='raw_phasex_spectra.png', raw_transx_path='raw_transx_spectra.png',
                 pred_phasey_path='pred_phasey_spectra.png', pred_transy_path='pred_transy_spectra.png',
                 raw_phasey_path='raw_phasey_spectra.png', raw_transy_path='raw_transy_spectra.png',
                 ):
    wavelengths = np.linspace(400e-9, 720e-9, 33)
    l_coords = np.linspace(100e-9, 300e-9, 51)
    w_coords = np.linspace(100e-9, 300e-9, 51)
    L, W = np.meshgrid(l_coords, w_coords)

    # 1. 预测相位谱x
    fig_phase, axes_phase = plt.subplots(5, 7, figsize=(20, 14), constrained_layout=True)
    axes_phase = axes_phase.flatten()
    for idx, wl in enumerate(wavelengths):
        phase_map = np.zeros((51, 51))
        for i in range(51):
            for j in range(51):
                phase, _ = predict(model, wl, L[i, j], W[i, j], norm_params)
                phase_map[j, i] = phase
        im = axes_phase[idx].imshow(phase_map, extent=[100, 300, 100, 300], cmap='viridis',
                                    origin='lower', vmin=-np.pi, vmax=np.pi)
        axes_phase[idx].set_title(f'{int(wl*1e9)} nm', fontsize=10)
        axes_phase[idx].set_xlabel('w (nm)', fontsize=8)
        axes_phase[idx].set_ylabel('l (nm)', fontsize=8)
        axes_phase[idx].tick_params(labelsize=6)
    for i in range(33, len(axes_phase)):
        fig_phase.delaxes(axes_phase[i])
    fig_phase.suptitle('Predicted Phasex Spectra (rad)', fontsize=16)
    plt.colorbar(im, ax=axes_phase, label='phase (rad)', fraction=0.046, pad=0.04)
    plt.savefig(pred_phasex_path, dpi=300, bbox_inches='tight')
    print(f"预测相位谱已保存至: {pred_phasex_path}")

    # 2. 预测透射谱x
    fig_trans, axes_trans = plt.subplots(5, 7, figsize=(20, 14), constrained_layout=True)
    axes_trans = axes_trans.flatten()
    for idx, wl in enumerate(wavelengths):
        trans_map = np.zeros((51, 51))
        for i in range(51):
            for j in range(51):
                _, trans = predict(model, wl, L[i, j], W[i, j], norm_params)
                trans_map[j, i] = trans
        im = axes_trans[idx].imshow(trans_map, extent=[100, 300, 100, 300], cmap='viridis',
                                    origin='lower', vmin=0, vmax=1)
        axes_trans[idx].set_title(f'{int(wl*1e9)} nm', fontsize=10)
        axes_trans[idx].set_xlabel('w(nm)', fontsize=8)
        axes_trans[idx].set_ylabel('l(nm)', fontsize=8)
        axes_trans[idx].tick_params(labelsize=6)
    for i in range(33, len(axes_trans)):
        fig_trans.delaxes(axes_trans[i])
    fig_trans.suptitle('Predicted Transmissionx Spectra', fontsize=16)
    plt.colorbar(im, ax=axes_trans, label='transmission', fraction=0.046, pad=0.04)
    plt.savefig(pred_transx_path, dpi=300, bbox_inches='tight')
    print(f"预测透射谱已保存至: {pred_transx_path}")

    # 3. 原始相位谱x
    fig_raw_phase, axes_raw_phase = plt.subplots(5, 7, figsize=(20, 14), constrained_layout=True)
    axes_raw_phase = axes_raw_phase.flatten()
    for idx in range(33):
        phase_map = phase_data[idx]  # 直接从原始数据提取相位
        im = axes_raw_phase[idx].imshow(phase_map, extent=[100, 300, 100, 300], cmap='viridis',
                                        origin='lower', vmin=-np.pi, vmax=np.pi)
        axes_raw_phase[idx].set_title(f'{int(wavelengths[idx]*1e9)} nm', fontsize=10)
        axes_raw_phase[idx].set_xlabel('w(nm)', fontsize=8)
        axes_raw_phase[idx].set_ylabel('l(nm)', fontsize=8)
        axes_raw_phase[idx].tick_params(labelsize=6)
    for i in range(33, len(axes_raw_phase)):
        fig_raw_phase.delaxes(axes_raw_phase[i])
    fig_raw_phase.suptitle('Raw Phase Spectrax (rad)', fontsize=16)
    plt.colorbar(im, ax=axes_raw_phase, label='phase (rad)', fraction=0.046, pad=0.04)
    plt.savefig(raw_phasex_path, dpi=300, bbox_inches='tight')
    print(f"原始相位谱已保存至: {raw_phasex_path}")

    # 4. 原始透射谱x
    fig_raw_trans, axes_raw_trans = plt.subplots(5, 7, figsize=(20, 14), constrained_layout=True)
    axes_raw_trans = axes_raw_trans.flatten()
    for idx in range(33):
        trans_map = trans_data[idx]  # 直接从原始数据提取透射率
        im = axes_raw_trans[idx].imshow(trans_map, extent=[100, 300, 100, 300], cmap='viridis',
                                        origin='lower', vmin=0, vmax=1)
        axes_raw_trans[idx].set_title(f'{int(wavelengths[idx]*1e9)} nm', fontsize=10)
        axes_raw_trans[idx].set_xlabel('w(nm)', fontsize=8)
        axes_raw_trans[idx].set_ylabel('l(nm)', fontsize=8)
        axes_raw_trans[idx].tick_params(labelsize=6)
    for i in range(33, len(axes_raw_trans)):
        fig_raw_trans.delaxes(axes_raw_trans[i])
    fig_raw_trans.suptitle('Raw Transmissionx Spectra', fontsize=16)
    plt.colorbar(im, ax=axes_raw_trans, label='transmission', fraction=0.046, pad=0.04)
    plt.savefig(raw_transx_path, dpi=300, bbox_inches='tight')
    print(f"原始透射谱已保存至: {raw_transx_path}")

    # 1. 预测相位谱y
    fig_phase, axes_phase = plt.subplots(5, 7, figsize=(20, 14), constrained_layout=True)
    axes_phase = axes_phase.flatten()
    for idx, wl in enumerate(wavelengths):
        phase_map = np.zeros((51, 51))
        for i in range(51):
            for j in range(51):
                phase, _ = predict(model, wl, W[i, j], L[i, j], norm_params)
                phase_map[j, i] = phase
        im = axes_phase[idx].imshow(phase_map, extent=[100, 300, 100, 300], cmap='viridis',
                                    origin='lower', vmin=-np.pi, vmax=np.pi)
        axes_phase[idx].set_title(f'{int(wl * 1e9)} nm', fontsize=10)
        axes_phase[idx].set_xlabel('w (nm)', fontsize=8)
        axes_phase[idx].set_ylabel('l (nm)', fontsize=8)
        axes_phase[idx].tick_params(labelsize=6)
    for i in range(33, len(axes_phase)):
        fig_phase.delaxes(axes_phase[i])
    fig_phase.suptitle('Predicted Phasey Spectra (rad)', fontsize=16)
    plt.colorbar(im, ax=axes_phase, label='phase (rad)', fraction=0.046, pad=0.04)
    plt.savefig(pred_phasey_path, dpi=300, bbox_inches='tight')
    print(f"预测相位谱已保存至: {pred_phasey_path}")

    # 2. 预测透射谱y
    fig_trans, axes_trans = plt.subplots(5, 7, figsize=(20, 14), constrained_layout=True)
    axes_trans = axes_trans.flatten()
    for idx, wl in enumerate(wavelengths):
        trans_map = np.zeros((51, 51))
        for i in range(51):
            for j in range(51):
                _, trans = predict(model, wl, W[i, j], L[i, j], norm_params)
                trans_map[j, i] = trans
        im = axes_trans[idx].imshow(trans_map, extent=[100, 300, 100, 300], cmap='viridis',
                                    origin='lower', vmin=0, vmax=1)
        axes_trans[idx].set_title(f'{int(wl * 1e9)} nm', fontsize=10)
        axes_trans[idx].set_xlabel('w(nm)', fontsize=8)
        axes_trans[idx].set_ylabel('l(nm)', fontsize=8)
        axes_trans[idx].tick_params(labelsize=6)
    for i in range(33, len(axes_trans)):
        fig_trans.delaxes(axes_trans[i])
    fig_trans.suptitle('Predicted Transmissiony Spectra', fontsize=16)
    plt.colorbar(im, ax=axes_trans, label='transmission', fraction=0.046, pad=0.04)
    plt.savefig(pred_transy_path, dpi=300, bbox_inches='tight')
    print(f"预测透射谱已保存至: {pred_transy_path}")

    # 3. 原始相位谱y
    fig_raw_phase, axes_raw_phase = plt.subplots(5, 7, figsize=(20, 14), constrained_layout=True)
    axes_raw_phase = axes_raw_phase.flatten()
    for idx in range(33):
        phase_map = phase_data[idx]  # 直接从原始数据提取相位
        phase_map = phase_map.T  # 转置以匹配预测数据
        im = axes_raw_phase[idx].imshow(phase_map, extent=[100, 300, 100, 300], cmap='viridis',
                                        origin='lower', vmin=-np.pi, vmax=np.pi)
        axes_raw_phase[idx].set_title(f'{int(wavelengths[idx] * 1e9)} nm', fontsize=10)
        axes_raw_phase[idx].set_xlabel('w(nm)', fontsize=8)
        axes_raw_phase[idx].set_ylabel('l(nm)', fontsize=8)
        axes_raw_phase[idx].tick_params(labelsize=6)
    for i in range(33, len(axes_raw_phase)):
        fig_raw_phase.delaxes(axes_raw_phase[i])
    fig_raw_phase.suptitle('Raw Phase Spectray (rad)', fontsize=16)
    plt.colorbar(im, ax=axes_raw_phase, label='phase (rad)', fraction=0.046, pad=0.04)
    plt.savefig(raw_phasey_path, dpi=300, bbox_inches='tight')
    print(f"原始相位谱已保存至: {raw_phasey_path}")

    # 4. 原始透射谱y
    fig_raw_trans, axes_raw_trans = plt.subplots(5, 7, figsize=(20, 14), constrained_layout=True)
    axes_raw_trans = axes_raw_trans.flatten()
    for idx in range(33):
        trans_map = trans_data[idx]  # 直接从原始数据提取透射率
        trans_map = trans_map.T
        im = axes_raw_trans[idx].imshow(trans_map, extent=[100, 300, 100, 300], cmap='viridis',
                                        origin='lower', vmin=0, vmax=1)
        axes_raw_trans[idx].set_title(f'{int(wavelengths[idx] * 1e9)} nm', fontsize=10)
        axes_raw_trans[idx].set_xlabel('w(nm)', fontsize=8)
        axes_raw_trans[idx].set_ylabel('l(nm)', fontsize=8)
        axes_raw_trans[idx].tick_params(labelsize=6)
    for i in range(33, len(axes_raw_trans)):
        fig_raw_trans.delaxes(axes_raw_trans[i])
    fig_raw_trans.suptitle('Raw Transmissiony Spectra', fontsize=16)
    plt.colorbar(im, ax=axes_raw_trans, label='transmission', fraction=0.046, pad=0.04)
    plt.savefig(raw_transy_path, dpi=300, bbox_inches='tight')
    print(f"原始透射谱已保存至: {raw_transy_path}")

    plt.show()

# 主程序
def main():
    # 加载数据
    phase_folder = 'D:\Data\Dataset/atomlibrary\XYPX_700'
    phase_data = load_mat_to_tensor(phase_folder, file_name='xyP_')
    if phase_data is None or phase_data.shape != (33, 51, 51):
        raise ValueError("数据加载失败或形状不符，期望 [33, 51, 51]")

    trans_folder = 'D:\Data\Dataset/atomlibrary\XYTX_700'
    trans_data = load_mat_to_tensor(trans_folder, file_name='xyT_')
    if trans_data is None or trans_data.shape != (33, 51, 51):
        raise ValueError("数据加载失败或形状不符，期望 [33, 51, 51]")

    # 预处理数据
    X_train, y_train, norm_params = preprocess_data(phase_data, trans_data)

    # 初始化并训练模型
    model = RectangleMLP().to('cuda')
    train_model(model, X_train, y_train, epochs=5000, lr=0.01)

    plot_spectra(model,phase_data, trans_data, norm_params)




if __name__ == "__main__":
    main()


