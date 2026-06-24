from HyperDepthReconstruction import DOEHyperDepthCamera
from argparse import Namespace
from utils.helper import *
import cupy as cp


def downsample_numpy(data, factor):
    """
    使用 NumPy 进行均值下采样
    参数:
        data: numpy.ndarray, 输入的 [N,N] 二维数组
        factor: int, 下采样系数
    返回:
        numpy.ndarray, 下采样后的数组
    """
    N = data.shape[0]
    return data.reshape(N // factor, factor, N // factor, factor).mean(axis=(1, 3))


def downsample_1d(data, factor):
    """
    对[N]一维数组进行下采样

    参数:
        data: numpy.ndarray, 输入的[N]一维数组
        factor: int, 下采样系数（必须为正整数）

    返回:
        numpy.ndarray, 下采样后的数组
    """
    # 输入验证
    if not isinstance(data, np.ndarray) or data.ndim != 1:
        raise ValueError("输入必须为一维numpy数组")
    if not isinstance(factor, int) or factor < 1:
        raise ValueError("下采样系数必须为正整数")

    # 获取输入尺寸
    N = data.shape[0]

    # 检查输入尺寸是否能被下采样系数整除
    if N % factor != 0:
        raise ValueError("输入数组尺寸必须能被下采样系数整除")

    # 计算输出尺寸
    output_size = N // factor

    # 进行下采样（取均值）
    return data.reshape(output_size, factor).mean(axis=1)


def transform_negative(data, lambda_val=546.1e-9, n=1.556):
    [height, width] = data.shape

    coeff = lambda_val / (n - 1)
    result = data

    for i in range(height):
        for j in range(width):
            if data[i, j] < 0:
                m_float = -data[i, j] * (n - 1) / lambda_val
                # 向上取整
                m = int(m_float + 1) if m_float != int(m_float) else int(m_float)
                # 检查是否正好等于0，若是则m加1
                temp_result = data[i, j] + coeff * m
                if temp_result <= 0:
                    m += 1
                result[i, j] = data[i, j] + coeff * m
    return result


def transform_negative_gpu(data, lambda_val=546.1e-9, n=1.556):
    # 将输入转换为 CuPy 数组
    data_gpu = cp.array(data)

    # 获取数组尺寸
    # height, width = data_gpu.shape

    # 预计算系数
    coeff = lambda_val / (n - 1)

    # 初始化结果为输入的副本
    result = data_gpu.copy()

    # 负值掩码
    neg_mask = data_gpu < 0

    if cp.any(neg_mask):
        # 对负值计算 m_float
        m_float = -data_gpu[neg_mask] * (n - 1) / lambda_val

        # 计算 m：若 m_float 是整数，取 floor；否则取 floor + 1
        m = cp.where(m_float == cp.floor(m_float),
                     cp.floor(m_float),
                     cp.floor(m_float) + 1).astype(cp.int32)

        # 计算临时结果
        temp_result = data_gpu[neg_mask] + coeff * m

        # 调整 m：若 temp_result <= 0，则 m += 1
        adjust_mask = temp_result <= 0
        m[adjust_mask] += 1

        # 更新负值区域的结果
        result[neg_mask] = data_gpu[neg_mask] + coeff * m

    # 将结果传回 CPU
    return cp.asnumpy(result)



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
    # result[mask] = invalid_value
    result[mask] = 0
    return result

def main():
    # Load the saved checkpoint
    # This is not a default way to load the checkpoint through Lightning.
    checkpoint = torch.load('D:\Software\Pycharm\PycharmProject/NewDOECamera\DepthHyperspectralReconstruction'+
                            '\data\logs\DOEHyperDepthReconstruction/version_3\checkpoints/best-epoch=33-val_loss=0.27.ckpt', map_location=lambda storage, loc: storage)
    hparams = checkpoint['hyper_parameters']
    hparams = Namespace(**hparams)
    print(hparams)
    model = DOEHyperDepthCamera(hparams)
    model.load_state_dict(checkpoint['state_dict'])

    # heightmap1d = model.camera.heightmap1d_
    heightmap1d = model.camera.param
    print("heightmap1d shape:", heightmap1d.shape)
    print("heightmap1d max:", heightmap1d.max())
    print("heightmap1d min:", heightmap1d.min())

    # heightmap1d = downsample_1d(heightmap1d.detach().cpu().numpy(), 1)


    heightmap1d = transform_negative_gpu(heightmap1d.detach().cpu().numpy())
    heightmap1d = torch.from_numpy(heightmap1d)
    heightmap = radial_symmetry(heightmap1d.unsqueeze(0))
    print("heightmap shape:", heightmap.shape)
    heightmap = heightmap.squeeze()
    heightmap = heightmap.detach().cpu().numpy()

    # heightmap = downsample_numpy(heightmap.detach().cpu().numpy(), 10)
    print("heightmap shape:", heightmap.shape)

    plt.figure(figsize=(6, 6))  # 设置图像大小
    img = plt.imshow(heightmap, cmap='gray')  # 显示图像
    plt.axis('off')  # 关闭坐标轴
    plt.title('Heightmap')
    plt.colorbar(img)  # 添加颜色条
    # plt.savefig('heightmapF.png', dpi=300, bbox_inches='tight')  # 保存图片

    plt.show()
    #
    # heightmap_positive = transform_negative_gpu(heightmap)
    heightmap_positive = heightmap
    heightmap_positive = heightmap_positive + 1e-6
    print("heightmap_positive max", np.max(heightmap_positive))
    print("heightmap_positive min", np.min(heightmap_positive))
    heightmap_positive = mask_outside_radius(heightmap_positive)


    plt.figure(figsize=(6, 6))  # 设置图像大小
    img = plt.imshow(heightmap_positive, cmap='gray')  # 显示图像
    plt.axis('off')  # 关闭坐标轴
    plt.title('heightmap_positive')
    plt.colorbar(img)  # 添加颜色条
    plt.savefig('heightmapF_positive.png', dpi=300, bbox_inches='tight')
    plt.show()

    # np.save('heightmapF_positive_noSubstrate.npy', heightmap_positive)
    heightmap_max = np.max(heightmap_positive)
    print('heightmap_max', heightmap_max)
    heightmap_normalized = heightmap_positive / heightmap_max
    plt.imsave('HeightmapOfDepthSpectralLens_30mm.png', heightmap_normalized, cmap='gray')

    # print(heightmap_positive)


if __name__ == "__main__":
    main()