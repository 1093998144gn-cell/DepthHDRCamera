import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import scipy.ndimage as ndimage

DATA_DIR ='F:\Dataset\syn'

class SynDataset(Dataset):
    """
    自定义PyTorch Dataset，用于加载syn数据集的disp0_0.npy和refimg0_0.npy文件。
    数据将转换为PyTorch Tensor，归一化到[0, 1]，必须进行随机裁剪，并包含数据增强功能。

    Args:
        data_dir (str): 数据集根目录
        dataset (str): 数据集类型，'train' 或 'val'
        image_sz (tuple): 随机裁剪后的图像尺寸 (height, width)，必填
        augment (bool): 是否应用数据增强（仅对训练集有效）
    """

    def __init__(self, data_dir=r'F:\Dataset\syn', dataset='train', image_sz=None, augment=True):
        super(SynDataset, self).__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.augment = augment and dataset == 'train'  # 仅训练集应用增强
        self.image_sz = image_sz

        # 确保image_sz已提供
        if self.image_sz is None:
            raise ValueError("image_sz must be provided as a tuple (height, width)")

        # 数据集子文件夹
        self.sub_datasets = ['CAVE', 'HSDB', 'Munsell_color_chips']

        # 设置数据集路径
        if dataset not in ['train', 'val']:
            raise ValueError("dataset must be 'train' or 'val'")

        # 收集样本
        self.sample_ids = []
        for sub_dataset in self.sub_datasets:
            data_path = os.path.join(self.data_dir, sub_dataset, dataset)
            if not os.path.exists(data_path):
                print(f"数据集路径不存在: {data_path}")
                continue

            for sample_dir in sorted(os.listdir(data_path)):
                sample_path = os.path.join(data_path, sample_dir)
                if not os.path.isdir(sample_path):
                    continue
                disparity_path = os.path.join(sample_path, 'disp0_0.npy')
                spectral_path = os.path.join(sample_path, 'refimg0_0.npy')
                if os.path.exists(disparity_path) and os.path.exists(spectral_path):
                    sample_id = {
                        'spectral_path': spectral_path,
                        'disparity_path': disparity_path,
                        'id': f"{sub_dataset}/{sample_dir}",
                    }
                    self.sample_ids.append(sample_id)
                else:
                    print(f'视差或光谱图像不存在！: {disparity_path}')

        if not self.sample_ids:
            raise ValueError(f"No valid samples found in {dataset} dataset")

    def __len__(self):
        return len(self.sample_ids)

    def _normalize(self, data):
        """
        最小-最大归一化：将数据缩放到[0, 1]
        """
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min == 0:
            return data  # 防止除以零，直接返回原数据
        return (data - data_min) / (data_max - data_min)

    def _crop_data(self, refimg, disp):
        """
        随机裁剪到指定尺寸。
        """
        h, w = self.image_sz
        img_h, img_w = refimg.shape[0:2]  # 假设refimg和disp形状一致
        if h > img_h or w > img_w:
            raise ValueError(f"裁剪尺寸 {self.image_sz} 超过图像尺寸 ({img_h}, {img_w})")

        # 随机选择裁剪起始点
        max_h_start = img_h - h
        max_w_start = img_w - w
        h_start = random.randint(0, max_h_start)
        w_start = random.randint(0, max_w_start)

        # 裁剪refimg和disp
        refimg = refimg[h_start:h_start + h, w_start:w_start + w, :].copy()
        disp = disp[h_start:h_start + h, w_start:w_start + w].copy()

        return refimg, disp

    def _augment_data(self, refimg, disp):
        """
        数据增强：随机水平翻转、垂直翻转和旋转。
        使用.copy()确保正向步幅。
        """
        # 随机水平翻转
        if random.random() > 0.5:
            refimg = np.fliplr(refimg).copy()
            disp = np.fliplr(disp).copy()

        # 随机垂直翻转
        if random.random() > 0.5:
            refimg = np.flipud(refimg).copy()
            disp = np.flipud(disp).copy()

        # 随机旋转（±15度）
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            refimg = ndimage.rotate(refimg, angle, reshape=False, mode='nearest').copy()
            disp = ndimage.rotate(disp, angle, reshape=False, mode='nearest').copy()

        return refimg, disp

    def __getitem__(self, idx):
        """
        获取单个样本，进行归一化、随机裁剪、数据增强，并返回Tensor类型数据。
        """
        sample = self.sample_ids[idx]
        spectral_path = sample['spectral_path']
        disparity_path = sample['disparity_path']

        # 加载数据
        refimg = np.load(spectral_path)
        disp = np.load(disparity_path)

        refimg = refimg.squeeze()
        disp = disp.squeeze()

        # 转换为float32类型
        refimg = refimg.astype(np.float32)
        disp = disp.astype(np.float32)

        # 随机裁剪（对所有样本）
        refimg, disp = self._crop_data(refimg, disp)

        # 数据增强（仅训练集）
        if self.augment:
            refimg, disp = self._augment_data(refimg, disp)

        # 归一化到[0, 1]
        refimg = self._normalize(refimg)
        disp = self._normalize(disp)

        # 转换为PyTorch Tensor
        refimg = torch.from_numpy(refimg)
        disp = torch.from_numpy(disp)[None, ...]

        refimg = refimg.permute(2, 0, 1)

        return {
            'refimg': refimg,
            'disp': disp,
            'id': sample['id'],
        }


if __name__ == "__main__":
    # 示例用法
    # data_dir = r"C:\Dataset\syn\syn"
    image_sz = (384, 384)  # 裁剪尺寸

    # 创建训练集和验证集
    train_dataset = SynDataset(DATA_DIR, dataset='train', image_sz=image_sz, augment=True)
    val_dataset = SynDataset(DATA_DIR, dataset='val', image_sz=image_sz, augment=False)

    # 打印数据集信息
    print(f"训练数据集大小: {len(train_dataset)}")
    print(f"验证数据集大小: {len(val_dataset)}")

    # 获取一个样本
    sample = train_dataset[0]
    print(f"样本 refimg 形状: {sample['refimg'].shape}, 类型: {type(sample['refimg'])}, 最大: {sample['refimg'].max()}, 最小: {sample['refimg'].min()}")
    print(f"样本 disp 形状: {sample['disp'].shape}, 类型: {type(sample['disp'])}, 最大: {sample['disp'].max()}, 最小: {sample['disp'].min()}")
    print(f"样本 id: {sample['id']}")

