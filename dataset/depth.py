import random
from typing import Tuple
import os
import torch
import numpy as np
import imageio
from torch.utils.data import Dataset




class SceneFlow(Dataset):

    def __init__(self, root_dir = 'depth', crop_size = (256, 256), augment = True, dataset = 'train'):
        """
        :param root_dir: 根目录
        :param crop_size: 裁剪尺寸
        :param depth_min: 最小深度
        :param depth_max: 最大深度
        :param augment: 数据增强
        """
        super().__init__()
        if dataset == 'train':
            image_dirs = [os.path.join(root_dir, 'FlyingThings3D_subset_image_clean\FlyingThings3D_subset/train\image_clean\left')]
            disparity_dirs = [os.path.join(root_dir, 'FlyingThings3D_subset_disparity\FlyingThings3D_subset/train\disparity\left')]
        elif dataset == 'val':
            image_dirs = [os.path.join(root_dir, 'FlyingThings3D_subset_image_clean\FlyingThings3D_subset/val\image_clean\left')]
            disparity_dirs = [os.path.join(root_dir, 'FlyingThings3D_subset_disparity\FlyingThings3D_subset/val\disparity\left')]
        else:
            raise ValueError(f'dataset ({dataset}) has to be "train," "val')

        self.crop_size = crop_size
        self.augment = augment

        self.sample_ids = []
        for image_dir, disparity_dir in zip(image_dirs, disparity_dirs):
            for filename in sorted(os.listdir(image_dir)):
                if '.png' in filename:
                    id = os.path.splitext(filename)[0]
                    disparity_path = os.path.join(disparity_dir, f'{id}.pfm')
                    if os.path.exists(disparity_path):
                        sample_id = {
                            'image_dir': image_dir,
                            'disparity_dir': disparity_dir,
                            'id': id,
                        }
                        self.sample_ids.append(sample_id)
                    else:
                        print(f'Disparity image does not exist!: {disparity_path}')


    def random_crop(self, array, top, left):
        """从缩放后的数组中裁剪到 crop_size，使用指定的 top 和 left"""
        h, w = array.shape[-2], array.shape[-1]
        ph, pw = self.crop_size

        if h < ph or w < pw:
            raise ValueError(f"缩放后大小 {h}x{w} 小于裁剪大小 {ph}x{pw}")
        if top + ph > h or left + pw > w:
            raise ValueError(f"裁剪位置 ({top}, {left}) 超出范围")

        if array.ndim == 2:
            return array[top:top + ph, left:left + pw]
        elif array.ndim == 3:
            return array[:, top:top + ph, left:left + pw]
        else:
            raise ValueError("数组维度必须为 2 或 3")

    def augment_data(self, depth_patch, hs_patch):
        """对深度和高光谱数据应用相同的数据增强"""
        if not self.augment:
            return depth_patch, hs_patch

        # 随机水平翻转
        if random.random() > 0.5:
            depth_patch = np.flip(depth_patch, axis=-1)
            hs_patch = np.flip(hs_patch, axis=-1)

        # 随机垂直翻转
        if random.random() > 0.5:
            depth_patch = np.flip(depth_patch, axis=-2)
            hs_patch = np.flip(hs_patch, axis=-2)

        # 随机旋转（0°, 90°, 180°, 270°）
        k = random.randint(0, 3)
        if k > 0:
            depth_patch = np.rot90(depth_patch, k, axes=(-2, -1))
            hs_patch = np.rot90(hs_patch, k, axes=(-2, -1))

        return depth_patch, hs_patch

    def normalize_depth(self, depth_patch):
        depth_max = depth_patch.max()
        depth_min = depth_patch.min()
        return (depth_patch - depth_min) / (depth_max - depth_min + 1e-8)

    def normalize_image(self, image_patch):
        return image_patch / 255.0 if image_patch.max() > 0 else image_patch

    def read_pfm(self, file_path):
        with open(file_path, 'rb') as f:
            header = f.readline().decode().strip()
            if header != 'Pf':
                raise ValueError(f"Invalid PFM header: {header}")
            dims = f.readline().decode().strip().split()
            width, height = int(dims[0]), int(dims[1])
            scale = float(f.readline().decode().strip())
            endian = 'little' if scale < 0 else 'big'
            data = np.fromfile(f, '<f4' if endian == 'little' else '>f4')
            data = data.reshape(height, width)
            data = np.flipud(data)  # 上下翻转
        return data.astype(np.float32)


    def __len__(self):
        return len(self.sample_ids)


    def __getitem__(self, index):
        if index >= len(self.sample_ids):
            raise IndexError("索引超出范围")

        sample_id = self.sample_ids[index]
        image_dir = sample_id['image_dir']
        disparity_dir = sample_id['disparity_dir']
        id = sample_id['id']

        disparity = self.read_pfm(os.path.join(disparity_dir, f'{id}.pfm'))
        img = imageio.imread(os.path.join(image_dir, f'{id}.png')).astype(np.float32)

        img = img.transpose(2, 0, 1)

        # 计算统一的随机裁剪位置
        h, w = img.shape[-2], img.shape[-1]
        ph, pw = self.crop_size


        top = np.random.randint(0, h - ph + 1)
        left = np.random.randint(0, w - pw + 1)

        # 使用相同的位置裁剪
        img_patch = self.random_crop(img, top, left)
        disparity_patch = self.random_crop(disparity, top, left)

        # 数据增强（保持对应性）
        img_patch, disparity_patch = self.augment_data(img_patch, disparity_patch)

        img_patch = self.normalize_image(img_patch)
        disparity_patch = self.normalize_depth(disparity_patch)

        img_patch = torch.from_numpy(img_patch.copy())
        disparity_patch = torch.from_numpy(disparity_patch.copy()[None, ...])

        sample = {
            'image': img_patch,
            'disparity': disparity_patch,
            'id': id
        }

        return sample



if __name__ == '__main__':
    dataset = SceneFlow(root_dir = 'F:\Dataset\FlyingThing3D_subset/', crop_size = (256, 256), augment = True, dataset = 'train')

    print(f"数据集大小: {len(dataset)}")
    sample = dataset[0]
    print(f"样本场景ID: {sample['id']}")
    print(f"图像形状: {sample['image'].shape}")
    print(f"视差图形状: {sample['disparity'].shape}")
    print(f"深度数据范围: [{sample['disparity'].min():.4f}, {sample['disparity'].max():.4f}]")
    print(f"图像数据范围 : [{sample['image'].min():.4f}, {sample['image'].max():.4f}]")

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        print(f"批量图像形状: {batch['image'].shape}")
        print(f"批量深度形状: {batch['disparity'].shape}")
        break