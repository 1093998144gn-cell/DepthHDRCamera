import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random


class SceneDataset(Dataset):
    def __init__(self, depth_dir="scene_depth", hs_dir="scene_hs",
                 min_target_height=512, max_target_height=768,
                 crop_size=(256, 256), augment=True):
        """
        Args:
            depth_dir (str): 深度数据文件夹路径
            hs_dir (str): HS数据文件夹路径
            min_target_height (int): 缩放后的目标最小高度
            max_target_height (int): 缩放后的目标最大高度
            crop_size (tuple): 裁剪目标大小 (height, width)
            depth_min (float): 深度归一化目标范围的最小值
            depth_max (float): 深度归一化目标范围的最大值
            augment (bool): 是否启用数据增强
        """
        self.depth_dir = depth_dir
        self.hs_dir = hs_dir
        self.min_target_height = min_target_height
        self.max_target_height = max_target_height
        self.crop_size = crop_size
        self.augment = augment

        # 检查高度范围是否合理
        assert min_target_height <= max_target_height, "min_target_height 必须小于等于 max_target_height"
        assert min_target_height >= crop_size[0], "min_target_height 必须大于等于 crop_size[0]"

        self.scene_ids = self._get_scene_ids()
        self.depth_range_per_scene = self._get_depth_range_per_scene()
        print(f"找到 {len(self.scene_ids)} 个有效场景")

    def _get_scene_ids(self):
        depth_files = [f for f in os.listdir(self.depth_dir) if f.endswith("_depth.npy")]
        hs_files = [f for f in os.listdir(self.hs_dir) if f.endswith("_hs.npy")]
        depth_ids = {f.split('_')[0].replace('scene', '') for f in depth_files}
        hs_ids = {f.split('_')[0].replace('scene', '') for f in hs_files}
        valid_ids = sorted(depth_ids.intersection(hs_ids))
        if not valid_ids:
            raise ValueError("未找到任何有效的场景数据对")
        return valid_ids

    def _get_depth_range_per_scene(self):
        depth_range_dict = {}
        for scene_id in self.scene_ids:
            depth_file = os.path.join(self.depth_dir, f"scene{scene_id}_depth.npy")
            depth_data = np.load(depth_file, allow_pickle=True).item()
            depth_array = depth_data['Y']
            min_val = depth_array.min()
            max_val = depth_array.max()
            depth_range_dict[scene_id] = (min_val, max_val if max_val > min_val else min_val + 1.0)
            print(f"scene{scene_id} 深度范围: [{min_val:.4f}, {max_val:.4f}]")
        return depth_range_dict

    def __len__(self):
        return len(self.scene_ids)

    def get_resize_size(self, original_size):
        """
        根据原始比例和随机目标高度计算缩放大小。

        Args:
            original_size (tuple): 原始图像大小 (height, width)

        Returns:
            tuple: 缩放后的大小 (height, width)
        """
        orig_height, orig_width = original_size
        aspect_ratio = orig_width / orig_height  # 宽高比（宽度/高度）

        # 随机选择目标高度
        target_height = random.randint(self.min_target_height, self.max_target_height)

        # 根据比例计算目标宽度
        target_width = int(target_height * aspect_ratio)

        return (target_height, target_width)

    def resize_array(self, array, size):
        """将二维或三维数组缩放到指定大小"""
        if array.ndim == 2:
            img = Image.fromarray(array)
            img_resized = img.resize(size[::-1], Image.BILINEAR)
            return np.array(img_resized)
        elif array.ndim == 3:
            channels = []
            for ch in range(array.shape[0]):
                img = Image.fromarray(array[ch])
                img_resized = img.resize(size[::-1], Image.BILINEAR)
                channels.append(np.array(img_resized))
            return np.stack(channels, axis=0)
        else:
            raise ValueError("数组维度必须为 2 或 3")

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

    def normalize_depth(self, depth_patch, scene_id):
        min_val, max_val = self.depth_range_per_scene[scene_id]
        if max_val > min_val:
            depth_patch = (depth_patch - min_val) / (max_val - min_val)
        else:
            depth_patch = np.full_like(depth_patch, 0)
        return depth_patch

    def normalize_hs(self, hs_patch, scene_id):
        scene_max = hs_patch.max()
        return hs_patch / scene_max if scene_max > 0 else hs_patch

    def __getitem__(self, idx):
        if idx >= len(self.scene_ids):
            raise IndexError("索引超出范围")

        scene_id = self.scene_ids[idx]

        depth_file = os.path.join(self.depth_dir, f"scene{scene_id}_depth.npy")
        hs_file = os.path.join(self.hs_dir, f"scene{scene_id}_hs.npy")

        depth_data = np.load(depth_file, allow_pickle=True).item()
        hs_data = np.load(hs_file, allow_pickle=True).item()

        depth_array = depth_data['Y']
        hs_channels = [str(w) for w in range(420, 701, 10)]
        hs_array = np.stack([hs_data[ch] for ch in hs_channels], axis=0)

        # 获取原始尺寸
        original_size = depth_array.shape

        # 按随机高度和原始比例计算缩放大小
        resize_size = self.get_resize_size(original_size)

        # 缩放到指定大小
        depth_resized = self.resize_array(depth_array, resize_size)
        hs_resized = self.resize_array(hs_array, resize_size)

        # 计算统一的随机裁剪位置
        h, w = depth_resized.shape[-2], depth_resized.shape[-1]
        ph, pw = self.crop_size
        top = np.random.randint(0, h - ph + 1)
        left = np.random.randint(0, w - pw + 1)

        # 使用相同的位置裁剪
        depth_patch = self.random_crop(depth_resized, top, left)
        hs_patch = self.random_crop(hs_resized, top, left)

        # 数据增强（保持对应性）
        depth_patch, hs_patch = self.augment_data(depth_patch, hs_patch)

        # 归一化
        depth_patch = self.normalize_depth(depth_patch, scene_id)
        hs_patch = self.normalize_hs(hs_patch, scene_id)

        # 转换为张量
        depth_tensor = torch.from_numpy(depth_patch).float().unsqueeze(0)  # [1, H, W]
        hs_tensor = torch.from_numpy(hs_patch).float()  # [29, H, W]

        sample = {
            'depth': depth_tensor,
            'hs': hs_tensor,
            'scene_id': scene_id
        }

        return sample


# 使用示例
if __name__ == "__main__":
    dataset = SceneDataset(
        depth_dir="C:/Dataset/scene_depth",
        hs_dir="C:/Dataset/scene_hs",
        min_target_height=512,  # 最小目标高度
        max_target_height=768,  # 最大目标高度
        crop_size=(128, 128),
        augment=True
    )

    print(f"数据集大小: {len(dataset)}")
    sample = dataset[0]
    print(f"样本场景ID: {sample['scene_id']}")
    print(f"深度数据形状: {sample['depth'].shape}")
    print(f"HS数据形状: {sample['hs'].shape}")
    print(f"深度数据范围: [{sample['depth'].min():.4f}, {sample['depth'].max():.4f}]")
    print(f"HS数据范围（场景 {sample['scene_id']}）: [{sample['hs'].min():.4f}, {sample['hs'].max():.4f}]")

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        print(f"批量深度形状: {batch['depth'].shape}")
        print(f"批量HS形状: {batch['hs'].shape}")
        break