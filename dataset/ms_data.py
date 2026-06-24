import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import glob
import torchvision.transforms as transforms
import random


class HyperspectralDataset(Dataset):
    def __init__(self, root_dir, crop_size=None, transform=None, apply_augmentation=True):
        """
        初始化高光谱数据集
        Args:
            root_dir (str): 数据集根目录，例如 'F:/Dataset/complete_ms_data'
            crop_size (list/tuple, optional): 裁剪后的大小 [H, W]，默认 None
            transform (callable, optional): 额外的用户定义变换
            apply_augmentation (bool): 是否应用数据增强，默认 True
        """
        self.root_dir = root_dir
        self.transform = transform
        self.crop_size = crop_size
        self.apply_augmentation = apply_augmentation
        self.data_list = []

        transform_list = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45)
        ]
        self.augmentation = transforms.Compose(transform_list) if apply_augmentation else None

        for scene_dir in os.listdir(root_dir):
            scene_path = os.path.join(root_dir, scene_dir)
            if not os.path.isdir(scene_path):
                continue

            data_path = os.path.join(scene_path, scene_dir)
            if not os.path.isdir(data_path):
                continue

            png_files = sorted(glob.glob(os.path.join(data_path, "*.png")))
            bmp_file = glob.glob(os.path.join(data_path, "*.bmp"))

            if len(png_files) == 31 and len(bmp_file) == 1:
                self.data_list.append({
                    'png_files': png_files,
                    'bmp_file': bmp_file[0],
                    'scene': scene_dir
                })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        png_files = data['png_files']
        bmp_file = data['bmp_file']
        scene = data['scene']

        hyperspectral_data = []
        for i, png_file in enumerate(png_files):
            img = Image.open(png_file)
            if img.mode != 'L':
                img = img.convert('L')
                if i == 0 and idx == 0:
                    print(f"警告: 场景 {scene} PNG图像模式为 {img.mode}，已转换为灰度")

            img_array = np.array(img, dtype=np.float32)
            # 16位PNG归一化
            img_array = img_array / 65535.0
            if i == 0 and idx == 0:
                raw_array = np.array(img)
                print(f"场景 {scene} 第一通道原始像素值范围: min={raw_array.min()}, max={raw_array.max()}")
                print(f"场景 {scene} 第一通道归一化后范围: min={img_array.min():.6f}, max={img_array.max():.6f}")

            img_tensor = torch.tensor(img_array)
            hyperspectral_data.append(img_tensor)

        hyperspectral_data = torch.stack(hyperspectral_data, dim=0)

        rgb_img = Image.open(bmp_file).convert('RGB')
        rgb_array = np.array(rgb_img, dtype=np.float32)
        # 24位RGB（8位每通道）
        rgb_array = rgb_array / 255.0
        if idx == 0:
            print(f"场景 {scene} RGB原始像素值范围: min={np.array(rgb_img).min()}, max={np.array(rgb_img).max()}")
            print(f"场景 {scene} RGB归一化后范围: min={rgb_array.min():.6f}, max={rgb_array.max():.6f}")

        rgb_tensor = torch.tensor(rgb_array).permute(2, 0, 1)

        if self.crop_size is not None:
            crop_h, crop_w = self.crop_size
            c, h, w = hyperspectral_data.shape
            if h < crop_h or w < crop_w:
                raise ValueError(f"图像尺寸 [{h}, {w}] 小于裁剪尺寸 [{crop_h}, {crop_w}]")
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            hyperspectral_data = hyperspectral_data[:, top:top + crop_h, left:left + crop_w]
            rgb_tensor = rgb_tensor[:, top:top + crop_h, left:left + crop_w]

        if self.apply_augmentation and self.augmentation is not None:
            seed = random.randint(0, 2 ** 32)
            torch.manual_seed(seed)
            hyperspectral_data = self.augmentation(hyperspectral_data)
            torch.manual_seed(seed)
            rgb_tensor = self.augmentation(rgb_tensor)

        if self.transform:
            hyperspectral_data = self.transform(hyperspectral_data)
            rgb_tensor = self.transform(rgb_tensor)

        return {
            'hyperspectral': hyperspectral_data,
            'rgb': rgb_tensor,
            'scene': scene
        }


def main():
    root_dir = 'F:/Dataset/complete_ms_data'
    crop_size = [256, 512]

    try:
        dataset = HyperspectralDataset(root_dir=root_dir, crop_size=crop_size)
        print(f"数据集大小: {len(dataset)} 个场景")

        if len(dataset) == 0:
            print("警告: 数据集为空，请检查根目录路径或数据格式！")
            return

        sample = dataset[0]
        hyperspectral = sample['hyperspectral']
        rgb = sample['rgb']
        scene = sample['scene']

        print(f"场景名称: {scene}")
        print(f"数据目录: {scene}/{scene}")
        print(f"高光谱数据形状: {hyperspectral.shape} [C=31, H, W]")
        print(f"RGB图像形状: {rgb.shape} [C=3, H, W]")
        print(f"高光谱数据范围: min={hyperspectral.min():.6f}, max={hyperspectral.max():.6f}")
        print(f"RGB数据范围: min={rgb.min():.6f}, max={rgb.max():.6f}")

        if hyperspectral.shape[0] == 31:
            print("高光谱通道数正确: 31")
        else:
            print(f"错误: 高光谱通道数为 {hyperspectral.shape[0]}，预期为 31")

        if rgb.shape[0] == 3:
            print("RGB通道数正确: 3")
        else:
            print(f"错误: RGB通道数为 {rgb.shape[0]}，预期为 3")

        if crop_size is not None:
            crop_h, crop_w = crop_size
            if hyperspectral.shape[1:] == (crop_h, crop_w) and rgb.shape[1:] == (crop_h, crop_w):
                print(f"随机裁剪正确: 数据裁剪为 [{crop_h}, {crop_w}]")
            else:
                print(f"错误: 裁剪尺寸为 {hyperspectral.shape[1:]}，预期为 [{crop_h}, {crop_w}]")

    except Exception as e:
        print(f"错误: 测试过程中发生异常: {str(e)}")


if __name__ == "__main__":
    main()