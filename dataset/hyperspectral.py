import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.functional import conv2d
import h5py
import scipy.io as sio
import matplotlib.image as mpimg
from glob import glob
from typing import Tuple, List
from PIL import Image as PILImage

from torchvision import transforms

def file_match(filetype, root):  #用作文件夹的获取

    files = []
    pattern   = "*.%s" %filetype
    for dir,_,_ in os.walk(root):  #os.work()是获取文件夹的路径
        files.extend(glob(os.path.join(dir,pattern)))

    return files

class ICVL(Dataset):
    def __init__(self, root='D:/Data/Dataset/Hyperspectral/', filetype='mat', transform=None):
        print("Start loading data from %s" % root)
        # build image list
        self.imglist = file_match(filetype, root)
        self.transform = transform

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        data_temp = h5py.File(self.imglist[index], 'r')
        image = np.flip(np.array(data_temp['rad']), 0).astype(np.float32)
        data = image / np.max(image)
        data = torch.Tensor(data)

        if self.transform:
            data = self.transform(data)

        return {'image': data}


class CAVE(Dataset):
    """用于高光谱数据的 PyTorch Dataset 类，包含 16 位或 32 位 PNG 光谱通道和 BMP RGB 图像。"""

    def __init__(
        self,
        image_size: Tuple[int, int],
        root: str = 'F:/Dataset/CAVE/',
        filetype: str = 'png',
        is_training: bool = True,
        randcrop: bool = False,
        augment: bool = False
    ):
        super().__init__()
        print("开始从 %s 加载数据" % root)
        self.root = root
        self.filetype = filetype
        self.image_size = image_size
        self.is_training = is_training
        self.randcrop = randcrop
        self.augment = augment

        # 获取所有 PNG 文件路径
        self.file_paths = file_match(filetype, root)
        if not self.file_paths:
            raise ValueError(f"在 {root} 中未找到 {filetype} 文件")

        # 提取场景名称
        self.imglist = self._extract_scenes()
        if not self.imglist:
            raise ValueError(f"在 {root} 中未找到有效场景")

        # 定义变换
        transform_list = []
        if randcrop and is_training:
            transform_list.append(transforms.RandomCrop(image_size, pad_if_needed=True))
        else:
            transform_list.append(transforms.CenterCrop(image_size))

        if augment and is_training:
            transform_list.append(transforms.RandomHorizontalFlip())

        self.transform = transforms.Compose(transform_list) if transform_list else None

        # 验证数据集
        self.data_info = {}
        self._validate_dataset()

    def _extract_scenes(self) -> List[str]:
        scenes = set()
        for file_path in self.file_paths:
            parts = file_path.replace('\\', '/').split('/')
            if len(parts) >= 3:
                scene_name = parts[-3]
                scenes.add(scene_name)
        return sorted(list(scenes))

    def _validate_dataset(self) -> None:
        for scene in self.imglist:
            scene_folder = os.path.join(self.root, scene, scene)
            if not os.path.exists(scene_folder) or not os.path.isdir(scene_folder):
                raise FileNotFoundError(f"场景文件夹 {scene_folder} 不存在或不是目录")

            png_files = sorted([f for f in os.listdir(scene_folder) if f.endswith(f".{self.filetype}")])
            if not png_files:
                raise ValueError(f"场景 {scene} 未包含 {self.filetype} 文件")

            bmp_files = [f for f in os.listdir(scene_folder) if f.endswith(".bmp")]
            if len(bmp_files) != 1:
                raise ValueError(f"场景 {scene} 包含 {len(bmp_files)} 个 BMP 文件，预期为 1")

            self.data_info[scene] = {
                "folder": scene_folder,
                "png_files": png_files,
                "bmp_file": bmp_files[0],
                "num_channels": len(png_files)
            }

    def __len__(self) -> int:
        return len(self.imglist)

    def __getitem__(self, index: int) -> dict:
        scene = self.imglist[index]
        scene_folder = self.data_info[scene]["folder"]
        png_files = self.data_info[scene]["png_files"]

        # 加载光谱数据
        spectral_data = []
        for png_file in png_files:
            png_filename = os.path.join(scene_folder, png_file)
            if not os.path.exists(png_filename):
                raise FileNotFoundError(f"文件 {png_filename} 不存在")

            try:
                img = PILImage.open(png_filename)
                img_array = np.array(img, dtype=np.float32)

                # 处理不同图像模式
                if img.mode == 'RGBA':  # 四通道 32 位图像
                    img_array = img_array[:, :, 0]  # 提取 R 通道
                elif img.mode in ['RGB', 'RGBA']:  # 三通道或四通道
                    img_array = np.mean(img_array[:, :, :3], axis=2)  # 平均转为灰度
                elif img.mode in ['L', 'I;16', 'I', 'F']:  # 单通道（8 位、16 位或 32 位浮点）
                    if img_array.ndim != 2:
                        raise ValueError(f"文件 {png_filename} 单通道图像维度不正确，形状为 {img_array.shape}")
                else:
                    raise ValueError(f"文件 {png_filename} 图像模式 {img.mode} 不支持")

                spectral_data.append(img_array)
            except Exception as e:
                raise RuntimeError(f"无法加载 PNG 文件 {png_filename}: {e}")

        spectral_data = np.stack(spectral_data, axis=-1)

        # 全局归一化
        max_value = np.max(spectral_data)
        if max_value == 0:
            print(f"警告: 场景 {scene} 的最大值为 0，跳过归一化")
            spectral_data = np.zeros_like(spectral_data)
        else:
            spectral_data = spectral_data / max_value

        # 加载 RGB 图像
        bmp_filename = os.path.join(scene_folder, self.data_info[scene]["bmp_file"])
        if not os.path.exists(bmp_filename):
            raise FileNotFoundError(f"RGB 图像 {bmp_filename} 不存在")

        try:
            rgb_image = PILImage.open(bmp_filename).convert("RGB")
            rgb_image = np.array(rgb_image, dtype=np.float32)
            rgb_image = rgb_image / 255.0
        except Exception as e:
            raise RuntimeError(f"无法加载 BMP 文件 {bmp_filename}: {e}")

        # 转换为 PyTorch 张量
        spectral_data = torch.from_numpy(spectral_data).permute(2, 0, 1)
        rgb_image = torch.from_numpy(rgb_image).permute(2, 0, 1)

        # 一致的数据增强
        if self.transform is not None:
            combined = torch.cat([spectral_data, rgb_image], dim=0)
            combined = self.transform(combined)
            spectral_data = combined[:spectral_data.shape[0]]
            rgb_image = combined[spectral_data.shape[0]:]

        return {
            'image': spectral_data[2:27,...]
        }


class HSDB(Dataset):
    def __init__(self,  image_size: Tuple[int, int], root='F:\Dataset\HSDB/', filetype = 'mat', is_training: bool = True):
        print("Start loading data from %s" % root)
        # build image list
        self.imglist = file_match(filetype, root)

        self.transform = transforms.Compose([transforms.RandomCrop([image_size[0], image_size[1]], pad_if_needed=True),
                                             transforms.RandomHorizontalFlip()])
        self.centercrop = transforms.CenterCrop([image_size[0], image_size[1]])
        self.is_training = torch.tensor(is_training)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        data_temp = sio.loadmat(self.imglist[index])
        image = data_temp['ref'].astype(np.float32)
        image = np.transpose(image, (2,0,1))
        data = torch.Tensor(image)
        data = data[:25, ...]  # 只保留前25个波段

        if self.is_training:
            data = self.transform(data)
        else:
            data = self.centercrop(data)

        data = data / data.max()

        return {'image': data}

