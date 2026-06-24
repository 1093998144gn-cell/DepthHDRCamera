import torch
import torch.nn as nn
import torchvision.models as models
from scipy.io import savemat
from torchvision.models import vgg16

from utils.helper import create_target_psf


class Vgg16PerceptualLoss(nn.Module):
    def __init__(self, layers=[3, 8, 15], requires_grad=False):
        """
        使用 VGG-16 提取感知损失
        参数:
        - layers: 选择用于计算损失的 VGG-16 层索引（默认: relu1_2, relu2_2, relu3_3）
        - requires_grad: 是否需要梯度（默认 False）
        """
        super(Vgg16PerceptualLoss, self).__init__()
        vgg_pretrained = models.vgg16(pretrained=True).features
        self.selected_layers = layers
        self.model = nn.Sequential(*[vgg_pretrained[i] for i in range(max(layers) + 1)])

        if not requires_grad:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, y):
        """
        计算感知损失
        - x: 生成图像
        - y: 目标图像
        """
        loss = 0
        for i, layer in enumerate(self.model):
            x = layer(x)
            y = layer(y)
            if i in self.selected_layers:
                loss += torch.nn.functional.mse_loss(x, y)
        return loss


class SpectralAngleMapperLoss(nn.Module):
    """
        Custom loss function based on Spectral Angle Mapper (SAM) for hyperspectral image reconstruction.
        This implementation avoids computing graph-related issues found in some torchmetrics classes.

        Args:
            eps (float): A small constant to prevent division by zero.
    """
    def __init__(self, eps: float = 1e-8):
        super(SpectralAngleMapperLoss, self).__init__()
        self.eps = eps

    def forward(self, est_spectrals: torch.Tensor, target_spectrals: torch.Tensor) -> torch.Tensor:
        """
        Compute the SAM loss between estimated and ground-truth hyperspectral images.

        Args:
            est_spectrals (torch.Tensor): Predicted hyperspectral tensor of shape [B, C, H, W].
            target_spectrals (torch.Tensor): Ground-truth hyperspectral tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: The mean SAM loss across all pixels.
        """
        B, C, H, W = est_spectrals.size()

        # Flatten spatial dimensions to compute SAM pixel-wise
        est_spectrals = est_spectrals.reshape(est_spectrals.size(0), -1)
        target_spectrals = target_spectrals.reshape(target_spectrals.size(0), -1)

        # Compute the dot product and norms
        dot_product = torch.sum(est_spectrals * target_spectrals, dim=1)
        est_norm = torch.norm(est_spectrals, p=2, dim=1)
        target_norm = torch.norm(target_spectrals, p=2, dim=1)

        # Compute the SAM in radians
        sam_angle = torch.acos(
            torch.clamp(dot_product / (est_norm * target_norm + self.eps), -1 + self.eps, 1 - self.eps))

        # Return the mean SAM loss
        return torch.mean(sam_angle)



class SpectralSmoothnessLoss(nn.Module):
    def __init__(self):
        super(SpectralSmoothnessLoss, self).__init__()

    def forward(self, output):
        """
        Compute spectral smoothness loss.

        Args:
            output (torch.Tensor): Hyperspectral image tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Spectral smoothness loss.
        """
        # Spectral smoothness (difference between consecutive spectral channels)
        spectral_diff = output[:, 1:, :, :] - output[:, :-1, :, :]  # Compute spectral differences
        spectral_smoothness_loss = torch.mean(spectral_diff ** 2)  # L2 norm for smoothness

        return spectral_smoothness_loss


class VGGLoss(nn.Module):
    def __init__(self, layer_indices=None, layer_weights=None, pretrained=True, requires_grad=False, reduction='mean'):
        """VGG 感知损失计算类。

        Args:
            layer_indices (list): 指定提取特征的层索引，例如 [3, 8, 15]，默认为 [3, 8, 15, 22, 29]。
            layer_weights (list): 每层的损失权重，默认为等权重。
            pretrained (bool): 是否使用预训练权重。
            requires_grad (bool): 是否计算梯度。
            reduction (str): 损失缩减方式，可选 'mean'（平均）或 'sum'（求和）。
        """
        super(VGGLoss, self).__init__()

        # 加载 VGG16 的特征部分
        vgg = vgg16(pretrained=pretrained).features

        # 默认层索引
        if layer_indices is None:
            layer_indices = [3, 8, 15, 22, 29]  # conv1_2, conv2_2, conv3_3, conv4_3, conv5_3

        self.layer_indices = layer_indices
        self.reduction = reduction
        self.features = nn.ModuleList()

        # 按层索引切分 VGG
        current_layer = 0
        for i, layer in enumerate(vgg):
            self.features.append(layer)
            if i in layer_indices:
                current_layer += 1
            if current_layer > len(layer_indices):
                break

        # 设置层权重
        if layer_weights is None:
            self.layer_weights = [1.0] * len(layer_indices)  # 默认等权重
        else:
            if len(layer_weights) != len(layer_indices):
                raise ValueError("Length of layer_weights must match the number of layer_indices.")
            self.layer_weights = layer_weights

        # 冻结参数
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, img1, img2):
        """计算两个图像之间的 VGG 感知损失。

        Args:
            img1 (torch.Tensor): 第一张图像，形状 [batch, channels, height, width]。
            img2 (torch.Tensor): 第二张图像，形状 [batch, channels, height, width]。

        Returns:
            torch.Tensor: 感知损失值。
        """
        # 输入验证
        if img1.shape != img2.shape:
            raise ValueError("Input images must have the same shape.")

        # 处理灰度图像
        if img1.shape[1] == 1:
            img1 = img1.repeat(1, 3, 1, 1)  # [batch, 1, H, W] -> [batch, 3, H, W]
            img2 = img2.repeat(1, 3, 1, 1)

        # 提取特征
        feats1 = []
        feats2 = []
        x1, x2 = img1, img2
        for i, layer in enumerate(self.features):
            x1 = layer(x1)
            x2 = layer(x2)
            if i in self.layer_indices:
                feats1.append(x1)
                feats2.append(x2)

        # 计算感知损失
        perceptual_loss = 0
        for f1, f2, w in zip(feats1, feats2, self.layer_weights):
            layer_loss = torch.mean((f1 - f2) ** 2) if self.reduction == 'mean' else torch.sum((f1 - f2) ** 2)
            perceptual_loss += w * layer_loss

        return perceptual_loss


class PSFLoss(nn.Module):
    def __init__(self, psf_range):
        super(PSFLoss, self).__init__()
        self.psf_range = psf_range

    def forward(self, psf):
        # Get the shape of the PSF tensor
        _, _, height, width = psf.shape

        # Calculate the center of the PSF
        center_x = height // 2
        center_y = width // 2

        # Create a mask of zeros with the same shape as PSF
        mask = torch.zeros_like(psf)

        # Set the central psf_range region to 1
        start_x = max(0, center_x - self.psf_range // 2)
        end_x = min(height, center_x + self.psf_range // 2 + 1)
        start_y = max(0, center_y - self.psf_range // 2)
        end_y = min(width, center_y + self.psf_range // 2 + 1)

        mask[:, :, start_x:end_x, start_y:end_y] = 1

        # Move mask to the same device as PSF
        mask = mask.to(psf.device)
        # savemat('mask.mat', {'mask': mask.detach().cpu().numpy()})

        # Compute energy terms
        psf_energy = torch.sum(psf, dim=(2, 3))
        target_energy = torch.sum(psf * mask, dim=(2, 3))
        # savemat('target_energy.mat', {'target_energy': target_energy.detach().cpu().numpy()})
        # savemat('psf_energy.mat', {'psf_energy': psf_energy.detach().cpu().numpy()})

        # Compute and return the mean absolute difference
        return torch.mean(torch.abs(target_energy - psf_energy))



class PSFHybridLoss:
    def __init__(self, psf_size, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化PSF损失函数，包含峰值强度、方差和熵损失。

        Args:
            psf_size (int): PSF的宽度（像素）。
            device (str): 计算设备。
        """
        self.psf_size = psf_size
        self.device = device
        x = torch.arange(-psf_size // 2, psf_size // 2, dtype=torch.float32, device=device)
        y = torch.arange(-psf_size // 2, psf_size // 2, dtype=torch.float32, device=device)
        x, y = torch.meshgrid(x, y, indexing='ij')
        self.rho_squared = (x ** 2 + y ** 2).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    def __call__(self, psf):
        """
        计算PSF损失，结合峰值强度、方差、熵和一致性。

        Args:
            psf (torch.Tensor): 输入PSF，形状为[num_depths, num_wavelengths, H, W]。

        Returns:
            torch.Tensor: 标量损失值。
        """
        # 归一化PSF
        psf = psf / (psf.sum(dim=(-2, -1), keepdim=True) + 1e-8)

        # 1. 峰值强度损失
        peak_loss = -torch.max(psf.view(psf.shape[0], psf.shape[1], -1), dim=-1)[0].mean()

        # 2. 方差损失
        variance_loss = torch.sqrt((psf * self.rho_squared).sum(dim=(-2, -1))).mean()

        # 3. 熵损失
        entropy_loss = -torch.sum(psf * torch.log(psf + 1e-8), dim=(-2, -1)).mean()

        # 4. 多波段一致性损失
        center_wl_idx = psf.shape[1] // 2
        center_psf = psf[:, center_wl_idx:center_wl_idx + 1, :, :]
        consistency_loss = torch.mean((psf - center_psf) ** 2)

        # 5. target_psf损失
        target_psf = create_target_psf(psf)
        target_psf_loss = torch.mean((psf - target_psf) ** 2)

        # 综合损失
        w_peak, w_variance, w_entropy, w_consistency, w_target = 1.0, 1.0, 0.1, 1.0, 1.0
        total_loss = w_peak * peak_loss + w_variance * variance_loss + w_entropy * entropy_loss + w_consistency * consistency_loss + w_target * target_psf_loss

        return total_loss