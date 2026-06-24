import copy
import os
from argparse import ArgumentParser
from collections import namedtuple
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning_fabric import seed_everything
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from HDRDOESimulator import HDRDOESimulator, HDRDOERunConfig, HDRDOESimulatorConfig, load_yaml_configs


SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".exr")
SUPPORTED_DEPTH_SUFFIXES = (".npy", ".pt", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr")


DualHeadUNetOutputs = namedtuple("DualHeadUNetOutputs", ["est_hdr", "est_depth"])
DepthHDRReconstructionOutputs = namedtuple(
    "DepthHDRReconstructionOutputs",
    [
        "sensor_tiles",
        "sensor_mosaic",
        "target_hdr",
        "est_hdr",
        "target_depth",
        "est_depth",
        "used_depth_encoding",
    ],
)


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _find_matching_depth(image_path: str) -> Optional[str]:
    image_dir = os.path.dirname(image_path)
    image_name = os.path.basename(image_path)
    stem, suffix = os.path.splitext(image_name)

    direct_candidates = [os.path.join(image_dir, stem + "_depth" + depth_suffix) for depth_suffix in SUPPORTED_DEPTH_SUFFIXES]
    depth_dir = os.path.join(image_dir, "Depth")
    depth_candidates = []
    if os.path.isdir(depth_dir):
        depth_candidates.append(os.path.join(depth_dir, image_name))
        depth_candidates.append(os.path.join(depth_dir, stem + "_depth" + suffix))
        depth_candidates.extend(os.path.join(depth_dir, stem + depth_suffix) for depth_suffix in SUPPORTED_DEPTH_SUFFIXES)
        depth_candidates.extend(os.path.join(depth_dir, stem + "_depth" + depth_suffix) for depth_suffix in SUPPORTED_DEPTH_SUFFIXES)

    for candidate in direct_candidates + depth_candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _load_standard_rgb_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    return HDRDOESimulator.srgb_to_linear(image_tensor).clamp_min(0.0)


def _load_exr_image(image_path: str) -> torch.Tensor:
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read EXR image: {image_path}")

    image = image.astype(np.float32, copy=False)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim == 3 and image.shape[2] >= 3:
        image = image[..., :3][..., ::-1]
    else:
        raise ValueError(f"Unsupported EXR image shape: {image.shape}")

    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).clamp_min(0.0)


def _load_depth_map(depth_path: str) -> torch.Tensor:
    suffix = os.path.splitext(depth_path)[1].lower()
    if suffix == ".npy":
        return torch.from_numpy(np.load(depth_path)).float()
    if suffix == ".pt":
        return torch.load(depth_path, map_location="cpu").float()
    if suffix == ".exr":
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Failed to read depth EXR: {depth_path}")
        if depth.ndim == 3:
            depth = depth[..., 0]
        return torch.from_numpy(depth.astype(np.float32, copy=False))

    image = Image.open(depth_path)
    depth_np = np.asarray(image, dtype=np.float32)
    if depth_np.ndim == 3:
        depth_np = depth_np[..., 0]
    return torch.from_numpy(depth_np)


class DepthHDRDataset(Dataset):
    def __init__(
        self,
        input_dir: str,
        require_depth: bool = True,
    ) -> None:
        super().__init__()
        self.samples: List[Tuple[str, Optional[str]]] = []
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

        for name in sorted(os.listdir(input_dir)):
            suffix = os.path.splitext(name)[1].lower()
            if suffix not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            image_path = os.path.join(input_dir, name)
            depth_path = _find_matching_depth(image_path)
            if require_depth and depth_path is None:
                continue
            self.samples.append((image_path, depth_path))

        if not self.samples:
            raise RuntimeError(f"No valid image/depth pairs were found under: {input_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_path, depth_path = self.samples[index]
        image_suffix = os.path.splitext(image_path)[1].lower()
        if image_suffix == ".exr":
            image = _load_exr_image(image_path)
        else:
            image = _load_standard_rgb_image(image_path)
        depth = _load_depth_map(depth_path) if depth_path is not None else None
        return {
            "image": image,
            "depth": depth,
            "image_path": image_path,
            "depth_path": depth_path,
            "name": os.path.splitext(os.path.basename(image_path))[0],
        }


def collate_depth_hdr_batch(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image": [sample["image"] for sample in samples],
        "depth": [sample["depth"] for sample in samples],
        "image_path": [sample["image_path"] for sample in samples],
        "depth_path": [sample["depth_path"] for sample in samples],
        "name": [sample["name"] for sample in samples],
    }


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_layer: nn.Module = nn.BatchNorm2d) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm_layer(out_ch),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv_block(x) + self.shortcut(x), inplace=True)


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = ResidualConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.block(x)
        return self.pool(skip), skip


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = ResidualConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class SharedEncoderDualHeadUNet(nn.Module):
    def __init__(
        self,
        in_ch: int,
        base_ch: int,
        n_layers: int,
        depth_min: float,
        depth_max: float,
    ) -> None:
        super().__init__()
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.n_layers = int(n_layers)

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )

        channels = [base_ch]
        self.encoder_blocks = nn.ModuleList()
        for _ in range(self.n_layers):
            self.encoder_blocks.append(DownsampleBlock(channels[-1], channels[-1] * 2))
            channels.append(channels[-1] * 2)

        self.bottleneck = ResidualConvBlock(channels[-1], channels[-1])

        self.hdr_decoder = nn.ModuleList()
        self.depth_decoder = nn.ModuleList()
        for layer_idx in range(self.n_layers):
            in_channels = channels[self.n_layers - layer_idx] + channels[self.n_layers - layer_idx]
            out_channels = channels[self.n_layers - layer_idx - 1]
            self.hdr_decoder.append(UpsampleBlock(in_channels, out_channels))
            self.depth_decoder.append(UpsampleBlock(in_channels, out_channels))

        self.hdr_head = nn.Conv2d(base_ch, 3, kernel_size=1)
        self.depth_head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, sensor_tiles: torch.Tensor) -> DualHeadUNetOutputs:
        x = self.input_proj(sensor_tiles)
        skips: List[torch.Tensor] = []
        for block in self.encoder_blocks:
            x, skip = block(x)
            skips.append(skip)

        latent = self.bottleneck(x)

        hdr_features = latent
        for idx, block in enumerate(self.hdr_decoder):
            hdr_features = block(hdr_features, skips[self.n_layers - 1 - idx])
        est_hdr = F.softplus(self.hdr_head(hdr_features))

        depth_features = latent
        for idx, block in enumerate(self.depth_decoder):
            depth_features = block(depth_features, skips[self.n_layers - 1 - idx])
        depth_norm = torch.sigmoid(self.depth_head(depth_features))
        est_depth = depth_norm * (self.depth_max - self.depth_min) + self.depth_min

        return DualHeadUNetOutputs(est_hdr=est_hdr, est_depth=est_depth)


class DifferentiableHDRDOEFrontEnd(nn.Module):
    def __init__(
        self,
        simulator: HDRDOESimulator,
        optimize_optics: bool = False,
        residual_grid: int = 16,
        residual_phase_max_rad: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.simulator = simulator
        self.optimize_optics = bool(optimize_optics)
        self.residual_phase_max_rad = float(
            residual_phase_max_rad if residual_phase_max_rad is not None else simulator.config.zero_order_residual_phase_max
        )

        self.register_buffer("base_branch_heightmaps_no_depth", simulator.branch_heightmaps_no_depth.detach().clone())
        self.register_buffer("base_branch_heightmaps_depth_encoded", simulator.branch_heightmaps_depth_encoded.detach().clone())

        if self.optimize_optics:
            self.branch_phase_biases = nn.Parameter(
                torch.zeros(simulator.num_branches, dtype=torch.float32, device=simulator.device)
            )
            self.coarse_residual_phase = nn.Parameter(
                torch.zeros((1, 1, residual_grid, residual_grid), dtype=torch.float32, device=simulator.device)
            )
        else:
            self.register_buffer(
                "branch_phase_biases",
                torch.zeros(simulator.num_branches, dtype=torch.float32, device=simulator.device),
            )
            self.register_buffer(
                "coarse_residual_phase",
                torch.zeros((1, 1, residual_grid, residual_grid), dtype=torch.float32, device=simulator.device),
            )

    def optics_parameters(self) -> List[nn.Parameter]:
        if not self.optimize_optics:
            return []
        return [self.branch_phase_biases, self.coarse_residual_phase]

    def optics_regularizer(self) -> torch.Tensor:
        if not self.optimize_optics:
            return torch.zeros((), dtype=torch.float32, device=self.base_branch_heightmaps_no_depth.device)
        residual_phase = self._dense_residual_phase()
        return self.branch_phase_biases.square().mean() + residual_phase.square().mean()

    def current_fabrication_variants(self) -> Dict[str, torch.Tensor]:
        variants: Dict[str, torch.Tensor] = {
            "no_depth": self._effective_branch_heightmaps(use_depth_encoding=False).sum(dim=0)
        }
        if any(abs(x) > 1e-8 for x in self.simulator.config.branch_depth_strengths):
            variants["depth_encoded"] = self._effective_branch_heightmaps(use_depth_encoding=True).sum(dim=0)
        return variants

    def export_current_doe(
        self,
        output_dir: str,
        prefer_depth_encoded: Optional[bool] = None,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        fabrication_variant_heightmaps = self.current_fabrication_variants()
        if prefer_depth_encoded is None:
            used_depth_encoding = "depth_encoded" in fabrication_variant_heightmaps
        else:
            used_depth_encoding = bool(prefer_depth_encoded and "depth_encoded" in fabrication_variant_heightmaps)
        self.simulator._save_fabrication_heightmap_exports(
            fabrication_variant_heightmaps=fabrication_variant_heightmaps,
            used_depth_encoding=used_depth_encoding,
            output_dir=output_dir,
        )

    def _dense_residual_phase(self) -> torch.Tensor:
        dense = F.interpolate(
            self.coarse_residual_phase,
            size=self.simulator.resolution,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        return self.residual_phase_max_rad * torch.tanh(dense)

    def _effective_branch_heightmaps(self, use_depth_encoding: bool) -> torch.Tensor:
        base = self.base_branch_heightmaps_depth_encoded if use_depth_encoding else self.base_branch_heightmaps_no_depth
        if not self.optimize_optics:
            return base
        return self.simulator._apply_branch_phase_corrections(
            base,
            self.branch_phase_biases,
            self._dense_residual_phase(),
        )

    def _crop_centered_window(
        self,
        image: torch.Tensor,
        center_y: float,
        center_x: float,
        out_h: int,
        out_w: int,
    ) -> torch.Tensor:
        # Differentiable sub-pixel centered crop, mirroring ValidateChromaticPreAlign so the
        # training tiles are extracted at the *actual* per-branch diffraction centers instead of
        # naive equal-quadrant splits.
        _, src_h, src_w = image.shape
        device = image.device
        dtype = image.dtype
        ys = torch.arange(out_h, device=device, dtype=dtype) - (out_h - 1) / 2.0 + center_y
        xs = torch.arange(out_w, device=device, dtype=dtype) - (out_w - 1) / 2.0 + center_x
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        norm_x = torch.zeros_like(grid_x) if src_w == 1 else (2.0 * grid_x / (src_w - 1) - 1.0)
        norm_y = torch.zeros_like(grid_y) if src_h == 1 else (2.0 * grid_y / (src_h - 1) - 1.0)
        grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)
        crop = F.grid_sample(
            image.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return crop.squeeze(0)

    def _branch_design_centers(self, sensor_shape: Tuple[int, int]) -> torch.Tensor:
        sensor_h, sensor_w = sensor_shape
        sensor_center = torch.tensor(
            [(sensor_h - 1) / 2.0, (sensor_w - 1) / 2.0],
            dtype=torch.float32,
            device=self.simulator.device,
        )
        branch_shifts = self.simulator._branch_channel_shifts_pixels()
        design_idx = int(
            torch.argmin(torch.abs(self.simulator.wavelengths - self.simulator.config.design_wavelength)).item()
        )
        return sensor_center.unsqueeze(0) + branch_shifts[:, design_idx]

    def _branch_centered_tile_stack(self, sensor_mosaic: torch.Tensor, tile_size: Tuple[int, int]) -> torch.Tensor:
        tile_h, tile_w = tile_size
        branch_centers = self._branch_design_centers(tuple(sensor_mosaic.shape[-2:]))
        tiles: List[torch.Tensor] = []
        for branch_idx in range(self.simulator.num_branches):
            center_y = float(branch_centers[branch_idx, 0].item())
            center_x = float(branch_centers[branch_idx, 1].item())
            tiles.append(self._crop_centered_window(sensor_mosaic, center_y, center_x, tile_h, tile_w))
        return torch.cat(tiles, dim=0)

    def forward_single(
        self,
        image: torch.Tensor,
        depthmap: Optional[torch.Tensor],
        tile_size: Optional[Tuple[int, int]],
    ) -> Dict[str, Any]:
        scene = self.simulator._prepare_image_tensor(image)
        if tile_size is None:
            tile_size = self.simulator._default_tile_size(scene.shape[-2], scene.shape[-1])
        tile_size = tuple(tile_size)

        if tile_size != tuple(scene.shape[-2:]):
            scene_linear = F.interpolate(
                scene.unsqueeze(0),
                size=tile_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            scene_linear = scene

        depth_tensor, _ = self.simulator._prepare_depth_tensor(depthmap, tile_size, device=self.simulator.device)
        used_depth_map = depthmap is not None
        used_depth_encoding = self.simulator._fixed_doe_uses_depth_encoding()
        branch_heightmaps = self._effective_branch_heightmaps(used_depth_encoding)
        total_heightmap = branch_heightmaps.sum(dim=0)

        active_sensor_shape = self.simulator._sensor_shape_from_tile(tile_size)
        sensor_guard_band = self.simulator._resolved_sensor_guard_band(tile_size)
        sensor_shape = (
            active_sensor_shape[0] + 2 * sensor_guard_band,
            active_sensor_shape[1] + 2 * sensor_guard_band,
        )

        render_depth_bins = (
            torch.tensor([self.simulator.config.scene_reference_depth], dtype=torch.float32, device=self.simulator.device)
            if not used_depth_map
            else self.simulator.depth_bins
        )
        global_psfs = self.simulator.compute_global_psfs(sensor_shape, total_heightmap, depth_bins=render_depth_bins)
        sensor_mosaic = self.simulator._render_single_shot_sensor(
            scene_linear,
            depth_tensor,
            global_psfs,
            sensor_shape,
            depth_bins=render_depth_bins,
        )
        sensor_tiles = self._branch_centered_tile_stack(sensor_mosaic, tile_size)

        return {
            "sensor_tiles": sensor_tiles,
            "sensor_mosaic": sensor_mosaic,
            "target_hdr": scene_linear,
            "target_depth": depth_tensor.unsqueeze(0),
            "used_depth_encoding": used_depth_encoding,
        }


class DOEHDRDepthCamera(pl.LightningModule):
    def __init__(self, hparams: Any, log_dir: Optional[str] = None) -> None:
        super().__init__()
        self.hparam = copy.deepcopy(hparams)
        self.save_hyperparameters(self.hparam)
        self.log_dir = log_dir
        self.__build_model()

    def __build_model(self) -> None:
        run_config, simulator_config = load_yaml_configs(self.hparams.simulator_config)
        if self.hparams.disable_sensor_noise_for_training:
            simulator_config.add_shot_noise = False
            simulator_config.read_noise_std = 0.0
            simulator_config.dark_current = 0.0
        simulator_config.save_debug_maps = False
        simulator_config.export_fabrication_heightmap_png = False

        resolved_device = _resolve_device(self.hparams.simulator_device)
        self.simulator = HDRDOESimulator(simulator_config, device=resolved_device)
        self.run_config = run_config
        self.tile_size = self._resolve_tile_size(run_config)
        self.front_end = DifferentiableHDRDOEFrontEnd(
            simulator=self.simulator,
            optimize_optics=self.hparams.optimize_optics,
            residual_grid=self.hparams.optics_residual_grid,
            residual_phase_max_rad=self.hparams.optics_phase_max_rad,
        )
        self.reconstruction_net = SharedEncoderDualHeadUNet(
            in_ch=self.simulator.num_branches * len(self.simulator.wavelengths),
            base_ch=self.hparams.model_base_ch,
            n_layers=self.hparams.model_layers,
            depth_min=self.simulator.config.depth_range[0],
            depth_max=self.simulator.config.depth_range[1],
        )

    def _resolve_tile_size(self, run_config: HDRDOERunConfig) -> Optional[Tuple[int, int]]:
        if self.hparams.tile_height > 0 and self.hparams.tile_width > 0:
            return (self.hparams.tile_height, self.hparams.tile_width)
        if not run_config.same_sensor_size and run_config.tile_height > 0 and run_config.tile_width > 0:
            return (run_config.tile_height, run_config.tile_width)
        return None

    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = [{"params": self.reconstruction_net.parameters(), "lr": self.hparams.cnn_lr}]
        optics_params = self.front_end.optics_parameters()
        if optics_params:
            params.append({"params": optics_params, "lr": self.hparams.optics_lr})
        return torch.optim.Adam(params, weight_decay=self.hparams.weight_decay)

    def forward(
        self,
        images: Sequence[torch.Tensor],
        depthmaps: Sequence[Optional[torch.Tensor]],
    ) -> DepthHDRReconstructionOutputs:
        sensor_tiles: List[torch.Tensor] = []
        sensor_mosaics: List[torch.Tensor] = []
        target_hdrs: List[torch.Tensor] = []
        target_depths: List[torch.Tensor] = []
        used_depth_encoding: List[bool] = []

        for image, depthmap in zip(images, depthmaps):
            sample = self.front_end.forward_single(image, depthmap, self.tile_size)
            sensor_tiles.append(sample["sensor_tiles"])
            sensor_mosaics.append(sample["sensor_mosaic"])
            target_hdrs.append(sample["target_hdr"])
            target_depths.append(sample["target_depth"])
            used_depth_encoding.append(sample["used_depth_encoding"])

        sensor_tiles_batch = torch.stack(sensor_tiles, dim=0)
        if self.hparams.sensor_log_input:
            network_input = torch.log1p(sensor_tiles_batch.clamp_min(0.0))
        else:
            network_input = sensor_tiles_batch

        network_outputs = self.reconstruction_net(network_input)

        return DepthHDRReconstructionOutputs(
            sensor_tiles=sensor_tiles_batch,
            sensor_mosaic=torch.stack(sensor_mosaics, dim=0),
            target_hdr=torch.stack(target_hdrs, dim=0),
            est_hdr=network_outputs.est_hdr,
            target_depth=torch.stack(target_depths, dim=0),
            est_depth=network_outputs.est_depth,
            used_depth_encoding=torch.tensor(used_depth_encoding, device=sensor_tiles_batch.device, dtype=torch.bool),
        )

    def _compute_loss(self, outputs: DepthHDRReconstructionOutputs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        hdr_l1 = F.l1_loss(outputs.est_hdr, outputs.target_hdr)
        hdr_log_l1 = F.l1_loss(torch.log1p(outputs.est_hdr), torch.log1p(outputs.target_hdr))
        depth_l1 = F.l1_loss(outputs.est_depth, outputs.target_depth)
        optics_reg = self.front_end.optics_regularizer()

        total_loss = (
            self.hparams.hdr_l1_weight * hdr_l1
            + self.hparams.hdr_log_l1_weight * hdr_log_l1
            + self.hparams.depth_l1_weight * depth_l1
            + self.hparams.optics_reg_weight * optics_reg
        )
        return total_loss, {
            "total_loss": total_loss,
            "hdr_l1": hdr_l1,
            "hdr_log_l1": hdr_log_l1,
            "depth_l1": depth_l1,
            "optics_reg": optics_reg,
        }

    def export_current_doe(self, output_dir: str, prefer_depth_encoded: Optional[bool] = None) -> None:
        self.front_end.export_current_doe(output_dir=output_dir, prefer_depth_encoded=prefer_depth_encoded)

    def training_step(self, samples: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch_size = len(samples["image"])
        outputs = self.forward(samples["image"], samples["depth"])
        total_loss, loss_logs = self._compute_loss(outputs)
        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log_dict(
            {f"train/{key}": val for key, val in loss_logs.items()},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        return total_loss

    def validation_step(self, samples: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch_size = len(samples["image"])
        outputs = self.forward(samples["image"], samples["depth"])
        total_loss, loss_logs = self._compute_loss(outputs)
        self.log("val_loss", total_loss, prog_bar=(batch_idx == 0), on_step=False, on_epoch=True, batch_size=batch_size)
        self.log_dict(
            {f"val/{key}": val for key, val in loss_logs.items()},
            prog_bar=(batch_idx == 0),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return total_loss

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("DepthHDRReconstruction")
        parser.add_argument("--simulator_config", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "HDRDOESimulator.yaml"))
        parser.add_argument("--simulator_device", type=str, default="auto")
        parser.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_images"))
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--val_ratio", type=float, default=0.2)
        parser.add_argument("--max_epochs", type=int, default=50)
        parser.add_argument("--cnn_lr", type=float, default=1e-4)
        parser.add_argument("--optics_lr", type=float, default=5e-5)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--model_base_ch", type=int, default=32)
        parser.add_argument("--model_layers", type=int, default=4)
        parser.add_argument("--tile_height", type=int, default=0)
        parser.add_argument("--tile_width", type=int, default=0)
        parser.add_argument("--hdr_l1_weight", type=float, default=1.0)
        parser.add_argument("--hdr_log_l1_weight", type=float, default=0.5)
        parser.add_argument("--depth_l1_weight", type=float, default=1.0)
        parser.add_argument("--optics_reg_weight", type=float, default=1e-4)
        parser.add_argument("--optimize_optics", action="store_true")
        parser.add_argument("--optics_residual_grid", type=int, default=16)
        parser.add_argument("--optics_phase_max_rad", type=float, default=0.90)
        parser.add_argument("--disable_sensor_noise_for_training", action="store_true")
        parser.add_argument("--sensor_log_input", action="store_true")
        parser.add_argument("--experiment_name", type=str, default="DOEHDRDepthReconstruction")
        parser.add_argument("--default_root_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "logs"))
        return parent_parser


def prepare_data(args: Any) -> Tuple[DataLoader, DataLoader]:
    dataset = DepthHDRDataset(input_dir=args.data_dir, require_depth=True)
    if len(dataset) == 1:
        train_dataset = dataset
        val_dataset = dataset
    else:
        val_len = max(1, int(round(len(dataset) * args.val_ratio)))
        val_len = min(val_len, len(dataset) - 1)
        train_len = len(dataset) - val_len
        train_dataset, val_dataset = random_split(
            dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(123),
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_depth_hdr_batch,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_depth_hdr_batch,
    )
    return train_dataloader, val_dataloader


def main() -> None:
    seed_everything(123)
    parser = ArgumentParser(add_help=True)
    parser = DOEHDRDepthCamera.add_model_specific_args(parser)
    args = parser.parse_args()

    resolved_device = _resolve_device(args.simulator_device)
    accelerator = "gpu" if resolved_device == "cuda" else "cpu"

    model = DOEHDRDepthCamera(args)
    train_dataloader, val_dataloader = prepare_data(args)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=args.max_epochs,
        default_root_dir=args.default_root_dir,
        log_every_n_steps=1,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
