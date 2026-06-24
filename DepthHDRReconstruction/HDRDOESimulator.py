import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from optics.light import LightWave
from optics.optical_element import DOE
from optics.propagation import Propagation


TensorLike = Union[torch.Tensor, np.ndarray]


@dataclass
class HDRDOERunConfig:
    input_path: Optional[str] = None
    depth_path: Optional[str] = None
    input_dir: str = os.path.join(CURRENT_DIR, "input_images")
    output_dir: str = os.path.join(CURRENT_DIR, "outputs_sensor")
    device: str = "auto"
    no_noise: bool = False
    same_sensor_size: bool = True
    tile_height: int = 0
    tile_width: int = 0
    synthetic_height: int = 512
    synthetic_width: int = 512


@dataclass
class HDRDOESimulatorConfig:
    # Fixed 2x2 sub-image layout on the sensor.
    grid_shape: Tuple[int, int] = (2, 2)
    # Three representative wavelengths used by the optical pipeline.
    wavelengths: Tuple[float, float, float] = (460e-9, 550e-9, 640e-9)
    design_wavelength: float = 550e-9
    refractive_index: float = 1.53

    # DOE sampling grid.
    optical_resolution: int = 256
    optical_pitch: float = 8e-6
    sensor_pixel_pitch: float = 4e-6
    aperture_fill_ratio: float = 0.92

    # Lens focuses the scene; DOE sits behind it for coding.
    lens_focal_length: float = 28e-3
    sensor_distance: float = 28e-3
    auto_focus_sensor_distance: bool = False

    # Depth modeling.
    scene_reference_depth: float = 2.0
    depth_range: Tuple[float, float] = (1.5, 3.0)
    num_depth_bins: int = 5

    # Local branch-PSF crop size for diagnostics.
    local_psf_size: int = 65

    # Relative branch energy targets for HDR.
    branch_power_weights: Tuple[float, float, float, float] = (1.0, 0.5, 0.25, 0.125)
    # Interleaved DOE cell block size.
    interleave_block_size: int = 8
    mask_seed: int = 13
    enforce_all_branches_per_block: bool = False
    # Per-branch depth-coding strengths.
    branch_depth_strengths: Tuple[float, float, float, float] = (-0.01, -0.003, 0.003, 0.01)
    # Real hardware uses one fixed DOE; it should not change when a sample has no depth GT.
    fixed_doe_variant: str = "depth_encoded"
    # Relative offsets as fractions of the max feasible shift.
    branch_target_offsets: Tuple[Tuple[float, float], ...] = (
        (-0.30, -0.42),
        (-0.30, 0.42),
        (0.30, -0.42),
        (0.30, 0.42),
    )
    # Pure optical pre-optimization for suppressing the center zero-order leakage.
    enable_zero_order_optimization: bool = True
    zero_order_opt_steps: int = 40
    zero_order_opt_lr: float = 0.08
    zero_order_center_window: int = 9
    zero_order_target_window: int = 11
    zero_order_center_weight: float = 8.0
    zero_order_target_weight: float = 0.25
    zero_order_crosstalk_weight: float = 2.5
    zero_order_wavelength_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    zero_order_phase_reg_weight: float = 1e-4
    zero_order_residual_grid: int = 16
    zero_order_residual_phase_max: float = 0.90
    zero_order_residual_reg_weight: float = 2e-3
    zero_order_residual_smoothness_weight: float = 1e-3

    # Sensor model.
    add_shot_noise: bool = True
    clip_sensor: bool = True
    base_exposure: float = 2.5
    full_well_capacity: float = 12000.0
    read_noise_std: float = 2.0
    dark_current: float = 0.0
    sensor_guard_band: int = -1
    sensor_guard_band_margin: int = 4

    # Sensor spectral response curve.
    use_sensor_response: bool = True
    sensor_response_path: str = os.path.join(PROJECT_ROOT, "optics", "response420660.pt")

    save_debug_maps: bool = False
    export_fabrication_heightmap_png: bool = True
    fabrication_target_aperture_diameter_mm: float = 0.0
    fabrication_pixel_pitch: float = 0.0
    doe_substrate_thickness_m: float = 0.3e-6

    def validate(self) -> None:
        self.grid_shape = tuple(self.grid_shape)
        self.wavelengths = tuple(self.wavelengths)
        self.depth_range = tuple(self.depth_range)
        self.scene_reference_depth = float(self.scene_reference_depth)
        self.branch_power_weights = tuple(self.branch_power_weights)
        self.branch_depth_strengths = tuple(self.branch_depth_strengths)
        self.fixed_doe_variant = str(self.fixed_doe_variant)
        self.branch_target_offsets = tuple(tuple(offset) for offset in self.branch_target_offsets)
        self.zero_order_wavelength_weights = tuple(self.zero_order_wavelength_weights)

        if self.grid_shape != (2, 2):
            raise ValueError("This simulator currently supports a 2x2 sensor mosaic only.")
        if self.optical_resolution <= 0:
            raise ValueError("optical_resolution must be positive.")
        if self.optical_pitch <= 0:
            raise ValueError("optical_pitch must be positive.")
        if self.sensor_pixel_pitch <= 0:
            raise ValueError("sensor_pixel_pitch must be positive.")
        if self.lens_focal_length <= 0:
            raise ValueError("lens_focal_length must be positive.")
        if self.sensor_distance <= 0:
            raise ValueError("sensor_distance must be positive.")
        if self.local_psf_size % 2 == 0:
            raise ValueError("local_psf_size must be odd.")
        if len(self.branch_power_weights) != 4:
            raise ValueError("branch_power_weights must contain 4 values.")
        if self.interleave_block_size <= 0:
            raise ValueError("interleave_block_size must be positive.")
        if len(self.branch_depth_strengths) != 4:
            raise ValueError("branch_depth_strengths must contain 4 values.")
        if self.fixed_doe_variant not in {"depth_encoded", "no_depth"}:
            raise ValueError("fixed_doe_variant must be 'depth_encoded' or 'no_depth'.")
        if len(self.branch_target_offsets) != 4:
            raise ValueError("branch_target_offsets must contain 4 (y, x) pairs.")
        if self.zero_order_opt_steps < 0:
            raise ValueError("zero_order_opt_steps must be non-negative.")
        if self.zero_order_opt_lr <= 0:
            raise ValueError("zero_order_opt_lr must be positive.")
        if self.zero_order_center_window <= 0 or self.zero_order_center_window % 2 == 0:
            raise ValueError("zero_order_center_window must be a positive odd integer.")
        if self.zero_order_target_window <= 0 or self.zero_order_target_window % 2 == 0:
            raise ValueError("zero_order_target_window must be a positive odd integer.")
        if len(self.zero_order_wavelength_weights) != len(self.wavelengths):
            raise ValueError("zero_order_wavelength_weights must match the number of wavelengths.")
        if any(weight < 0 for weight in self.zero_order_wavelength_weights):
            raise ValueError("zero_order_wavelength_weights must be non-negative.")
        if sum(self.zero_order_wavelength_weights) <= 0:
            raise ValueError("zero_order_wavelength_weights must have a positive sum.")
        if self.zero_order_residual_grid <= 0:
            raise ValueError("zero_order_residual_grid must be positive.")
        if self.zero_order_residual_phase_max <= 0:
            raise ValueError("zero_order_residual_phase_max must be positive.")
        if self.num_depth_bins <= 0:
            raise ValueError("num_depth_bins must be positive.")
        if self.depth_range[0] <= 0 or self.depth_range[1] <= self.depth_range[0]:
            raise ValueError("depth_range must be a positive ascending interval.")
        if self.sensor_guard_band < -1:
            raise ValueError("sensor_guard_band must be -1 (auto) or a non-negative integer.")
        if self.sensor_guard_band_margin < 0:
            raise ValueError("sensor_guard_band_margin must be non-negative.")
        if self.fabrication_target_aperture_diameter_mm < 0:
            raise ValueError("fabrication_target_aperture_diameter_mm must be non-negative.")
        if self.fabrication_pixel_pitch < 0:
            raise ValueError("fabrication_pixel_pitch must be non-negative.")
        if self.doe_substrate_thickness_m < 0:
            raise ValueError("doe_substrate_thickness_m must be non-negative.")


@dataclass
class HDRDOEResult:
    source_scene_preview: torch.Tensor
    scene_linear: torch.Tensor
    scene_preview: torch.Tensor
    depthmap: torch.Tensor
    depth_preview: torch.Tensor
    mosaic_linear: torch.Tensor
    mosaic_preview: torch.Tensor
    mosaic_debug_preview: torch.Tensor
    global_psfs: torch.Tensor
    global_psf_preview: torch.Tensor
    branch_psfs: torch.Tensor
    branch_psf_preview: torch.Tensor
    branch_masks: torch.Tensor
    branch_cell_counts: torch.Tensor
    branch_heightmaps: torch.Tensor
    total_heightmap: torch.Tensor
    sensor_response_gains: torch.Tensor
    depth_bins: torch.Tensor
    used_depth_map: bool
    used_depth_encoding: bool
    depth_processing_info: Dict[str, Any]
    source_scene_shape: Tuple[int, int]
    tile_size: Tuple[int, int]
    active_sensor_shape: Tuple[int, int]
    sensor_shape: Tuple[int, int]
    sensor_guard_band: int


def _update_dataclass_from_dict(instance: Any, updates: Dict[str, Any]) -> Any:
    valid_keys = set(instance.__dataclass_fields__.keys())
    for key, value in updates.items():
        if key not in valid_keys:
            raise KeyError(f"Unknown config field: {key}")
        setattr(instance, key, value)
    return instance


def load_yaml_configs(config_path: str) -> Tuple[HDRDOERunConfig, HDRDOESimulatorConfig]:
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    run_config = HDRDOERunConfig()
    simulator_config = HDRDOESimulatorConfig()

    if "run" in raw:
        run_config = _update_dataclass_from_dict(run_config, raw["run"])
    if "simulator" in raw:
        simulator_config = _update_dataclass_from_dict(simulator_config, raw["simulator"])

    simulator_config.validate()
    return run_config, simulator_config


class HDRDOESimulator:
    def __init__(self, config: Optional[HDRDOESimulatorConfig] = None, device: Optional[str] = None) -> None:
        self.config = config or HDRDOESimulatorConfig()
        self.config.validate()

        resolved_device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)
        self.wavelengths = torch.tensor(self.config.wavelengths, dtype=torch.float32, device=self.device)
        self.rgb_to_wavelength_indices = self._rgb_to_wavelength_indices()
        self.zero_order_wavelength_weights = self._normalized_zero_order_wavelength_weights()
        self.depth_bins = torch.linspace(
            self.config.depth_range[0],
            self.config.depth_range[1],
            steps=self.config.num_depth_bins,
            device=self.device,
        )

        self.resolution = (self.config.optical_resolution, self.config.optical_resolution)
        self.pitch = (self.config.optical_pitch, self.config.optical_pitch)
        self.sensor_distance = self._resolve_sensor_distance()
        self._lens_modulation = self._build_lens_modulation()
        self._aperture_modulation = self._build_aperture_modulation()

        self.propagator = Propagation(mode="D_FFT", device=self.device.type)
        self.sensor_response_gains = self._load_sensor_response_gains()

        self.branch_power_weights = torch.tensor(
            self.config.branch_power_weights,
            dtype=torch.float32,
            device=self.device,
        )
        self.branch_allocation_weights = torch.sqrt(self.branch_power_weights.clamp_min(0.0))
        self.branch_area_fractions = (
            self.branch_allocation_weights / self.branch_allocation_weights.sum().clamp_min(1e-8)
        )
        self.aperture_mask = self._build_aperture_mask()
        self.branch_masks = self._build_branch_masks()
        self.branch_cell_counts = self.branch_masks.sum(dim=(-2, -1))
        zero_depth_strengths = tuple(0.0 for _ in range(self.num_branches))
        self.branch_heightmaps_no_depth = self._build_branch_heightmaps(zero_depth_strengths)
        self.branch_heightmaps_depth_encoded = self._build_branch_heightmaps(self.config.branch_depth_strengths)
        self.total_amplitude_mask = self.aperture_mask
        self.zero_order_phase_biases = torch.zeros(self.num_branches, dtype=torch.float32, device=self.device)
        self.zero_order_residual_phase = torch.zeros(self.resolution, dtype=torch.float32, device=self.device)
        self.zero_order_optimization_summary: Dict[str, Any] = {
            "enabled": False,
            "phase_biases_rad": [0.0 for _ in range(self.num_branches)],
        }
        if self.config.enable_zero_order_optimization and self.config.zero_order_opt_steps > 0:
            optimization_variants = self._zero_order_optimization_branch_heightmap_variants()
            phase_biases, residual_phase, optimization_summary = self._optimize_zero_order_phase_biases(
                optimization_variants
            )
            self.zero_order_phase_biases = phase_biases.detach()
            self.zero_order_residual_phase = residual_phase.detach()
            self.zero_order_optimization_summary = optimization_summary
            self.branch_heightmaps_no_depth = self._apply_branch_phase_corrections(
                self.branch_heightmaps_no_depth,
                self.zero_order_phase_biases,
                self.zero_order_residual_phase,
            )
            self.branch_heightmaps_depth_encoded = self._apply_branch_phase_corrections(
                self.branch_heightmaps_depth_encoded,
                self.zero_order_phase_biases,
                self.zero_order_residual_phase,
            )

        self.branch_heightmaps = self.branch_heightmaps_no_depth
        self.total_heightmap = self.branch_heightmaps.sum(dim=0)

    @property
    def num_branches(self) -> int:
        return 4

    def simulate_from_path(
        self,
        image_path: str,
        depth_path: Optional[str] = None,
        tile_size: Optional[Tuple[int, int]] = None,
    ) -> HDRDOEResult:
        scene = self.load_image(image_path)
        depth = self.load_depth(depth_path) if depth_path is not None else None
        return self.simulate(scene, depthmap=depth, tile_size=tile_size)

    def simulate(
        self,
        image: TensorLike,
        depthmap: Optional[TensorLike] = None,
        tile_size: Optional[Tuple[int, int]] = None,
    ) -> HDRDOEResult:
        scene = self._prepare_image_tensor(image)
        source_scene_shape = tuple(scene.shape[-2:])
        if tile_size is None:
            tile_size = self._default_tile_size(scene.shape[-2], scene.shape[-1])

        source_scene_preview = self.linear_to_srgb(self._normalize_for_preview(scene))
        if tuple(tile_size) != tuple(scene.shape[-2:]):
            scene_linear = F.interpolate(
                scene.unsqueeze(0),
                size=tile_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            scene_linear = scene

        scene_preview = self.linear_to_srgb(self._normalize_for_preview(scene_linear))
        depth_tensor, depth_processing_info = self._prepare_depth_tensor(depthmap, tile_size, device=self.device)
        used_depth_map = depthmap is not None
        used_depth_encoding = self._fixed_doe_uses_depth_encoding()
        effective_branch_heightmaps = self._fixed_branch_heightmaps()
        effective_total_heightmap = effective_branch_heightmaps.sum(dim=0)
        active_sensor_shape = self._sensor_shape_from_tile(tile_size)
        sensor_guard_band = self._resolved_sensor_guard_band(tile_size)
        sensor_shape = (
            active_sensor_shape[0] + 2 * sensor_guard_band,
            active_sensor_shape[1] + 2 * sensor_guard_band,
        )
        render_depth_bins = (
            torch.tensor([self.config.scene_reference_depth], dtype=torch.float32, device=self.device)
            if not used_depth_map
            else self.depth_bins
        )
        global_psfs = self.compute_global_psfs(sensor_shape, effective_total_heightmap, depth_bins=render_depth_bins)
        branch_psfs = self.compute_branch_psfs(effective_branch_heightmaps)
        mosaic = self._render_single_shot_sensor(
            scene_linear,
            depth_tensor,
            global_psfs,
            sensor_shape,
            depth_bins=render_depth_bins,
        )
        active_mosaic = self._crop_center_tensor(mosaic, active_sensor_shape)
        full_mosaic_preview = self.linear_to_srgb(mosaic)
        full_mosaic_debug_preview = self.linear_to_srgb(self._normalize_for_preview(mosaic))

        return HDRDOEResult(
            source_scene_preview=source_scene_preview,
            scene_linear=scene_linear,
            scene_preview=scene_preview,
            depthmap=depth_tensor,
            depth_preview=self._depth_preview(depth_tensor),
            mosaic_linear=mosaic,
            mosaic_preview=full_mosaic_preview,
            mosaic_debug_preview=full_mosaic_debug_preview,
            global_psfs=global_psfs,
            global_psf_preview=self._global_psf_preview(global_psfs),
            branch_psfs=branch_psfs,
            branch_psf_preview=self._branch_psf_preview(branch_psfs),
            branch_masks=self.branch_masks.detach().clone(),
            branch_cell_counts=self.branch_cell_counts.detach().clone(),
            branch_heightmaps=effective_branch_heightmaps.detach().clone(),
            total_heightmap=effective_total_heightmap.detach().clone(),
            sensor_response_gains=self._wavelength_order_to_rgb(self.sensor_response_gains).detach().clone(),
            depth_bins=render_depth_bins.detach().clone(),
            used_depth_map=used_depth_map,
            used_depth_encoding=used_depth_encoding,
            depth_processing_info=depth_processing_info,
            source_scene_shape=source_scene_shape,
            tile_size=tile_size,
            active_sensor_shape=active_sensor_shape,
            sensor_shape=sensor_shape,
            sensor_guard_band=sensor_guard_band,
        )

    def compute_global_psfs(
        self,
        sensor_shape: Tuple[int, int],
        total_heightmap: torch.Tensor,
        depth_bins: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        depths = self.depth_bins if depth_bins is None else depth_bins.to(self.device)
        all_depth_psfs = []
        for depth in depths:
            psf = self._compute_full_global_psf(depth, sensor_shape, total_heightmap)
            psf = psf / psf.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
            all_depth_psfs.append(psf)
        return torch.stack(all_depth_psfs, dim=0)

    def compute_branch_psfs(self, branch_heightmaps: torch.Tensor) -> torch.Tensor:
        psf_shape = (self.config.local_psf_size, self.config.local_psf_size)
        propagation_shape = self._branch_psf_canvas_shape(psf_shape)
        all_depth_psfs = []
        for depth in self.depth_bins:
            branch_psfs = []
            for branch_idx in range(self.num_branches):
                full_psf = self._compute_branch_global_psf(depth, branch_idx, propagation_shape, branch_heightmaps)
                branch_psfs.append(self._crop_peak_tensor(full_psf, psf_shape))
            branch_psfs = torch.stack(branch_psfs, dim=0)
            channel_energy = branch_psfs.sum(dim=(0, 2, 3), keepdim=True).clamp_min(1e-12)
            branch_psfs = branch_psfs / channel_energy
            all_depth_psfs.append(branch_psfs)
        return torch.stack(all_depth_psfs, dim=0)

    def save_result(self, result: HDRDOEResult, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        debug_preview_filenames = [
            "global_psf_log_preview.png",
            "branch_psf_log_preview.png",
            "doe_branch_mask_preview.png",
            "doe_total_heightmap_preview.png",
        ]
        obsolete_nonessential_filenames = [
            "source_scene_preview.png",
            "scene_input_preview.png",
            "scene_preview.png",
            "depth_preview.png",
            "sensor_mosaic_debug.png",
            "sensor_mosaic_active.png",
            "sensor_mosaic_active_debug.png",
            "scene_input_linear.exr",
            "global_psfs.pt",
            "branch_psfs.pt",
            "branch_heightmaps.pt",
            "total_heightmap.pt",
            "branch_masks.pt",
            "branch_cell_counts.pt",
            "sensor_response_gains.pt",
            "doe_fabrication_heightmap_current_run_16bit.png",
            "doe_fabrication_heightmap_no_depth_16bit.png",
            "doe_fabrication_heightmap_depth_encoded_16bit.png",
        ]
        obsolete_debug_preview_filenames = [
            "global_psf_preview.png",
            "branch_psf_preview.png",
            "branch_mask_preview.png",
            "branch_heightmap_preview.png",
            "total_heightmap_preview.png",
        ]
        for filename in obsolete_nonessential_filenames + obsolete_debug_preview_filenames:
            path = os.path.join(output_dir, filename)
            if os.path.exists(path):
                os.remove(path)
        if not self.config.save_debug_maps:
            for filename in debug_preview_filenames:
                path = os.path.join(output_dir, filename)
                if os.path.exists(path):
                    os.remove(path)

        active_mosaic_linear = self._crop_center_tensor(result.mosaic_linear, result.active_sensor_shape)
        self._save_tensor_image(result.mosaic_preview, os.path.join(output_dir, "sensor_mosaic.png"))

        torch.save(result.scene_linear.cpu(), os.path.join(output_dir, "scene_input_linear.pt"))
        torch.save(result.mosaic_linear.cpu(), os.path.join(output_dir, "sensor_mosaic_linear.pt"))

        fabrication_variant_heightmaps: Dict[str, torch.Tensor] = {
            "current_run": result.total_heightmap,
            "no_depth": self.branch_heightmaps_no_depth.sum(dim=0),
        }
        if any(abs(x) > 1e-8 for x in self.config.branch_depth_strengths):
            fabrication_variant_heightmaps["depth_encoded"] = self.branch_heightmaps_depth_encoded.sum(dim=0)
        if self.config.export_fabrication_heightmap_png:
            self._save_fabrication_heightmap_exports(
                fabrication_variant_heightmaps=fabrication_variant_heightmaps,
                used_depth_encoding=result.used_depth_encoding,
                output_dir=output_dir,
            )

        total_sensor_energy = result.mosaic_linear.sum(dim=(-2, -1)).clamp_min(1e-12)
        active_sensor_energy = active_mosaic_linear.sum(dim=(-2, -1))
        outside_sensor_energy = (total_sensor_energy - active_sensor_energy).clamp_min(0.0)

        branch_stats = {
            "branch_power_weights": list(self.config.branch_power_weights),
            "branch_allocation_weights": self.branch_allocation_weights.detach().cpu().tolist(),
            "target_unit_fractions": self.branch_area_fractions.detach().cpu().tolist(),
            "actual_unit_counts": result.branch_cell_counts.detach().cpu().to(torch.int64).tolist(),
            "actual_unit_fractions": (
                result.branch_cell_counts / result.branch_cell_counts.sum().clamp_min(1e-8)
            ).detach().cpu().tolist(),
            "optical_pitch": float(self.config.optical_pitch),
            "sensor_pixel_pitch": float(self.config.sensor_pixel_pitch),
            "lens_focal_length": float(self.config.lens_focal_length),
            "sensor_distance_effective": float(self.sensor_distance),
            "auto_focus_sensor_distance": bool(self.config.auto_focus_sensor_distance),
            "fixed_doe_variant": self.config.fixed_doe_variant,
            "wavelengths_m": self.wavelengths.detach().cpu().tolist(),
            "base_exposure": float(self.config.base_exposure),
            "full_well_capacity": float(self.config.full_well_capacity),
            "read_noise_std": float(self.config.read_noise_std),
            "dark_current": float(self.config.dark_current),
            "zero_order_wavelength_weights": self.zero_order_wavelength_weights.detach().cpu().tolist(),
            "branch_channel_shifts_pixels": self._branch_channel_shifts_pixels().detach().cpu().tolist(),
            "branch_channel_residual_shifts_pixels": self._branch_channel_residual_shifts_pixels().detach().cpu().tolist(),
            "zero_order_optimization": self.zero_order_optimization_summary,
            "interleave_block_size": int(self.config.interleave_block_size),
            "mask_seed": int(self.config.mask_seed),
            "enforce_all_branches_per_block": bool(self.config.enforce_all_branches_per_block),
            "used_depth_map": bool(result.used_depth_map),
            "used_depth_encoding": bool(result.used_depth_encoding),
            "depth_processing_info": result.depth_processing_info,
            "render_depth_bins_m": result.depth_bins.detach().cpu().tolist(),
            "source_scene_shape": list(result.source_scene_shape),
            "scene_input_shape": list(result.tile_size),
            "scene_downsample_scale_y": float(result.tile_size[0] / result.source_scene_shape[0]),
            "scene_downsample_scale_x": float(result.tile_size[1] / result.source_scene_shape[1]),
            "tile_size": list(result.tile_size),
            "active_sensor_shape": list(result.active_sensor_shape),
            "sensor_shape": list(result.sensor_shape),
            "sensor_guard_band": int(result.sensor_guard_band),
            "active_sensor_origin_yx": [int(result.sensor_guard_band), int(result.sensor_guard_band)],
            "sensor_render_mode": "one_shot_coherent",
            "is_sensor_mosaic_png_true_ldr_preview": True,
            "sensor_mosaic_png_represents_full_sensor": True,
            "sensor_mosaic_active_png_saved": False,
            "active_sensor_energy_fraction_total": float(active_sensor_energy.sum().item() / total_sensor_energy.sum().item()),
            "outside_active_sensor_energy_fraction_total": float(outside_sensor_energy.sum().item() / total_sensor_energy.sum().item()),
            "active_sensor_energy_fraction_rgb": (
                active_sensor_energy / total_sensor_energy
            ).detach().cpu().tolist(),
            "outside_active_sensor_energy_fraction_rgb": (
                outside_sensor_energy / total_sensor_energy
            ).detach().cpu().tolist(),
            "scene_input_preview_png_saved": False,
            "source_scene_preview_png_saved": False,
            "global_psf_log_preview_scale": "normalize(log1p(psf * 1e3))",
            "branch_psf_log_preview_scale": "normalize(log1p(psf * 1e3))",
            "saved_debug_previews": debug_preview_filenames if self.config.save_debug_maps else [],
            "fabrication_heightmap_png_enabled": bool(self.config.export_fabrication_heightmap_png),
            "fabrication_target_aperture_diameter_mm": float(
                self._fabrication_target_aperture_diameter_m() * 1e3
            ),
            "saved_primary_outputs": [
                "sensor_mosaic.png",
                "sensor_mosaic_linear.pt",
                "scene_input_linear.pt",
                "branch_layout_stats.yaml",
                "doe_fabrication_heightmap_16bit.png",
                "doe_fabrication_metadata.yaml",
            ],
            "resolution_budget_report": self._resolution_budget_report(result.tile_size),
        }
        with open(os.path.join(output_dir, "branch_layout_stats.yaml"), "w", encoding="utf-8") as handle:
            yaml.safe_dump(branch_stats, handle, sort_keys=False, allow_unicode=True)

        if self.config.save_debug_maps:
            self._save_tensor_image(result.global_psf_preview, os.path.join(output_dir, "global_psf_log_preview.png"))
            self._save_tensor_image(result.branch_psf_preview, os.path.join(output_dir, "branch_psf_log_preview.png"))
            self._save_tensor_image(
                self._branch_mask_preview(result.branch_masks),
                os.path.join(output_dir, "doe_branch_mask_preview.png"),
            )
            self._save_tensor_image(
                self._normalize_for_preview(result.total_heightmap).unsqueeze(0).repeat(3, 1, 1),
                os.path.join(output_dir, "doe_total_heightmap_preview.png"),
            )

    def load_image(self, image_path: str) -> torch.Tensor:
        suffix = os.path.splitext(image_path)[1].lower()
        if suffix == ".exr":
            return self._load_exr_image(image_path)
        return self._load_standard_image(image_path)

    def load_depth(self, depth_path: str) -> torch.Tensor:
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

    def _compute_full_global_psf(
        self,
        depth: torch.Tensor,
        sensor_shape: Tuple[int, int],
        total_heightmap_tensor: torch.Tensor,
    ) -> torch.Tensor:
        propagation_shape = self._propagation_canvas_shape(sensor_shape)
        light = LightWave(propagation_shape, self.pitch, self.wavelengths, device=self.device.type)
        input_wave = self._init_spherical_wave(propagation_shape, depth.view(1))

        lens_modulation = self._embed_center_tensor(self._lens_modulation, propagation_shape, fill_value=1.0 + 0.0j)
        aperture_modulation = self._embed_center_tensor(
            self._aperture_modulation,
            propagation_shape,
            fill_value=0.0 + 0.0j,
        )
        amplitude_mask = self._embed_center_tensor(self.total_amplitude_mask, propagation_shape, fill_value=0.0)
        total_heightmap = self._embed_center_tensor(total_heightmap_tensor, propagation_shape, fill_value=0.0)

        total_doe = DOE(
            propagation_shape,
            self.pitch,
            self.wavelengths,
            heightmap=total_heightmap,
            material_refractive_index=self.config.refractive_index,
        )
        total_doe.device = self.device.type

        field = input_wave * lens_modulation
        field = total_doe.forward(field)
        field = field * amplitude_mask.view(1, 1, *propagation_shape)
        field = field * aperture_modulation
        light.set_complex(field)

        sensor_field = self.propagator.forward(light, self.sensor_distance)
        irradiance = torch.abs(sensor_field.squeeze(0)).float().square()
        return self._resample_to_sensor_grid(irradiance, sensor_shape)

    def _compute_branch_global_psf(
        self,
        depth: torch.Tensor,
        branch_idx: int,
        sensor_shape: Tuple[int, int],
        branch_heightmaps: torch.Tensor,
    ) -> torch.Tensor:
        propagation_shape = self._propagation_canvas_shape(sensor_shape)
        light = LightWave(propagation_shape, self.pitch, self.wavelengths, device=self.device.type)
        input_wave = self._init_spherical_wave(propagation_shape, depth.view(1))

        lens_modulation = self._embed_center_tensor(self._lens_modulation, propagation_shape, fill_value=1.0 + 0.0j)
        aperture_modulation = self._embed_center_tensor(
            self._aperture_modulation,
            propagation_shape,
            fill_value=0.0 + 0.0j,
        )
        amplitude_mask = self._embed_center_tensor(self.branch_masks[branch_idx], propagation_shape, fill_value=0.0)
        heightmap = self._embed_center_tensor(branch_heightmaps[branch_idx], propagation_shape, fill_value=0.0)

        branch_doe = DOE(
            propagation_shape,
            self.pitch,
            self.wavelengths,
            heightmap=heightmap,
            material_refractive_index=self.config.refractive_index,
        )
        branch_doe.device = self.device.type

        field = input_wave * lens_modulation
        field = branch_doe.forward(field)
        field = field * amplitude_mask.view(1, 1, *propagation_shape)
        field = field * aperture_modulation
        light.set_complex(field)

        sensor_field = self.propagator.forward(light, self.sensor_distance)
        irradiance = torch.abs(sensor_field.squeeze(0)).float().square()
        return self._resample_to_sensor_grid(irradiance, sensor_shape)

    def _render_single_shot_sensor(
        self,
        scene_linear: torch.Tensor,
        depthmap: torch.Tensor,
        global_psfs: torch.Tensor,
        sensor_shape: Tuple[int, int],
        depth_bins: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sensor_image = torch.zeros((3, sensor_shape[0], sensor_shape[1]), dtype=scene_linear.dtype, device=self.device)
        render_depth_bins = self.depth_bins if depth_bins is None else depth_bins.to(depthmap.device)
        depth_indices = self._quantize_depth(depthmap, render_depth_bins)
        channel_gains = self._wavelength_order_to_rgb(self.sensor_response_gains).view(3, 1, 1)

        for depth_idx in range(render_depth_bins.numel()):
            depth_mask = (depth_indices == depth_idx).float().unsqueeze(0)
            if depth_mask.sum() <= 0:
                continue

            masked_scene = scene_linear * depth_mask
            for rgb_channel_idx, wavelength_idx in enumerate(self.rgb_to_wavelength_indices.tolist()):
                full_response = self._full_convolution(
                    masked_scene[rgb_channel_idx],
                    global_psfs[depth_idx, wavelength_idx],
                )
                sensor_image[rgb_channel_idx] += self._crop_center(full_response, sensor_shape)

        sensor_image = sensor_image * channel_gains
        sensor_image = sensor_image * self.config.base_exposure
        sensor_image = sensor_image * self.config.full_well_capacity + self.config.dark_current

        if self.config.add_shot_noise:
            sensor_image = torch.poisson(sensor_image.clamp_min(0.0))
        if self.config.read_noise_std > 0:
            sensor_image = sensor_image + torch.randn_like(sensor_image) * self.config.read_noise_std
        if self.config.clip_sensor:
            sensor_image = sensor_image.clamp(0.0, self.config.full_well_capacity)

        return (sensor_image / self.config.full_well_capacity).clamp(0.0, 1.0)

    def _build_branch_masks(self) -> torch.Tensor:
        h, w = self.resolution
        block_size = int(self.config.interleave_block_size)
        aperture_mask = self.aperture_mask.detach().cpu() > 0.5
        masks = torch.zeros((self.num_branches, h, w), dtype=torch.float32)

        allocation_bias = torch.zeros(self.num_branches, dtype=torch.float64)
        for y0 in range(0, h, block_size):
            for x0 in range(0, w, block_size):
                y1 = min(y0 + block_size, h)
                x1 = min(x0 + block_size, w)
                block_mask = aperture_mask[y0:y1, x0:x1]
                valid_positions = torch.nonzero(block_mask, as_tuple=False)
                num_valid = int(valid_positions.shape[0])
                if num_valid == 0:
                    continue

                counts, allocation_bias = self._allocate_branch_counts(num_valid, allocation_bias)
                global_y = valid_positions[:, 0] + y0
                global_x = valid_positions[:, 1] + x0
                order = torch.argsort(self._coordinate_hash_scores(global_y, global_x))

                start = 0
                for branch_idx, count in enumerate(counts.tolist()):
                    if count <= 0:
                        continue
                    selected = valid_positions[order[start:start + count]]
                    masks[branch_idx, y0 + selected[:, 0], x0 + selected[:, 1]] = 1.0
                    start += count

        masks = masks.to(self.device)
        if not torch.allclose(masks.sum(dim=0), self.aperture_mask, atol=1e-5):
            raise RuntimeError("Interleaved branch masks must form a disjoint partition of the aperture.")
        return masks

    def _build_branch_heightmaps(
        self,
        depth_strengths: Optional[Tuple[float, float, float, float]] = None,
    ) -> torch.Tensor:
        heightmaps = []
        for branch_idx in range(self.num_branches):
            split_phase = self._branch_tilt_phase(branch_idx)
            depth_phase = self._branch_depth_phase(branch_idx, depth_strengths)
            total_phase = torch.remainder(split_phase + depth_phase, 2.0 * math.pi)
            height = total_phase * self.config.design_wavelength / (
                2.0 * math.pi * (self.config.refractive_index - 1.0)
            )
            heightmaps.append(height * self.branch_masks[branch_idx])
        return torch.stack(heightmaps, dim=0)

    def _apply_branch_phase_corrections(
        self,
        branch_heightmaps: torch.Tensor,
        phase_biases: torch.Tensor,
        residual_phase: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        phase_scale = 2.0 * math.pi * (self.config.refractive_index - 1.0) / self.config.design_wavelength
        branch_phase = branch_heightmaps * phase_scale
        correction = phase_biases.view(-1, 1, 1) * self.branch_masks
        if residual_phase is not None:
            correction = correction + residual_phase.unsqueeze(0) * self.branch_masks
        biased_phase = torch.remainder(branch_phase + correction, 2.0 * math.pi)
        return (biased_phase / phase_scale) * self.branch_masks

    def _fixed_doe_uses_depth_encoding(self) -> bool:
        return (
            self.config.fixed_doe_variant == "depth_encoded"
            and any(abs(x) > 1e-8 for x in self.config.branch_depth_strengths)
        )

    def _fixed_branch_heightmaps(self) -> torch.Tensor:
        if self._fixed_doe_uses_depth_encoding():
            return self.branch_heightmaps_depth_encoded
        return self.branch_heightmaps_no_depth

    def _zero_order_optimization_branch_heightmap_variants(self) -> List[torch.Tensor]:
        return [self._fixed_branch_heightmaps()]

    def _optimize_zero_order_phase_biases(
        self,
        branch_heightmap_variants: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        sensor_shape = self._zero_order_optimization_sensor_shape()
        target_mask, center_mask, other_mask = self._zero_order_masks(sensor_shape)

        phase_biases = torch.nn.Parameter(torch.zeros(self.num_branches, dtype=torch.float32, device=self.device))
        coarse_residual = torch.nn.Parameter(
            torch.zeros(
                (1, 1, self.config.zero_order_residual_grid, self.config.zero_order_residual_grid),
                dtype=torch.float32,
                device=self.device,
            )
        )
        optimizer = torch.optim.Adam([phase_biases, coarse_residual], lr=self.config.zero_order_opt_lr)

        with torch.no_grad():
            zero_residual = torch.zeros(self.resolution, dtype=torch.float32, device=self.device)
            initial_metrics = self._zero_order_metrics(
                branch_heightmap_variants,
                self.depth_bins,
                sensor_shape,
                target_mask,
                center_mask,
                other_mask,
                phase_biases,
                zero_residual,
            )

        for _ in range(self.config.zero_order_opt_steps):
            optimizer.zero_grad(set_to_none=True)
            residual_phase = self._residual_phase_from_coarse(coarse_residual)
            metrics = self._zero_order_metrics(
                branch_heightmap_variants,
                self.depth_bins,
                sensor_shape,
                target_mask,
                center_mask,
                other_mask,
                phase_biases,
                residual_phase,
            )
            phase_regularizer = torch.mean((phase_biases - phase_biases.mean()).square())
            residual_regularizer = torch.mean((residual_phase / self.config.zero_order_residual_phase_max).square())
            smoothness_regularizer = self._total_variation_loss(residual_phase)
            loss = (
                self.config.zero_order_center_weight * metrics["center_target_ratio"]
                + self.config.zero_order_crosstalk_weight * metrics["crosstalk_target_ratio"]
                - self.config.zero_order_target_weight * metrics["target_energy"]
                + self.config.zero_order_phase_reg_weight * phase_regularizer
                + self.config.zero_order_residual_reg_weight * residual_regularizer
                + self.config.zero_order_residual_smoothness_weight * smoothness_regularizer
            )
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                phase_biases.copy_(torch.atan2(torch.sin(phase_biases), torch.cos(phase_biases)))
                phase_biases.sub_(phase_biases.mean())

        with torch.no_grad():
            residual_phase = self._residual_phase_from_coarse(coarse_residual)
            final_metrics = self._zero_order_metrics(
                branch_heightmap_variants,
                self.depth_bins,
                sensor_shape,
                target_mask,
                center_mask,
                other_mask,
                phase_biases,
                residual_phase,
            )

        optimization_summary = {
            "enabled": True,
            "sensor_shape": list(sensor_shape),
            "depths_evaluated_m": self.depth_bins.detach().cpu().tolist(),
            "num_depths_evaluated": int(self.depth_bins.numel()),
            "num_heightmap_variants": int(len(branch_heightmap_variants)),
            "objective": "relative_center_target_ratio_plus_relative_crosstalk_target_ratio",
            "wavelength_weights": self.zero_order_wavelength_weights.detach().cpu().tolist(),
            "phase_biases_rad": phase_biases.detach().cpu().tolist(),
            "residual_phase_min_rad": float(residual_phase.min().item()),
            "residual_phase_max_rad": float(residual_phase.max().item()),
            "residual_phase_rms_rad": float(torch.sqrt(torch.mean(residual_phase.square())).item()),
            "initial_center_energy": float(initial_metrics["center_energy"].item()),
            "final_center_energy": float(final_metrics["center_energy"].item()),
            "initial_target_energy": float(initial_metrics["target_energy"].item()),
            "final_target_energy": float(final_metrics["target_energy"].item()),
            "initial_crosstalk_energy": float(initial_metrics["crosstalk_energy"].item()),
            "final_crosstalk_energy": float(final_metrics["crosstalk_energy"].item()),
            "initial_center_target_ratio": float(initial_metrics["center_target_ratio"].item()),
            "final_center_target_ratio": float(final_metrics["center_target_ratio"].item()),
            "initial_crosstalk_target_ratio": float(initial_metrics["crosstalk_target_ratio"].item()),
            "final_crosstalk_target_ratio": float(final_metrics["crosstalk_target_ratio"].item()),
            "steps": int(self.config.zero_order_opt_steps),
            "learning_rate": float(self.config.zero_order_opt_lr),
        }
        return phase_biases.detach(), residual_phase.detach(), optimization_summary

    def _zero_order_metrics(
        self,
        branch_heightmap_variants: Sequence[torch.Tensor],
        depths: torch.Tensor,
        sensor_shape: Tuple[int, int],
        target_mask: torch.Tensor,
        center_mask: torch.Tensor,
        other_mask: torch.Tensor,
        phase_biases: torch.Tensor,
        residual_phase: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if len(branch_heightmap_variants) == 0:
            raise ValueError("branch_heightmap_variants must not be empty.")

        channel_weights = self.zero_order_wavelength_weights
        center_energies = []
        target_energies = []
        crosstalk_energies = []
        center_target_ratios = []
        crosstalk_target_ratios = []
        for branch_heightmaps in branch_heightmap_variants:
            biased_heightmaps = self._apply_branch_phase_corrections(branch_heightmaps, phase_biases, residual_phase)
            total_heightmap = biased_heightmaps.sum(dim=0)
            for depth in depths:
                psf = self._compute_full_global_psf(depth, sensor_shape, total_heightmap)
                psf = psf / psf.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
                center_per_channel = (psf * center_mask).sum(dim=(-2, -1))
                target_per_channel = (psf * target_mask).sum(dim=(-2, -1))
                crosstalk_per_channel = (psf * other_mask).sum(dim=(-2, -1))
                center_energies.append((center_per_channel * channel_weights).sum())
                target_energies.append((target_per_channel * channel_weights).sum())
                crosstalk_energies.append((crosstalk_per_channel * channel_weights).sum())
                center_target_ratios.append(
                    ((center_per_channel / target_per_channel.clamp_min(1e-8)) * channel_weights).sum()
                )
                crosstalk_target_ratios.append(
                    ((crosstalk_per_channel / target_per_channel.clamp_min(1e-8)) * channel_weights).sum()
                )

        center_energy = torch.stack(center_energies).mean()
        target_energy = torch.stack(target_energies).mean()
        crosstalk_energy = torch.stack(crosstalk_energies).mean()
        center_target_ratio = torch.stack(center_target_ratios).mean()
        crosstalk_target_ratio = torch.stack(crosstalk_target_ratios).mean()
        return {
            "center_energy": center_energy,
            "target_energy": target_energy,
            "crosstalk_energy": crosstalk_energy,
            "center_target_ratio": center_target_ratio,
            "crosstalk_target_ratio": crosstalk_target_ratio,
        }

    def _zero_order_optimization_sensor_shape(self) -> Tuple[int, int]:
        shifts = self._branch_channel_shifts_pixels()
        max_shift_y = int(math.ceil(float(shifts[..., 0].abs().max().item())))
        max_shift_x = int(math.ceil(float(shifts[..., 1].abs().max().item())))
        margin = max(self.config.zero_order_center_window, self.config.zero_order_target_window) + 6
        height = max(self.resolution[0], 2 * (max_shift_y + margin) + 1)
        width = max(self.resolution[1], 2 * (max_shift_x + margin) + 1)
        return (height, width)

    def _zero_order_masks(self, sensor_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = sensor_shape
        target_mask = torch.zeros((3, height, width), dtype=torch.float32, device=self.device)
        center_mask = torch.zeros((3, height, width), dtype=torch.float32, device=self.device)
        center_row = (height - 1) / 2.0
        center_col = (width - 1) / 2.0
        shifts = self._branch_channel_shifts_pixels()

        for channel_idx in range(3):
            center_mask[channel_idx] = self._square_window_mask(
                sensor_shape,
                center_row,
                center_col,
                self.config.zero_order_center_window,
            )
            for branch_idx in range(self.num_branches):
                shift_y = float(shifts[branch_idx, channel_idx, 0].item())
                shift_x = float(shifts[branch_idx, channel_idx, 1].item())
                branch_window = self._square_window_mask(
                    sensor_shape,
                    center_row + shift_y,
                    center_col + shift_x,
                    self.config.zero_order_target_window,
                )
                target_mask[channel_idx] = torch.maximum(target_mask[channel_idx], branch_window)

        occupied_mask = torch.clamp(target_mask + center_mask, 0.0, 1.0)
        other_mask = 1.0 - occupied_mask
        return target_mask, center_mask, other_mask

    def _residual_phase_from_coarse(self, coarse_residual: torch.Tensor) -> torch.Tensor:
        residual_phase = F.interpolate(
            coarse_residual,
            size=self.resolution,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        residual_phase = self.config.zero_order_residual_phase_max * torch.tanh(residual_phase)
        return residual_phase * self.aperture_mask

    def _total_variation_loss(self, tensor: torch.Tensor) -> torch.Tensor:
        grad_y = tensor[1:, :] - tensor[:-1, :]
        grad_x = tensor[:, 1:] - tensor[:, :-1]
        return grad_y.abs().mean() + grad_x.abs().mean()

    def _square_window_mask(
        self,
        shape: Tuple[int, int],
        center_y: float,
        center_x: float,
        window_size: int,
    ) -> torch.Tensor:
        height, width = shape
        mask = torch.zeros((height, width), dtype=torch.float32, device=self.device)
        half = window_size // 2
        center_y = int(round(center_y))
        center_x = int(round(center_x))
        start_y = max(0, center_y - half)
        end_y = min(height, center_y + half + 1)
        start_x = max(0, center_x - half)
        end_x = min(width, center_x + half + 1)
        mask[start_y:end_y, start_x:end_x] = 1.0
        return mask

    def _branch_tilt_phase(self, branch_idx: int) -> torch.Tensor:
        target_y_ratio, target_x_ratio = self.config.branch_target_offsets[branch_idx]
        max_shift_px = self._max_feasible_shift_pixels()

        if abs(target_x_ratio) > 1.0 or abs(target_y_ratio) > 1.0:
            print(
                f"WARNING: branch_target_offsets[{branch_idx}] exceeds feasible ratio range [-1,1]. "
                "It will be clipped."
            )

        target_x_ratio = max(-1.0, min(1.0, float(target_x_ratio)))
        target_y_ratio = max(-1.0, min(1.0, float(target_y_ratio)))

        target_shift_x_px = target_x_ratio * max_shift_px
        target_shift_y_px = target_y_ratio * max_shift_px
        sin_theta_x = target_shift_x_px * self.config.sensor_pixel_pitch / self.sensor_distance
        sin_theta_y = target_shift_y_px * self.config.sensor_pixel_pitch / self.sensor_distance

        x, y = self._physical_coordinates(*self.resolution, self.config.optical_pitch)
        return (2.0 * math.pi / self.config.design_wavelength) * (sin_theta_x * x + sin_theta_y * y)

    def _branch_depth_phase(
        self,
        branch_idx: int,
        depth_strengths: Optional[Tuple[float, float, float, float]] = None,
    ) -> torch.Tensor:
        strengths = self.config.branch_depth_strengths if depth_strengths is None else depth_strengths
        strength = float(strengths[branch_idx])
        if abs(strength) < 1e-8:
            return torch.zeros(self.resolution, dtype=torch.float32, device=self.device)

        x, y = self._physical_coordinates(*self.resolution, self.config.optical_pitch)
        radius_sq = x.square() + y.square()
        return strength * (-math.pi * radius_sq / (self.config.design_wavelength * self.sensor_distance))

    def _build_aperture_mask(self) -> torch.Tensor:
        x, y = self._physical_coordinates(*self.resolution, self.config.optical_pitch)
        radius_sq = x.square() + y.square()
        aperture_radius = (
            self.config.aperture_fill_ratio * self.config.optical_resolution * self.config.optical_pitch * 0.5
        )
        return (radius_sq <= aperture_radius**2).float()

    def _allocate_branch_counts(
        self,
        num_pixels: int,
        allocation_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fractions = self.branch_area_fractions.detach().cpu().to(torch.float64)
        carry = torch.zeros_like(fractions) if allocation_bias is None else allocation_bias.to(torch.float64)
        expected = fractions * float(num_pixels) + carry
        counts = torch.floor(expected).clamp_min(0.0).to(torch.int64)

        remaining = int(num_pixels - counts.sum().item())
        if remaining > 0:
            remainders = expected - counts.to(torch.float64)
            for branch_idx in torch.argsort(remainders, descending=True)[:remaining].tolist():
                counts[branch_idx] += 1

        full_block_capacity = int(self.config.interleave_block_size) ** 2
        if self.config.enforce_all_branches_per_block and num_pixels == full_block_capacity:
            positive = fractions > 0
            positive_count = int(positive.sum().item())
            if num_pixels >= positive_count:
                missing = torch.nonzero(positive & (counts == 0), as_tuple=False).flatten()
                for receiver in missing.tolist():
                    donor_priority = counts.to(torch.float64) - expected
                    donor_order = torch.argsort(donor_priority, descending=True)
                    for donor in donor_order.tolist():
                        if donor == receiver:
                            continue
                        min_allowed = 1 if positive[donor] else 0
                        if counts[donor].item() > min_allowed:
                            counts[donor] -= 1
                            counts[receiver] += 1
                            break

        if int(counts.sum().item()) != num_pixels:
            raise RuntimeError("Failed to allocate the expected number of DOE units to the 4 branches.")
        next_bias = carry + fractions * float(num_pixels) - counts.to(torch.float64)
        return counts, next_bias

    def _coordinate_hash_scores(self, y_coords: torch.Tensor, x_coords: torch.Tensor) -> torch.Tensor:
        y = y_coords.to(torch.float64) + 1.0
        x = x_coords.to(torch.float64) + 1.0
        seed = float(self.config.mask_seed)
        raw = torch.sin(y * 12.9898 + x * 78.233 + (x * y) * 0.00017 + seed * 37.719)
        return torch.remainder(raw * 43758.5453123, 1.0).float()

    def _max_feasible_shift_pixels(self) -> float:
        max_shift_px = (
            self.sensor_distance
            * self.config.design_wavelength
            / (2.0 * self.config.optical_pitch * self.config.sensor_pixel_pitch)
        )
        return float(max_shift_px * 0.95)

    def _load_sensor_response_gains(self) -> torch.Tensor:
        if not self.config.use_sensor_response or not os.path.exists(self.config.sensor_response_path):
            return torch.ones(3, device=self.device, dtype=torch.float32)

        try:
            response = torch.load(self.config.sensor_response_path, map_location=self.device, weights_only=True)
        except TypeError:
            response = torch.load(self.config.sensor_response_path, map_location=self.device)

        if not isinstance(response, torch.Tensor):
            return torch.ones(3, device=self.device, dtype=torch.float32)

        response = response.squeeze(-1).squeeze(-1).float()
        sampled_wavelengths = torch.linspace(420e-9, 660e-9, steps=response.shape[1], device=self.device)

        gains = []
        for channel_idx, wavelength in enumerate(self.config.wavelengths):
            nearest = torch.argmin(torch.abs(sampled_wavelengths - wavelength)).item()
            gains.append(response[channel_idx, nearest])

        gains = torch.stack(gains).clamp_min(0.0)
        return gains / gains.max().clamp_min(1e-8)

    def _prepare_image_tensor(self, image: TensorLike) -> torch.Tensor:
        is_integer_input = False
        if isinstance(image, torch.Tensor):
            tensor = image.detach().clone()
            is_integer_input = not torch.is_floating_point(image)
        else:
            array = np.asarray(image)
            is_integer_input = np.issubdtype(array.dtype, np.integer)
            tensor = torch.from_numpy(array)

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        elif tensor.ndim == 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        elif tensor.ndim == 3 and tensor.shape[0] == 3:
            pass
        else:
            raise ValueError("image must have shape [H,W], [H,W,3], or [3,H,W]")

        tensor = tensor.to(self.device, dtype=torch.float32)
        if is_integer_input:
            tensor = tensor / 255.0
        return tensor.clamp_min(0.0)

    def _prepare_depth_tensor(
        self,
        depthmap: Optional[TensorLike],
        tile_size: Tuple[int, int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if depthmap is None:
            depth = torch.full(tile_size, self.config.scene_reference_depth, dtype=torch.float32, device=device)
            return depth, {
                "mode": "constant_reference_depth",
                "used_input_depth": False,
                "unit_scale": 1.0,
                "mapped_to_depth_range": False,
                "depth_min": float(self.config.scene_reference_depth),
                "depth_max": float(self.config.scene_reference_depth),
            }

        if isinstance(depthmap, torch.Tensor):
            tensor = depthmap.detach().clone().float()
        else:
            tensor = torch.from_numpy(np.asarray(depthmap)).float()

        if tensor.ndim == 3:
            tensor = tensor[0] if tensor.shape[0] == 1 else tensor[..., 0]
        if tensor.ndim != 2:
            raise ValueError("depthmap must be a single-channel 2D array")

        tensor = tensor.to(device)
        finite_mask = torch.isfinite(tensor)
        positive_mask = tensor > 0
        valid_mask = finite_mask & positive_mask
        if valid_mask.sum() == 0:
            depth = torch.full(tile_size, self.config.scene_reference_depth, dtype=torch.float32, device=device)
            return depth, {
                "mode": "fallback_constant_invalid_depth",
                "used_input_depth": False,
                "unit_scale": 1.0,
                "mapped_to_depth_range": False,
                "depth_min": float(self.config.scene_reference_depth),
                "depth_max": float(self.config.scene_reference_depth),
            }

        values = tensor[valid_mask]
        p05 = torch.quantile(values, 0.05)
        p50 = torch.quantile(values, 0.50)
        p95 = torch.quantile(values, 0.95)
        p99 = torch.quantile(values, 0.99)

        unit_scale = 1.0
        unit_mode = "meters_assumed"
        if float(p50.item()) > 20.0 or float(p95.item()) > 100.0:
            unit_scale = 1e-3
            unit_mode = "millimeters_to_meters"
            tensor = tensor * unit_scale
            values = values * unit_scale
            p05 = p05 * unit_scale
            p50 = p50 * unit_scale
            p95 = p95 * unit_scale
            p99 = p99 * unit_scale

        robust_low = float(p05.item())
        robust_high = float(p95.item())
        if robust_high <= robust_low + 1e-6:
            robust_low = float(values.min().item())
            robust_high = float(values.max().item())

        fill_value = robust_high
        tensor = torch.where(valid_mask, tensor, torch.full_like(tensor, fill_value))
        tensor = tensor.clamp(robust_low, robust_high)

        low, high = self.config.depth_range
        within_ratio = ((values >= low) & (values <= high)).float().mean()
        mapped_to_depth_range = False
        if float(within_ratio.item()) >= 0.9:
            tensor = tensor.clamp(low, high)
        else:
            tensor = (tensor - robust_low) / max(robust_high - robust_low, 1e-6)
            tensor = low + tensor * (high - low)
            mapped_to_depth_range = True

        tensor = F.interpolate(
            tensor.unsqueeze(0).unsqueeze(0),
            size=tile_size,
            mode="nearest",
        ).squeeze(0).squeeze(0)
        tensor = tensor.clamp(self.config.depth_range[0], self.config.depth_range[1])
        return tensor, {
            "mode": "input_depth_map",
            "used_input_depth": True,
            "unit_scale": unit_scale,
            "unit_mode": unit_mode,
            "mapped_to_depth_range": mapped_to_depth_range,
            "raw_p05": robust_low / unit_scale,
            "raw_p50": float(p50.item() / unit_scale),
            "raw_p95": robust_high / unit_scale,
            "processed_p05_m": robust_low,
            "processed_p50_m": float(p50.item()),
            "processed_p95_m": robust_high,
            "within_config_depth_range_ratio": float(within_ratio.item()),
            "depth_min": float(tensor.min().item()),
            "depth_max": float(tensor.max().item()),
        }

    def _load_standard_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(self.device)
        return self.srgb_to_linear(image_tensor).clamp_min(0.0)

    def _load_exr_image(self, image_path: str) -> torch.Tensor:
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

        return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).to(self.device).clamp_min(0.0)

    def _default_tile_size(self, height: int, width: int) -> Tuple[int, int]:
        return height, width

    def _fit_tile_size_to_shift_budget(self, tile_size: Tuple[int, int]) -> Tuple[int, int]:
        return tile_size

    def _sensor_shape_from_tile(self, tile_size: Tuple[int, int]) -> Tuple[int, int]:
        return self.config.grid_shape[0] * tile_size[0], self.config.grid_shape[1] * tile_size[1]

    def _alignment_crop_margin_yx(self) -> Tuple[int, int]:
        residual_shifts = self._branch_channel_residual_shifts_pixels().abs()
        max_residual_y = float(residual_shifts[..., 0].max().item())
        max_residual_x = float(residual_shifts[..., 1].max().item())
        extra = float(self.config.sensor_guard_band_margin)
        return (
            int(math.ceil(max_residual_y + extra)),
            int(math.ceil(max_residual_x + extra)),
        )

    def _recommended_sensor_guard_band(self, tile_size: Tuple[int, int]) -> int:
        active_height, active_width = self._sensor_shape_from_tile(tile_size)
        design_idx = int(torch.argmin(torch.abs(self.wavelengths - self.config.design_wavelength)).item())
        design_shifts = self._branch_channel_shifts_pixels()[:, design_idx, :]
        align_margin_y, align_margin_x = self._alignment_crop_margin_yx()
        expanded_height = tile_size[0] + 2 * align_margin_y
        expanded_width = tile_size[1] + 2 * align_margin_x

        half_active_height = 0.5 * (active_height - 1)
        half_active_width = 0.5 * (active_width - 1)
        half_expanded_height = 0.5 * (expanded_height - 1)
        half_expanded_width = 0.5 * (expanded_width - 1)

        max_shift_y = float(design_shifts[:, 0].abs().max().item())
        max_shift_x = float(design_shifts[:, 1].abs().max().item())
        overflow_y = max(0.0, max_shift_y + half_expanded_height - half_active_height)
        overflow_x = max(0.0, max_shift_x + half_expanded_width - half_active_width)
        return int(math.ceil(max(overflow_y, overflow_x)))

    def _resolved_sensor_guard_band(self, tile_size: Tuple[int, int]) -> int:
        if self.config.sensor_guard_band >= 0:
            return int(self.config.sensor_guard_band)
        return self._recommended_sensor_guard_band(tile_size)

    def _branch_psf_canvas_shape(self, psf_shape: Tuple[int, int]) -> Tuple[int, int]:
        max_target_ratio = max(
            max(abs(offset[0]), abs(offset[1])) for offset in self.config.branch_target_offsets
        )
        max_shift = int(math.ceil(max_target_ratio * self._max_feasible_shift_pixels()))
        extent = max(
            self.resolution[0],
            self.resolution[1],
            psf_shape[0] + 2 * max_shift + 8,
            psf_shape[1] + 2 * max_shift + 8,
        )
        return (extent, extent)

    def _full_convolution(self, image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        out_height = image.shape[-2] + kernel.shape[-2] - 1
        out_width = image.shape[-1] + kernel.shape[-1] - 1
        image_fft = torch.fft.fft2(image, s=(out_height, out_width))
        kernel_fft = torch.fft.fft2(kernel, s=(out_height, out_width))
        return torch.fft.ifft2(image_fft * kernel_fft).real.clamp_min(0.0)

    def _crop_center(self, image: torch.Tensor, out_shape: Tuple[int, int]) -> torch.Tensor:
        out_h, out_w = out_shape
        start_y = max(0, (image.shape[-2] - out_h) // 2)
        start_x = max(0, (image.shape[-1] - out_w) // 2)
        return image[start_y:start_y + out_h, start_x:start_x + out_w]

    def _crop_center_tensor(self, tensor: torch.Tensor, out_shape: Tuple[int, int]) -> torch.Tensor:
        out_h, out_w = out_shape
        start_y = max(0, (tensor.shape[-2] - out_h) // 2)
        start_x = max(0, (tensor.shape[-1] - out_w) // 2)
        return tensor[..., start_y:start_y + out_h, start_x:start_x + out_w]

    def _crop_peak_tensor(self, tensor: torch.Tensor, out_shape: Tuple[int, int]) -> torch.Tensor:
        cropped = []
        out_h, out_w = out_shape
        for channel_idx in range(tensor.shape[0]):
            channel = tensor[channel_idx]
            peak_linear = int(torch.argmax(channel).item())
            peak_y = peak_linear // channel.shape[1]
            peak_x = peak_linear % channel.shape[1]

            start_y = peak_y - out_h // 2
            start_x = peak_x - out_w // 2
            start_y = min(max(0, start_y), max(0, channel.shape[0] - out_h))
            start_x = min(max(0, start_x), max(0, channel.shape[1] - out_w))
            crop = channel[start_y:start_y + out_h, start_x:start_x + out_w]
            cropped.append(crop)
        return torch.stack(cropped, dim=0)

    def _branch_channel_shifts_pixels(self) -> torch.Tensor:
        max_shift = self._max_feasible_shift_pixels()
        shifts = []
        wavelength_scales = self.wavelengths / self.config.design_wavelength
        for target_y_ratio, target_x_ratio in self.config.branch_target_offsets:
            base_shift = torch.tensor(
                [target_y_ratio * max_shift, target_x_ratio * max_shift],
                dtype=torch.float32,
                device=self.device,
            )
            branch_shifts = wavelength_scales.view(-1, 1) * base_shift.view(1, 2)
            shifts.append(branch_shifts)
        return torch.stack(shifts, dim=0)

    def _branch_channel_residual_shifts_pixels(self) -> torch.Tensor:
        shifts = self._branch_channel_shifts_pixels()
        design_idx = int(torch.argmin(torch.abs(self.wavelengths - self.config.design_wavelength)).item())
        reference = shifts[:, design_idx:design_idx + 1, :]
        return shifts - reference

    def _resolution_budget_report(self, tile_size: Tuple[int, int]) -> Dict[str, Any]:
        design_idx = int(torch.argmin(torch.abs(self.wavelengths - self.config.design_wavelength)).item())
        design_shifts = self._branch_channel_shifts_pixels()[:, design_idx, :]
        max_shift_pixels = float(self._max_feasible_shift_pixels())
        spacing_y = float((design_shifts[:, 0].max() - design_shifts[:, 0].min()).item())
        spacing_x = float((design_shifts[:, 1].max() - design_shifts[:, 1].min()).item())
        residual_shifts = self._branch_channel_residual_shifts_pixels().abs()
        max_residual_y = float(residual_shifts[..., 0].max().item())
        max_residual_x = float(residual_shifts[..., 1].max().item())
        alignment_margin_y, alignment_margin_x = self._alignment_crop_margin_yx()
        recommended_sensor_guard_band = self._recommended_sensor_guard_band(tile_size)
        padding_y = max_residual_y + self.config.zero_order_target_window // 2 + 2.0
        padding_x = max_residual_x + self.config.zero_order_target_window // 2 + 2.0
        recommended_tile_height = max(1, int(math.floor(spacing_y - 2.0 * padding_y)))
        recommended_tile_width = max(1, int(math.floor(spacing_x - 2.0 * padding_x)))

        sensor_distance_up_25 = float(max_shift_pixels * 1.25)
        optical_pitch_down_25 = float(max_shift_pixels / 0.75)
        sensor_pixel_pitch_down_25 = float(max_shift_pixels / 0.75)
        aperture_diameter_mm = (
            self.config.aperture_fill_ratio
            * self.config.optical_resolution
            * self.config.optical_pitch
            * 1e3
        )
        aperture_diameter_if_resolution_384_mm = (
            self.config.aperture_fill_ratio * 384 * self.config.optical_pitch * 1e3
        )
        aperture_diameter_if_resolution_512_mm = (
            self.config.aperture_fill_ratio * 512 * self.config.optical_pitch * 1e3
        )

        return {
            "max_feasible_shift_pixels": max_shift_pixels,
            "design_shift_pixels_yx": design_shifts.detach().cpu().tolist(),
            "design_center_spacing_y_pixels": spacing_y,
            "design_center_spacing_x_pixels": spacing_x,
            "max_chromatic_residual_y_pixels": max_residual_y,
            "max_chromatic_residual_x_pixels": max_residual_x,
            "recommended_alignment_crop_margin_y_pixels": alignment_margin_y,
            "recommended_alignment_crop_margin_x_pixels": alignment_margin_x,
            "recommended_sensor_guard_band_for_full_rgb_alignment": recommended_sensor_guard_band,
            "current_tile_height": int(tile_size[0]),
            "current_tile_width": int(tile_size[1]),
            "recommended_tile_height_for_cleaner_separation": recommended_tile_height,
            "recommended_tile_width_for_cleaner_separation": recommended_tile_width,
            "current_height_to_spacing_ratio": float(tile_size[0] / max(spacing_y, 1e-6)),
            "current_width_to_spacing_ratio": float(tile_size[1] / max(spacing_x, 1e-6)),
            "aperture_diameter_mm_current": aperture_diameter_mm,
            "aperture_diameter_mm_if_optical_resolution_384": aperture_diameter_if_resolution_384_mm,
            "aperture_diameter_mm_if_optical_resolution_512": aperture_diameter_if_resolution_512_mm,
            "shift_budget_pixels_if_sensor_distance_plus_25pct": sensor_distance_up_25,
            "shift_budget_pixels_if_optical_pitch_minus_25pct": optical_pitch_down_25,
            "shift_budget_pixels_if_sensor_pixel_pitch_minus_25pct": sensor_pixel_pitch_down_25,
            "parameters_that_most_directly_increase_subimage_resolution": [
                "sensor_distance (linear increase in shift budget)",
                "optical_pitch (smaller pitch increases shift budget, but may increase fabrication difficulty)",
                "sensor_pixel_pitch (smaller sensor pitch increases shift budget in pixels)",
                "optical_resolution (larger DOE diameter reduces diffraction blur, but does not directly increase shift budget)",
            ],
        }

    def _quantize_depth(self, depthmap: torch.Tensor, depth_bins: Optional[torch.Tensor] = None) -> torch.Tensor:
        bins = self.depth_bins if depth_bins is None else depth_bins.to(depthmap.device)
        if bins.numel() == 1:
            return torch.zeros_like(depthmap, dtype=torch.long)
        sorted_bins, sorted_indices = torch.sort(bins)
        mid_edges = 0.5 * (sorted_bins[1:] + sorted_bins[:-1])
        sorted_depth_indices = torch.bucketize(depthmap, mid_edges)
        return sorted_indices[sorted_depth_indices]

    def _global_psf_preview(self, global_psfs: torch.Tensor) -> torch.Tensor:
        depth_idx = global_psfs.shape[0] // 2
        preview = self._log_psf_preview(self._wavelength_order_to_rgb(global_psfs[depth_idx]))
        return self._normalize_for_preview(preview)

    def _branch_psf_preview(self, branch_psfs: torch.Tensor) -> torch.Tensor:
        depth_idx = branch_psfs.shape[0] // 2
        preview_tiles = []
        for branch_idx in range(self.num_branches):
            tile = self._log_psf_preview(self._wavelength_order_to_rgb(branch_psfs[depth_idx, branch_idx]))
            preview_tiles.append(self._normalize_for_preview(tile))
        top = torch.cat([preview_tiles[0], preview_tiles[1]], dim=-1)
        bottom = torch.cat([preview_tiles[2], preview_tiles[3]], dim=-1)
        return torch.cat([top, bottom], dim=-2)

    def _branch_mask_preview(self, branch_masks: torch.Tensor) -> torch.Tensor:
        palette = torch.tensor(
            [
                [0.95, 0.25, 0.25],
                [0.25, 0.75, 0.30],
                [0.20, 0.45, 0.90],
                [0.90, 0.75, 0.20],
            ],
            dtype=torch.float32,
            device=branch_masks.device,
        )
        tiles = []
        for branch_idx in range(self.num_branches):
            tile = branch_masks[branch_idx].unsqueeze(0) * palette[branch_idx].view(3, 1, 1)
            tiles.append(tile)
        top = torch.cat([tiles[0], tiles[1]], dim=-1)
        bottom = torch.cat([tiles[2], tiles[3]], dim=-1)
        return torch.cat([top, bottom], dim=-2)

    def _heightmap_preview(self, branch_heightmaps: torch.Tensor) -> torch.Tensor:
        preview_tiles = []
        for branch_idx in range(self.num_branches):
            tile = self._normalize_for_preview(branch_heightmaps[branch_idx]).unsqueeze(0).repeat(3, 1, 1)
            preview_tiles.append(tile)
        top = torch.cat([preview_tiles[0], preview_tiles[1]], dim=-1)
        bottom = torch.cat([preview_tiles[2], preview_tiles[3]], dim=-1)
        return torch.cat([top, bottom], dim=-2)

    def _depth_preview(self, depthmap: torch.Tensor) -> torch.Tensor:
        normalized = (depthmap - depthmap.min()) / (depthmap.max() - depthmap.min() + 1e-8)
        return normalized.unsqueeze(0).repeat(3, 1, 1)

    def _normalize_for_preview(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach()
        tensor = tensor - tensor.min()
        return (tensor / tensor.max().clamp_min(1e-8)).float()

    def _normalized_zero_order_wavelength_weights(self) -> torch.Tensor:
        weights = torch.tensor(
            self.config.zero_order_wavelength_weights,
            dtype=torch.float32,
            device=self.device,
        )
        return weights / weights.sum().clamp_min(1e-8)

    def _rgb_to_wavelength_indices(self) -> torch.Tensor:
        return torch.argsort(self.wavelengths, descending=True)

    def _wavelength_order_to_rgb(self, tensor: torch.Tensor) -> torch.Tensor:
        indices = self.rgb_to_wavelength_indices.to(tensor.device)
        return torch.index_select(tensor, 0, indices)

    def _log_psf_preview(self, psf: torch.Tensor) -> torch.Tensor:
        return torch.log1p(psf.clamp_min(0.0) * 1e3)

    def _doe_height_wrap_m(self) -> float:
        return float(self.config.design_wavelength / max(self.config.refractive_index - 1.0, 1e-8))

    def _simulated_aperture_diameter_m(self) -> float:
        return float(
            self.config.aperture_fill_ratio * self.config.optical_resolution * self.config.optical_pitch
        )

    def _fabrication_target_aperture_diameter_m(self) -> float:
        if self.config.fabrication_target_aperture_diameter_mm > 0:
            return float(self.config.fabrication_target_aperture_diameter_mm * 1e-3)
        return self._simulated_aperture_diameter_m()

    def _aperture_bounding_box(self) -> Tuple[int, int, int, int]:
        coords = torch.nonzero(self.aperture_mask > 0.5, as_tuple=False)
        if coords.numel() == 0:
            raise RuntimeError("Aperture mask is empty; cannot determine fabrication crop.")
        min_y = int(coords[:, 0].min().item())
        max_y = int(coords[:, 0].max().item()) + 1
        min_x = int(coords[:, 1].min().item())
        max_x = int(coords[:, 1].max().item()) + 1
        return min_y, max_y, min_x, max_x

    def _fabrication_crop_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        min_y, max_y, min_x, max_x = self._aperture_bounding_box()
        return tensor[..., min_y:max_y, min_x:max_x]

    def _fabrication_absolute_heightmap(self, heightmap: torch.Tensor) -> torch.Tensor:
        cropped_heightmap = self._fabrication_crop_tensor(heightmap)
        cropped_aperture_mask = self._fabrication_crop_tensor(self.aperture_mask)
        substrate_height = float(self.config.doe_substrate_thickness_m)
        inside_aperture_height = cropped_heightmap + substrate_height
        return torch.where(
            cropped_aperture_mask > 0.5,
            inside_aperture_height,
            torch.zeros_like(cropped_heightmap),
        )

    def _save_heightmap_image_16bit(self, absolute_heightmap: torch.Tensor, path: str) -> None:
        max_height = self.config.doe_substrate_thickness_m + self._doe_height_wrap_m()
        image = absolute_heightmap.detach().cpu().clamp(0.0, max_height)
        image = (image / max(max_height, 1e-12) * 65535.0).round().to(torch.uint16).numpy()
        Image.fromarray(image, mode="I;16").save(path)

    def _save_fabrication_heightmap_exports(
        self,
        fabrication_variant_heightmaps: Dict[str, torch.Tensor],
        used_depth_encoding: bool,
        output_dir: str,
    ) -> None:
        primary_variant = "depth_encoded" if used_depth_encoding and "depth_encoded" in fabrication_variant_heightmaps else "no_depth"
        wrap_height_m = self._doe_height_wrap_m()
        absolute_max_height_m = wrap_height_m + float(self.config.doe_substrate_thickness_m)
        simulated_aperture_diameter_m = self._simulated_aperture_diameter_m()
        fabrication_aperture_diameter_m = self._fabrication_target_aperture_diameter_m()
        fabrication_canvas_width_m = fabrication_aperture_diameter_m
        fabrication_pixel_pitch_m = float(self.config.optical_pitch)
        scale_factor = fabrication_aperture_diameter_m / max(simulated_aperture_diameter_m, 1e-12)
        aperture_crop = self._fabrication_crop_tensor(self.aperture_mask)
        png_shape_hw = [int(aperture_crop.shape[-2]), int(aperture_crop.shape[-1])]

        primary_filename = "doe_fabrication_heightmap_16bit.png"
        self._save_heightmap_image_16bit(
            self._fabrication_absolute_heightmap(fabrication_variant_heightmaps[primary_variant]),
            os.path.join(output_dir, primary_filename),
        )

        no_depth_map = fabrication_variant_heightmaps.get("no_depth")
        depth_encoded_map = fabrication_variant_heightmaps.get("depth_encoded")
        variants_differ = False
        if no_depth_map is not None and depth_encoded_map is not None:
            variants_differ = bool(
                not torch.allclose(no_depth_map, depth_encoded_map, atol=1e-12, rtol=0.0)
            )

        metadata = {
            "png_encoding": "uint16 grayscale, 0 outside aperture; inside aperture linear absolute height including substrate, 0..absolute_height_max_m mapped to 0..65535",
            "design_wavelength_m": float(self.config.design_wavelength),
            "refractive_index_design": float(self.config.refractive_index),
            "wrap_height_m": float(wrap_height_m),
            "wrap_height_um": float(wrap_height_m * 1e6),
            "doe_substrate_thickness_m": float(self.config.doe_substrate_thickness_m),
            "doe_substrate_thickness_um": float(self.config.doe_substrate_thickness_m * 1e6),
            "outside_aperture_gray_value": 0,
            "absolute_height_max_m": float(absolute_max_height_m),
            "absolute_height_max_um": float(absolute_max_height_m * 1e6),
            "quantization_step_m": float(absolute_max_height_m / 65535.0),
            "quantization_step_nm": float(absolute_max_height_m * 1e9 / 65535.0),
            "png_shape_hw": png_shape_hw,
            "aperture_fill_ratio": float(self.config.aperture_fill_ratio),
            "simulated_optical_pitch_um": float(self.config.optical_pitch * 1e6),
            "simulated_aperture_diameter_mm": float(simulated_aperture_diameter_m * 1e3),
            "fabrication_target_aperture_diameter_mm": float(fabrication_aperture_diameter_m * 1e3),
            "fabrication_canvas_width_mm": float(fabrication_canvas_width_m * 1e3),
            "fabrication_pixel_pitch_um": float(fabrication_pixel_pitch_m * 1e6),
            "full_simulation_canvas_width_mm": float(self.config.optical_resolution * self.config.optical_pitch * 1e3),
            "cropped_to_circular_aperture_bounding_box": True,
            "lateral_scale_factor_vs_simulation": float(scale_factor),
            "fabrication_target_differs_from_simulation": bool(
                not math.isclose(fabrication_aperture_diameter_m, simulated_aperture_diameter_m, rel_tol=0.0, abs_tol=1e-12)
            ),
            "current_run_used_depth_encoding": bool(used_depth_encoding),
            "exported_fabrication_variant": primary_variant,
            "fabrication_png_filename": primary_filename,
            "available_internal_variants": list(fabrication_variant_heightmaps.keys()),
            "depth_encoded_variant_differs_from_no_depth_variant": bool(variants_differ),
            "warning": (
                "The exported fabrication PNG is cropped to the circular DOE aperture bounding box. Pixels outside "
                "the circular aperture are forced to zero; pixels inside the aperture encode absolute DOE height "
                "including the uniform substrate thickness."
            ),
        }
        with open(os.path.join(output_dir, "doe_fabrication_metadata.yaml"), "w", encoding="utf-8") as handle:
            yaml.safe_dump(metadata, handle, sort_keys=False, allow_unicode=True)

    def _save_tensor_image(self, image: torch.Tensor, path: str) -> None:
        image = image.detach().cpu().clamp(0.0, 1.0)
        image = (image * 255.0).round().byte()
        image = image.permute(1, 2, 0).numpy()
        Image.fromarray(image).save(path)

    def _save_linear_exr(self, image: torch.Tensor, path: str) -> None:
        image = image.detach().cpu().clamp_min(0.0)
        image = image.permute(1, 2, 0).numpy().astype(np.float32, copy=False)
        image_bgr = image[..., ::-1]
        if not cv2.imwrite(path, image_bgr):
            raise ValueError(f"Failed to save EXR image: {path}")

    def _physical_coordinates(self, height: int, width: int, pixel_pitch: float) -> Tuple[torch.Tensor, torch.Tensor]:
        y_coords = (torch.arange(height, device=self.device, dtype=torch.float32) - (height - 1) / 2.0) * pixel_pitch
        x_coords = (torch.arange(width, device=self.device, dtype=torch.float32) - (width - 1) / 2.0) * pixel_pitch
        y, x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        return x, y

    def _resolve_sensor_distance(self) -> float:
        if not self.config.auto_focus_sensor_distance:
            return float(self.config.sensor_distance)

        object_distance = float(self.config.scene_reference_depth)
        focal_length = float(self.config.lens_focal_length)
        if object_distance <= focal_length:
            raise ValueError("scene_reference_depth must be greater than lens_focal_length when auto focus is enabled.")
        return 1.0 / (1.0 / focal_length - 1.0 / object_distance)

    def _build_lens_modulation(self) -> torch.Tensor:
        x, y = self._physical_coordinates(*self.resolution, self.config.optical_pitch)
        radius_sq = x.square() + y.square()
        wavelengths = self.wavelengths.view(1, -1, 1, 1)
        phase = -(2.0 * math.pi / wavelengths) * radius_sq.unsqueeze(0).unsqueeze(0) / (2.0 * self.config.lens_focal_length)
        return torch.exp(1j * phase)

    def _build_aperture_modulation(self) -> torch.Tensor:
        x, y = self._physical_coordinates(*self.resolution, self.config.optical_pitch)
        aperture_radius = (
            self.config.aperture_fill_ratio * self.config.optical_resolution * self.config.optical_pitch * 0.5
        )
        aperture_mask = (x.square() + y.square() <= aperture_radius**2).float()
        return aperture_mask.unsqueeze(0).unsqueeze(0).expand(1, self.wavelengths.shape[0], *self.resolution)

    def _init_spherical_wave(self, resolution: Tuple[int, int], depths: torch.Tensor) -> torch.Tensor:
        x, y = self._physical_coordinates(resolution[0], resolution[1], self.config.optical_pitch)
        x = x.unsqueeze(0).unsqueeze(0)
        y = y.unsqueeze(0).unsqueeze(0)
        k = (2.0 * math.pi / self.wavelengths).view(1, self.wavelengths.shape[0], 1, 1)
        depths = depths.view(-1, 1, 1, 1).to(self.device)
        phase = k * (x.square() + y.square()) / (2.0 * depths)
        return torch.exp(1j * phase)

    def _centered_index_coordinates(self, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        y_coords = torch.arange(height, device=self.device, dtype=torch.float32) - (height - 1) / 2.0
        x_coords = torch.arange(width, device=self.device, dtype=torch.float32) - (width - 1) / 2.0
        y, x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        return y, x

    def _embed_center_tensor(
        self,
        tensor: torch.Tensor,
        target_shape: Tuple[int, int],
        fill_value: Union[float, complex],
    ) -> torch.Tensor:
        if tensor.dim() == 2:
            canvas = tensor.new_full(target_shape, fill_value)
            source_h, source_w = tensor.shape
            target_h, target_w = target_shape
            start_y = (target_h - source_h) // 2
            start_x = (target_w - source_w) // 2
            canvas[start_y:start_y + source_h, start_x:start_x + source_w] = tensor
            return canvas

        if tensor.dim() == 3:
            canvas = tensor.new_full((tensor.shape[0], target_shape[0], target_shape[1]), fill_value)
            source_h, source_w = tensor.shape[-2:]
            target_h, target_w = target_shape
            start_y = (target_h - source_h) // 2
            start_x = (target_w - source_w) // 2
            canvas[:, start_y:start_y + source_h, start_x:start_x + source_w] = tensor
            return canvas

        if tensor.dim() == 4:
            canvas = tensor.new_full((tensor.shape[0], tensor.shape[1], target_shape[0], target_shape[1]), fill_value)
            source_h, source_w = tensor.shape[-2:]
            target_h, target_w = target_shape
            start_y = (target_h - source_h) // 2
            start_x = (target_w - source_w) // 2
            canvas[:, :, start_y:start_y + source_h, start_x:start_x + source_w] = tensor
            return canvas

        raise ValueError("Only 2D, 3D, or 4D tensors can be embedded.")

    def _propagation_canvas_shape(self, sensor_shape: Tuple[int, int]) -> Tuple[int, int]:
        sensor_height, sensor_width = sensor_shape
        required_height = int(math.ceil(sensor_height * self.config.sensor_pixel_pitch / self.config.optical_pitch))
        required_width = int(math.ceil(sensor_width * self.config.sensor_pixel_pitch / self.config.optical_pitch))
        return (
            max(self.resolution[0], required_height),
            max(self.resolution[1], required_width),
        )

    def _resample_to_sensor_grid(self, tensor: torch.Tensor, sensor_shape: Tuple[int, int]) -> torch.Tensor:
        if tensor.dim() != 3:
            raise ValueError("tensor must have shape [channels, height, width].")

        source_height, source_width = tensor.shape[-2:]
        target_height, target_width = sensor_shape
        sensor_x, sensor_y = self._physical_coordinates(
            target_height,
            target_width,
            self.config.sensor_pixel_pitch,
        )
        source_center_x = 0.5 * (source_width - 1)
        source_center_y = 0.5 * (source_height - 1)
        source_x = sensor_x / self.config.optical_pitch + source_center_x
        source_y = sensor_y / self.config.optical_pitch + source_center_y

        if source_width == 1:
            normalized_x = torch.zeros_like(source_x)
        else:
            normalized_x = 2.0 * source_x / (source_width - 1) - 1.0
        if source_height == 1:
            normalized_y = torch.zeros_like(source_y)
        else:
            normalized_y = 2.0 * source_y / (source_height - 1) - 1.0

        sampling_grid = torch.stack((normalized_x, normalized_y), dim=-1).unsqueeze(0)
        sampled = F.grid_sample(
            tensor.unsqueeze(0),
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled.squeeze(0).clamp_min(0.0)

    @staticmethod
    def srgb_to_linear(image: torch.Tensor) -> torch.Tensor:
        threshold = 0.04045
        return torch.where(image <= threshold, image / 12.92, ((image + 0.055) / 1.055).pow(2.4))

    @staticmethod
    def linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
        threshold = 0.0031308
        return torch.where(
            image <= threshold,
            12.92 * image,
            1.055 * image.clamp_min(0.0).pow(1.0 / 2.4) - 0.055,
        ).clamp(0.0, 1.0)

    @staticmethod
    def generate_synthetic_scene(
        height: int = 512,
        width: int = 512,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        y = torch.linspace(0.0, 1.0, steps=height, device=device).view(height, 1)
        x = torch.linspace(0.0, 1.0, steps=width, device=device).view(1, width)

        red = 0.18 + 0.72 * x
        green = 0.12 + 0.68 * y
        blue = 0.10 + 0.55 * (1.0 - x) * (1.0 - y)
        scene = torch.stack(
            [
                red.expand(height, width),
                green.expand(height, width),
                blue.expand(height, width),
            ],
            dim=0,
        )

        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        depth = 1.7 + 1.0 * y.expand(height, width)

        blobs = [
            (0.30 * height, 0.30 * width, 0.10 * height, torch.tensor([1.8, 1.2, 0.8], device=device), 1.8),
            (0.70 * height, 0.35 * width, 0.12 * height, torch.tensor([0.7, 1.6, 1.7], device=device), 2.4),
            (0.55 * height, 0.72 * width, 0.09 * height, torch.tensor([2.5, 2.3, 1.9], device=device), 2.9),
        ]
        for cy, cx, sigma, color, depth_val in blobs:
            blob = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2))
            scene = scene + color[:, None, None] * blob.unsqueeze(0)
            depth = depth + (depth_val - depth) * blob

        return scene.clamp_min(0.0), depth.clamp(1.5, 3.0)




