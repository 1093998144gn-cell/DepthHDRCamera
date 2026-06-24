import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from HDRDOESimulator import HDRDOESimulator, HDRDOERunConfig, load_yaml_configs
from optics.light import LightWave
from optics.propagation import Propagation


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _allocate_counts(num_pixels: int, fractions: np.ndarray, carry: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    expected = fractions * float(num_pixels) + carry
    counts = np.maximum(np.floor(expected), 0).astype(np.int64)
    remaining = int(num_pixels - counts.sum())
    if remaining > 0:
        remainders = expected - counts
        order = np.argsort(-remainders)
        cursor = 0
        while remaining > 0:
            branch_idx = int(order[cursor % len(order)])
            counts[branch_idx] += 1
            remaining -= 1
            cursor += 1
    elif remaining < 0:
        surplus = -remaining
        donor_priority = counts.astype(np.float64) - expected
        order = np.argsort(-donor_priority)
        cursor = 0
        while surplus > 0:
            branch_idx = int(order[cursor % len(order)])
            cursor += 1
            if counts[branch_idx] <= 0:
                if cursor > len(order) * 2:
                    break
                continue
            counts[branch_idx] -= 1
            surplus -= 1
    if int(counts.sum()) != num_pixels:
        raise RuntimeError("Failed to allocate the expected number of high-res DOE cells.")
    next_carry = carry + fractions * float(num_pixels) - counts
    return counts, next_carry


def _coordinate_hash_scores(y_coords: np.ndarray, x_coords: np.ndarray, seed: float) -> np.ndarray:
    y = y_coords.astype(np.float64) + 1.0
    x = x_coords.astype(np.float64) + 1.0
    raw = np.sin(y * 12.9898 + x * 78.233 + (x * y) * 0.00017 + seed * 37.719)
    return np.mod(raw * 43758.5453123, 1.0).astype(np.float32)


def _target_sines(simulator: HDRDOESimulator) -> Tuple[np.ndarray, np.ndarray]:
    max_shift_px = simulator._max_feasible_shift_pixels()
    sin_y: List[float] = []
    sin_x: List[float] = []
    for target_y_ratio, target_x_ratio in simulator.config.branch_target_offsets:
        target_shift_y_px = float(target_y_ratio) * max_shift_px
        target_shift_x_px = float(target_x_ratio) * max_shift_px
        sin_y.append(target_shift_y_px * simulator.config.sensor_pixel_pitch / simulator.sensor_distance)
        sin_x.append(target_shift_x_px * simulator.config.sensor_pixel_pitch / simulator.sensor_distance)
    return np.asarray(sin_y, dtype=np.float32), np.asarray(sin_x, dtype=np.float32)


def _resolve_preview_tile_size(
    run_config: HDRDOERunConfig,
    simulator: HDRDOESimulator,
    override_height: int,
    override_width: int,
) -> Tuple[Tuple[int, int], int, str]:
    if override_height > 0 and override_width > 0:
        tile_size = (int(override_height), int(override_width))
        return tile_size, simulator._resolved_sensor_guard_band(tile_size), "cli_override"

    if not run_config.same_sensor_size and run_config.tile_height > 0 and run_config.tile_width > 0:
        tile_size = (int(run_config.tile_height), int(run_config.tile_width))
        return tile_size, simulator._resolved_sensor_guard_band(tile_size), "run_config"

    active_height = simulator.resolution[0]
    active_width = simulator.resolution[1]
    tile_size = (
        max(1, active_height // simulator.config.grid_shape[0]),
        max(1, active_width // simulator.config.grid_shape[1]),
    )
    return tile_size, 0, "fallback_from_optical_resolution"


def _physical_coordinates(
    height: int,
    width: int,
    pixel_pitch: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    y_coords = (torch.arange(height, device=device, dtype=torch.float32) - (height - 1) / 2.0) * pixel_pitch
    x_coords = (torch.arange(width, device=device, dtype=torch.float32) - (width - 1) / 2.0) * pixel_pitch
    y, x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    return x, y


def _propagation_canvas_shape(
    fabrication_resolution: Tuple[int, int],
    fabrication_pitch: float,
    sensor_shape: Tuple[int, int],
    sensor_pixel_pitch: float,
) -> Tuple[int, int]:
    required_height = int(math.ceil(sensor_shape[0] * sensor_pixel_pitch / fabrication_pitch))
    required_width = int(math.ceil(sensor_shape[1] * sensor_pixel_pitch / fabrication_pitch))
    return (
        max(fabrication_resolution[0], required_height),
        max(fabrication_resolution[1], required_width),
    )


def _resample_to_sensor_grid(
    tensor: torch.Tensor,
    sensor_shape: Tuple[int, int],
    source_pitch: float,
    sensor_pixel_pitch: float,
) -> torch.Tensor:
    if tensor.dim() != 2:
        raise ValueError("tensor must have shape [height, width].")

    device = tensor.device
    source_height, source_width = tensor.shape
    target_height, target_width = sensor_shape
    sensor_x, sensor_y = _physical_coordinates(target_height, target_width, sensor_pixel_pitch, device=device)
    source_center_x = 0.5 * (source_width - 1)
    source_center_y = 0.5 * (source_height - 1)
    source_x = sensor_x / source_pitch + source_center_x
    source_y = sensor_y / source_pitch + source_center_y

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
        tensor.unsqueeze(0).unsqueeze(0),
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.squeeze(0).squeeze(0).clamp_min(0.0)


def _export_fabrication_grid_global_psf(
    simulator: HDRDOESimulator,
    variant: str,
    output_dir: str,
    run_config: HDRDOERunConfig,
    fabrication_png_gray: np.ndarray,
    fabrication_pitch: float,
    substrate_m: float,
    absolute_height_max_m: float,
    psf_tile_height: int,
    psf_tile_width: int,
    psf_wavelength_m: Optional[float],
    psf_depth_m: Optional[float],
) -> None:
    tile_size, sensor_guard_band, tile_size_source = _resolve_preview_tile_size(
        run_config=run_config,
        simulator=simulator,
        override_height=psf_tile_height,
        override_width=psf_tile_width,
    )
    active_sensor_shape = simulator._sensor_shape_from_tile(tile_size)
    sensor_shape = (
        active_sensor_shape[0] + 2 * sensor_guard_band,
        active_sensor_shape[1] + 2 * sensor_guard_band,
    )
    wavelength_m = float(psf_wavelength_m if psf_wavelength_m is not None else simulator.config.design_wavelength)
    depth_m = float(psf_depth_m if psf_depth_m is not None else simulator.config.scene_reference_depth)

    fabrication_absolute_height = (
        torch.from_numpy(fabrication_png_gray.astype(np.float32, copy=False)).to(simulator.device)
        / 65535.0
        * absolute_height_max_m
    )
    fabrication_heightmap = torch.where(
        fabrication_absolute_height > 0.0,
        (fabrication_absolute_height - substrate_m).clamp_min(0.0),
        torch.zeros_like(fabrication_absolute_height),
    )
    fabrication_aperture_mask = (fabrication_absolute_height > 0.0).float()
    fabrication_resolution = tuple(int(x) for x in fabrication_heightmap.shape)
    propagation_shape = _propagation_canvas_shape(
        fabrication_resolution=fabrication_resolution,
        fabrication_pitch=fabrication_pitch,
        sensor_shape=sensor_shape,
        sensor_pixel_pitch=simulator.config.sensor_pixel_pitch,
    )

    wavelengths = torch.tensor([wavelength_m], dtype=torch.float32, device=simulator.device)
    depth_tensor = torch.tensor([depth_m], dtype=torch.float32, device=simulator.device)
    light = LightWave(propagation_shape, (fabrication_pitch, fabrication_pitch), wavelengths, device=simulator.device.type)
    input_wave = light.init_wave(depth_tensor, wave_type="spherical")

    lens_x, lens_y = _physical_coordinates(
        propagation_shape[0],
        propagation_shape[1],
        fabrication_pitch,
        device=simulator.device,
    )
    radius_sq = lens_x.square() + lens_y.square()
    lens_phase = -(2.0 * math.pi / wavelength_m) * radius_sq / (2.0 * simulator.config.lens_focal_length)
    lens_modulation = torch.exp(1j * lens_phase)

    heightmap_embedded = simulator._embed_center_tensor(
        fabrication_heightmap,
        propagation_shape,
        fill_value=0.0,
    )
    aperture_embedded = simulator._embed_center_tensor(
        fabrication_aperture_mask,
        propagation_shape,
        fill_value=0.0,
    )
    doe_phase = heightmap_embedded * (
        2.0 * math.pi * (simulator.config.refractive_index - 1.0) / wavelength_m
    )
    doe_modulation = torch.exp(1j * doe_phase)

    field = input_wave * lens_modulation.unsqueeze(0).unsqueeze(0)
    field = field * doe_modulation.unsqueeze(0).unsqueeze(0)
    field = field * aperture_embedded.unsqueeze(0).unsqueeze(0)
    light.set_complex(field)

    propagator = Propagation(mode="D_FFT", device=simulator.device.type)
    try:
        sensor_field = propagator.forward(light, simulator.sensor_distance)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message:
            raise RuntimeError(
                "Failed to compute the 0.4 um full-sensor PSF because the propagation ran out of memory. "
                "Try running on CPU with large system RAM, reducing the sensor tile size, or increasing the "
                "fabrication pitch for this diagnostic run."
            ) from exc
        raise

    fabrication_psf = torch.abs(sensor_field.squeeze(0).squeeze(0)).float().square()
    fabrication_psf = fabrication_psf / fabrication_psf.sum().clamp_min(1e-12)
    global_psf = _resample_to_sensor_grid(
        fabrication_psf,
        sensor_shape=sensor_shape,
        source_pitch=fabrication_pitch,
        sensor_pixel_pitch=simulator.config.sensor_pixel_pitch,
    )
    global_psf = global_psf / global_psf.sum().clamp_min(1e-12)

    simulator._save_tensor_image(
        simulator._normalize_for_preview(torch.log1p(global_psf * 1e3)).unsqueeze(0).repeat(3, 1, 1),
        os.path.join(output_dir, "fabrication_grid_global_psf_preview.png"),
    )
    torch.save(global_psf.cpu(), os.path.join(output_dir, "fabrication_grid_global_psf.pt"))

    psf_summary: Dict[str, object] = {
        "exported_fabrication_variant": variant,
        "tile_size_for_preview": [int(tile_size[0]), int(tile_size[1])],
        "tile_size_source": tile_size_source,
        "active_sensor_shape": [int(active_sensor_shape[0]), int(active_sensor_shape[1])],
        "sensor_guard_band": int(sensor_guard_band),
        "sensor_shape_for_psf": [int(sensor_shape[0]), int(sensor_shape[1])],
        "psf_wavelength_m": wavelength_m,
        "psf_wavelength_nm": wavelength_m * 1e9,
        "psf_depth_m": depth_m,
        "fabrication_pitch_um": fabrication_pitch * 1e6,
        "sensor_pixel_pitch_um": float(simulator.config.sensor_pixel_pitch * 1e6),
        "fabrication_heightmap_shape_hw": [int(fabrication_resolution[0]), int(fabrication_resolution[1])],
        "propagation_shape_hw": [int(propagation_shape[0]), int(propagation_shape[1])],
        "output_files": [
            "fabrication_grid_global_psf_preview.png",
            "fabrication_grid_global_psf.pt",
        ],
        "note": (
            "This PSF is computed from the quantized exported fabrication DOE on the fabrication grid, using a "
            "single wavelength and a single reference depth. It is then resampled onto the full sensor grid."
        ),
    }
    with open(os.path.join(output_dir, "psf_preview_metadata.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(psf_summary, handle, sort_keys=False, allow_unicode=True)


def export_high_res_doe(
    config_path: str,
    output_dir: str,
    fabrication_pitch: float,
    variant: str,
    device: str,
    save_psf_previews: bool,
    psf_tile_height: int,
    psf_tile_width: int,
    psf_wavelength_m: Optional[float],
    psf_depth_m: Optional[float],
) -> None:
    run_config, simulator_config = load_yaml_configs(config_path)
    simulator = HDRDOESimulator(simulator_config, device=_resolve_device(device))

    if variant not in {"no_depth", "depth_encoded"}:
        raise ValueError("variant must be 'no_depth' or 'depth_encoded'.")

    aperture_diameter_m = simulator._fabrication_target_aperture_diameter_m()
    resolution = int(round(aperture_diameter_m / fabrication_pitch))
    if resolution <= 0:
        raise ValueError("Computed high-res fabrication resolution is invalid.")

    max_height_m = simulator.config.doe_substrate_thickness_m + simulator._doe_height_wrap_m()
    phase_to_height = simulator.config.design_wavelength / (
        2.0 * math.pi * (simulator.config.refractive_index - 1.0)
    )
    phase_scale = 2.0 * math.pi / simulator.config.design_wavelength
    substrate_m = float(simulator.config.doe_substrate_thickness_m)
    aperture_radius_m = 0.5 * aperture_diameter_m

    physical_block_m = simulator.config.interleave_block_size * simulator.config.optical_pitch
    fabrication_block_size = max(1, int(round(physical_block_m / fabrication_pitch)))
    fractions = simulator.branch_area_fractions.detach().cpu().numpy().astype(np.float64)
    carry = np.zeros(4, dtype=np.float64)
    sin_y, sin_x = _target_sines(simulator)
    strengths = (
        np.asarray(simulator.config.branch_depth_strengths, dtype=np.float32)
        if variant == "depth_encoded"
        else np.zeros(4, dtype=np.float32)
    )
    phase_biases = simulator.zero_order_phase_biases.detach().cpu().numpy().astype(np.float32)
    # Use the smooth low-res zero-order residual only as a correction term. The main phase is
    # analytically recomputed on the high-res fabrication grid, not interpolated from the PNG.
    residual_low = simulator.zero_order_residual_phase.detach().cpu().numpy().astype(np.float32)
    low_resolution = int(simulator.config.optical_resolution)
    low_center = 0.5 * (low_resolution - 1)
    high_center = 0.5 * (resolution - 1)

    image = np.zeros((resolution, resolution), dtype=np.uint16)
    x_coords_all = (np.arange(resolution, dtype=np.float32) - high_center) * fabrication_pitch
    low_x_all = np.rint(x_coords_all / simulator.config.optical_pitch + low_center).astype(np.int64)
    low_x_all = np.clip(low_x_all, 0, low_resolution - 1)

    for y0 in range(0, resolution, fabrication_block_size):
        y1 = min(resolution, y0 + fabrication_block_size)
        y_idx = np.arange(y0, y1, dtype=np.int64)
        y_coords = (y_idx.astype(np.float32) - high_center) * fabrication_pitch
        low_y = np.rint(y_coords / simulator.config.optical_pitch + low_center).astype(np.int64)
        low_y = np.clip(low_y, 0, low_resolution - 1)

        for x0 in range(0, resolution, fabrication_block_size):
            x1 = min(resolution, x0 + fabrication_block_size)
            x_idx = np.arange(x0, x1, dtype=np.int64)
            x_coords = x_coords_all[x0:x1]
            xx, yy = np.meshgrid(x_coords, y_coords)
            aperture = (xx * xx + yy * yy) <= aperture_radius_m * aperture_radius_m
            valid = np.argwhere(aperture)
            num_valid = int(valid.shape[0])
            if num_valid == 0:
                continue

            counts, carry = _allocate_counts(num_valid, fractions, carry)
            global_y = y_idx[valid[:, 0]]
            global_x = x_idx[valid[:, 1]]
            order = np.argsort(_coordinate_hash_scores(global_y, global_x, simulator.config.mask_seed))

            block_gray = image[y0:y1, x0:x1]
            start = 0
            for branch_idx, count in enumerate(counts.tolist()):
                if count <= 0:
                    continue
                selected = valid[order[start:start + count]]
                start += count
                local_y = selected[:, 0]
                local_x = selected[:, 1]
                x_sel = x_coords[local_x]
                y_sel = y_coords[local_y]
                phase = phase_scale * (sin_x[branch_idx] * x_sel + sin_y[branch_idx] * y_sel)
                if abs(float(strengths[branch_idx])) > 1e-8:
                    phase += strengths[branch_idx] * (
                        -math.pi * (x_sel * x_sel + y_sel * y_sel)
                        / (simulator.config.design_wavelength * simulator.sensor_distance)
                    )
                phase += float(phase_biases[branch_idx])
                phase += residual_low[low_y[local_y], low_x_all[x0:x1][local_x]]
                wrapped = np.mod(phase, 2.0 * math.pi)
                absolute_height = wrapped * phase_to_height + substrate_m
                gray = np.rint(np.clip(absolute_height, 0.0, max_height_m) / max_height_m * 65535.0)
                block_gray[local_y, local_x] = gray.astype(np.uint16)

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "doe_fabrication_heightmap_16bit.png")
    Image.fromarray(image, mode="I;16").save(png_path)

    metadata = {
        "png_encoding": (
            "uint16 grayscale, 0 outside aperture; inside aperture linear absolute height including substrate, "
            "0..absolute_height_max_m mapped to 0..65535"
        ),
        "design_wavelength_m": float(simulator.config.design_wavelength),
        "refractive_index_design": float(simulator.config.refractive_index),
        "wrap_height_m": float(simulator._doe_height_wrap_m()),
        "wrap_height_um": float(simulator._doe_height_wrap_m() * 1e6),
        "doe_substrate_thickness_m": float(simulator.config.doe_substrate_thickness_m),
        "doe_substrate_thickness_um": float(simulator.config.doe_substrate_thickness_m * 1e6),
        "outside_aperture_gray_value": 0,
        "absolute_height_max_m": float(max_height_m),
        "absolute_height_max_um": float(max_height_m * 1e6),
        "quantization_step_m": float(max_height_m / 65535.0),
        "quantization_step_nm": float(max_height_m * 1e9 / 65535.0),
        "png_shape_hw": [int(resolution), int(resolution)],
        "aperture_fill_ratio": float(simulator.config.aperture_fill_ratio),
        "simulated_optical_pitch_um": float(simulator.config.optical_pitch * 1e6),
        "simulated_aperture_diameter_mm": float(simulator._simulated_aperture_diameter_m() * 1e3),
        "fabrication_target_aperture_diameter_mm": float(aperture_diameter_m * 1e3),
        "fabrication_canvas_width_mm": float(aperture_diameter_m * 1e3),
        "fabrication_pixel_pitch_um": float(fabrication_pitch * 1e6),
        "fabrication_interleave_block_size_cells": int(fabrication_block_size),
        "fabrication_interleave_block_size_um": float(fabrication_block_size * fabrication_pitch * 1e6),
        "full_simulation_canvas_width_mm": float(
            simulator.config.optical_resolution * simulator.config.optical_pitch * 1e3
        ),
        "cropped_to_circular_aperture_bounding_box": True,
        "lateral_scale_factor_vs_simulation": 1.0,
        "fabrication_target_differs_from_simulation": False,
        "current_run_used_depth_encoding": bool(variant == "depth_encoded"),
        "exported_fabrication_variant": variant,
        "fabrication_png_filename": "doe_fabrication_heightmap_16bit.png",
        "generated_from_continuous_phase_at_fabrication_pitch": True,
        "not_simple_interpolation_from_previous_png": True,
        "note": (
            "The high-resolution fabrication heightmap recomputes the splitter phase on the fabrication grid. "
            "The full sensor simulation remains limited to the simulation optical grid because full 0.4 um FFT "
            "propagation over the 2048x2488 sensor would require tens of GB of temporary memory."
        ),
    }
    with open(os.path.join(output_dir, "doe_fabrication_metadata.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False, allow_unicode=True)

    if save_psf_previews:
        _export_fabrication_grid_global_psf(
            simulator=simulator,
            variant=variant,
            output_dir=output_dir,
            run_config=run_config,
            fabrication_png_gray=image,
            fabrication_pitch=fabrication_pitch,
            substrate_m=substrate_m,
            absolute_height_max_m=max_height_m,
            psf_tile_height=psf_tile_height,
            psf_tile_width=psf_tile_width,
            psf_wavelength_m=psf_wavelength_m,
            psf_depth_m=psf_depth_m,
        )

    print(f"Saved high-res DOE PNG: {png_path}")
    print(f"Resolution: {resolution} x {resolution}, fabrication pitch: {fabrication_pitch * 1e6:.3f} um")
    if save_psf_previews:
        print(f"Saved fabrication-grid full-sensor PSF outputs to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a high-resolution fabrication DOE heightmap.")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "HDRDOESimulator.yaml"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fabrication_pitch", type=float, default=None)
    parser.add_argument("--variant", choices=("no_depth", "depth_encoded"), default="no_depth")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--save_psf_previews",
        dest="save_psf_previews",
        action="store_true",
        help="Also save fabrication-grid single-wavelength full-sensor PSF outputs for the exported DOE variant.",
    )
    parser.add_argument(
        "--no_psf_previews",
        dest="save_psf_previews",
        action="store_false",
        help="Skip PSF preview export.",
    )
    parser.set_defaults(save_psf_previews=True)
    parser.add_argument(
        "--psf_tile_height",
        type=int,
        default=0,
        help="Optional tile height used when choosing the sensor shape for the full-sensor PSF export.",
    )
    parser.add_argument(
        "--psf_tile_width",
        type=int,
        default=0,
        help="Optional tile width used when choosing the sensor shape for the full-sensor PSF export.",
    )
    parser.add_argument(
        "--psf_wavelength_m",
        type=float,
        default=0.0,
        help="Optional single wavelength in meters used for the fabrication-grid PSF. Defaults to design wavelength.",
    )
    parser.add_argument(
        "--psf_depth_m",
        type=float,
        default=0.0,
        help="Optional single reference depth in meters used for the fabrication-grid PSF. Defaults to scene_reference_depth.",
    )
    args = parser.parse_args()

    _, simulator_config = load_yaml_configs(args.config)
    fabrication_pitch = args.fabrication_pitch or simulator_config.fabrication_pixel_pitch
    if fabrication_pitch <= 0:
        raise ValueError("Set --fabrication_pitch or simulator.fabrication_pixel_pitch to a positive value.")

    export_high_res_doe(
        config_path=args.config,
        output_dir=args.output_dir,
        fabrication_pitch=fabrication_pitch,
        variant=args.variant,
        device=args.device,
        save_psf_previews=args.save_psf_previews,
        psf_tile_height=args.psf_tile_height,
        psf_tile_width=args.psf_tile_width,
        psf_wavelength_m=(args.psf_wavelength_m if args.psf_wavelength_m > 0 else None),
        psf_depth_m=(args.psf_depth_m if args.psf_depth_m > 0 else None),
    )


if __name__ == "__main__":
    main()
