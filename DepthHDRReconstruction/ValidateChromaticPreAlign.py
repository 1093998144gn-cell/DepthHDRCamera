import argparse
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from HDRDOESimulator import HDRDOESimulator, load_yaml_configs


def _save_tensor_image(image: torch.Tensor, path: str) -> None:
    image = image.detach().cpu().clamp(0.0, 1.0)
    image = (image * 255.0).round().byte()
    image = image.permute(1, 2, 0).numpy()
    Image.fromarray(image).save(path)


def _default_validation_output_dir(result_dir: str) -> str:
    return os.path.join(result_dir, "chromatic_prealign_validation")


def _normalize_for_preview(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    tensor = tensor - tensor.min()
    return (tensor / tensor.max().clamp_min(1e-8)).float()


def _crop_centered_window(
    image: torch.Tensor,
    center_y: float,
    center_x: float,
    out_h: int,
    out_w: int,
) -> torch.Tensor:
    if image.dim() != 3:
        raise ValueError("image must have shape [C,H,W]")

    _, src_h, src_w = image.shape
    ys = torch.arange(out_h, device=image.device, dtype=torch.float32) - (out_h - 1) / 2.0 + center_y
    xs = torch.arange(out_w, device=image.device, dtype=torch.float32) - (out_w - 1) / 2.0 + center_x
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    if src_w == 1:
        norm_x = torch.zeros_like(grid_x)
    else:
        norm_x = 2.0 * grid_x / (src_w - 1) - 1.0
    if src_h == 1:
        norm_y = torch.zeros_like(grid_y)
    else:
        norm_y = 2.0 * grid_y / (src_h - 1) - 1.0

    grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)
    crop = F.grid_sample(
        image.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return crop.squeeze(0)


def _extract_sensor_grid_tile(
    image: torch.Tensor,
    tile_h: int,
    tile_w: int,
    branch_idx: int,
    grid_shape: Tuple[int, int],
) -> torch.Tensor:
    if image.dim() != 3:
        raise ValueError("image must have shape [C,H,W]")

    row = branch_idx // grid_shape[1]
    col = branch_idx % grid_shape[1]
    y0 = row * tile_h
    y1 = y0 + tile_h
    x0 = col * tile_w
    x1 = x0 + tile_w
    return image[:, y0:y1, x0:x1]


def _centered_window_fits(
    image_shape: Tuple[int, int],
    center_y: float,
    center_x: float,
    out_h: int,
    out_w: int,
) -> bool:
    half_h = 0.5 * (out_h - 1)
    half_w = 0.5 * (out_w - 1)
    return (
        center_y - half_h >= 0.0
        and center_y + half_h <= image_shape[0] - 1
        and center_x - half_w >= 0.0
        and center_x + half_w <= image_shape[1] - 1
    )


def _translate_channel_affine(channel: torch.Tensor, shift_y: float, shift_x: float) -> torch.Tensor:
    if channel.dim() != 2:
        raise ValueError("channel must have shape [H,W]")

    h, w = channel.shape
    theta = torch.eye(2, 3, dtype=channel.dtype, device=channel.device)
    if w > 1:
        theta[0, 2] = 2.0 * shift_x / (w - 1)
    if h > 1:
        theta[1, 2] = 2.0 * shift_y / (h - 1)

    grid = F.affine_grid(theta.unsqueeze(0), size=(1, 1, h, w), align_corners=True)
    aligned = F.grid_sample(
        channel.unsqueeze(0).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return aligned.squeeze(0).squeeze(0)


def _align_tile_rgb(tile: torch.Tensor, residual_shifts: torch.Tensor) -> torch.Tensor:
    aligned_channels = []
    for channel_idx in range(tile.shape[0]):
        shift_y = float(residual_shifts[channel_idx, 0].item())
        shift_x = float(residual_shifts[channel_idx, 1].item())
        aligned_channels.append(_translate_channel_affine(tile[channel_idx], shift_y, shift_x))
    return torch.stack(aligned_channels, dim=0)


def _translate_channel_flow(
    channel: torch.Tensor,
    shift_y_map: torch.Tensor,
    shift_x_map: torch.Tensor,
) -> torch.Tensor:
    if channel.dim() != 2:
        raise ValueError("channel must have shape [H,W]")
    if shift_y_map.shape != channel.shape or shift_x_map.shape != channel.shape:
        raise ValueError("shift maps must match the channel shape")

    h, w = channel.shape
    y_coords = torch.arange(h, device=channel.device, dtype=channel.dtype).view(h, 1).expand(h, w)
    x_coords = torch.arange(w, device=channel.device, dtype=channel.dtype).view(1, w).expand(h, w)

    source_y = y_coords + shift_y_map.to(device=channel.device, dtype=channel.dtype)
    source_x = x_coords + shift_x_map.to(device=channel.device, dtype=channel.dtype)

    if w == 1:
        norm_x = torch.zeros_like(source_x)
    else:
        norm_x = 2.0 * source_x / (w - 1) - 1.0
    if h == 1:
        norm_y = torch.zeros_like(source_y)
    else:
        norm_y = 2.0 * source_y / (h - 1) - 1.0

    grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)
    aligned = F.grid_sample(
        channel.unsqueeze(0).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return aligned.squeeze(0).squeeze(0)


def _extract_patch(image_2d: torch.Tensor, center_y: int, center_x: int, patch_size: int) -> torch.Tensor:
    if image_2d.dim() != 2:
        raise ValueError("image_2d must have shape [H,W]")
    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer")

    half = patch_size // 2
    y0 = int(center_y) - half
    x0 = int(center_x) - half
    y1 = y0 + patch_size
    x1 = x0 + patch_size
    if y0 < 0 or x0 < 0 or y1 > image_2d.shape[0] or x1 > image_2d.shape[1]:
        raise ValueError("Requested patch falls outside the image bounds")
    return image_2d[y0:y1, x0:x1]


def _grid_centers(
    length: int,
    num_points: int,
    patch_size: int,
    search_radius: int,
    margin: int,
) -> List[int]:
    half = patch_size // 2
    start = margin + search_radius + half
    end = length - 1 - margin - search_radius - half
    if end < start:
        return []
    if num_points <= 1 or end == start:
        return [int(round((start + end) * 0.5))]

    centers = []
    for value in torch.round(torch.linspace(float(start), float(end), steps=num_points)).to(torch.int64).tolist():
        value = int(value)
        if not centers or centers[-1] != value:
            centers.append(value)
    return centers


def _upsample_shift_grid(shift_grid: torch.Tensor, out_shape: Tuple[int, int]) -> torch.Tensor:
    if shift_grid.dim() != 2:
        raise ValueError("shift_grid must have shape [Gh,Gw]")
    return F.interpolate(
        shift_grid.unsqueeze(0).unsqueeze(0),
        size=out_shape,
        mode="bilinear",
        align_corners=True,
    ).squeeze(0).squeeze(0)


def _estimate_local_shift_grid(
    reference_grad: torch.Tensor,
    moving_grad: torch.Tensor,
    grid_shape: Tuple[int, int],
    patch_size: int,
    search_radius: int,
    margin_y: int,
    margin_x: int,
    min_gradient_std: float,
    min_correlation: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if reference_grad.shape != moving_grad.shape:
        raise ValueError("reference_grad and moving_grad must have the same shape")

    centers_y = _grid_centers(reference_grad.shape[0], grid_shape[0], patch_size, search_radius, margin_y)
    centers_x = _grid_centers(reference_grad.shape[1], grid_shape[1], patch_size, search_radius, margin_x)
    if not centers_y or not centers_x:
        empty = reference_grad.new_zeros((1, 1))
        return empty, empty.clone(), empty.clone(), empty.clone()

    shift_y_grid = reference_grad.new_zeros((len(centers_y), len(centers_x)))
    shift_x_grid = reference_grad.new_zeros((len(centers_y), len(centers_x)))
    score_grid = reference_grad.new_zeros((len(centers_y), len(centers_x)))
    confidence_grid = reference_grad.new_zeros((len(centers_y), len(centers_x)))
    if search_radius <= 0:
        return shift_y_grid, shift_x_grid, score_grid, confidence_grid

    correlation_range = max(1e-6, 1.0 - min_correlation)
    for row_idx, center_y in enumerate(centers_y):
        for col_idx, center_x in enumerate(centers_x):
            reference_patch = _extract_patch(reference_grad, center_y, center_x, patch_size)
            if float(reference_patch.std().item()) < min_gradient_std:
                continue

            best_score = -1.0
            best_shift_y = 0.0
            best_shift_x = 0.0
            for shift_y in range(-search_radius, search_radius + 1):
                for shift_x in range(-search_radius, search_radius + 1):
                    moving_patch = _extract_patch(moving_grad, center_y + shift_y, center_x + shift_x, patch_size)
                    score = _pearson_corr(reference_patch, moving_patch)
                    if score > best_score:
                        best_score = score
                        best_shift_y = float(shift_y)
                        best_shift_x = float(shift_x)

            score_grid[row_idx, col_idx] = best_score
            if best_score < min_correlation:
                continue

            confidence = max(0.0, min(1.0, (best_score - min_correlation) / correlation_range))
            confidence_grid[row_idx, col_idx] = confidence
            shift_y_grid[row_idx, col_idx] = best_shift_y * confidence
            shift_x_grid[row_idx, col_idx] = best_shift_x * confidence

    return shift_y_grid, shift_x_grid, score_grid, confidence_grid


def _default_local_alignment_metadata(
    channel_idx: int,
    enabled: bool,
    grid_shape: Tuple[int, int],
    patch_size: int,
    search_radius: int,
) -> Dict[str, Any]:
    return {
        "channel_index": channel_idx,
        "is_reference_channel": channel_idx == 1,
        "local_block_correction_enabled": enabled,
        "local_grid_shape": [int(max(1, grid_shape[0])), int(max(1, grid_shape[1]))],
        "local_patch_size": int(patch_size),
        "local_search_radius": int(search_radius),
        "num_active_blocks": 0,
        "mean_confidence": 0.0,
        "max_abs_shift_y_pixels": 0.0,
        "max_abs_shift_x_pixels": 0.0,
        "shift_y_grid_pixels": [[0.0]],
        "shift_x_grid_pixels": [[0.0]],
        "correlation_grid": [[0.0]],
        "confidence_grid": [[0.0]],
    }


def _apply_local_block_correction(
    globally_aligned_tile: torch.Tensor,
    margin_y: int,
    margin_x: int,
    grid_shape: Tuple[int, int],
    patch_size: int,
    search_radius: int,
    min_gradient_std: float,
    min_correlation: float,
    enabled: bool,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Tuple[int, int]]:
    if globally_aligned_tile.dim() != 3 or globally_aligned_tile.shape[0] != 3:
        raise ValueError("globally_aligned_tile must have shape [3,H,W]")

    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError("local_patch_size must be a positive odd integer")
    if search_radius < 0:
        raise ValueError("local_search_radius must be non-negative")

    if not enabled or grid_shape[0] <= 0 or grid_shape[1] <= 0 or search_radius == 0:
        metadata = [
            _default_local_alignment_metadata(channel_idx, enabled, grid_shape, patch_size, search_radius)
            for channel_idx in range(globally_aligned_tile.shape[0])
        ]
        return globally_aligned_tile, metadata, (margin_y, margin_x)

    reference_grad = _gradient_magnitude(globally_aligned_tile[1])
    corrected_channels = [globally_aligned_tile[channel_idx] for channel_idx in range(globally_aligned_tile.shape[0])]
    metadata: List[Dict[str, Any]] = []
    max_residual_y = 0.0
    max_residual_x = 0.0

    for channel_idx in range(globally_aligned_tile.shape[0]):
        if channel_idx == 1:
            metadata.append(
                _default_local_alignment_metadata(channel_idx, True, grid_shape, patch_size, search_radius)
            )
            continue

        moving_grad = _gradient_magnitude(globally_aligned_tile[channel_idx])
        shift_y_grid, shift_x_grid, score_grid, confidence_grid = _estimate_local_shift_grid(
            reference_grad,
            moving_grad,
            grid_shape=grid_shape,
            patch_size=patch_size,
            search_radius=search_radius,
            margin_y=margin_y,
            margin_x=margin_x,
            min_gradient_std=min_gradient_std,
            min_correlation=min_correlation,
        )
        shift_y_map = _upsample_shift_grid(shift_y_grid, globally_aligned_tile.shape[-2:])
        shift_x_map = _upsample_shift_grid(shift_x_grid, globally_aligned_tile.shape[-2:])
        corrected_channels[channel_idx] = _translate_channel_flow(
            globally_aligned_tile[channel_idx],
            shift_y_map,
            shift_x_map,
        )

        channel_max_shift_y = float(shift_y_grid.abs().max().item())
        channel_max_shift_x = float(shift_x_grid.abs().max().item())
        max_residual_y = max(max_residual_y, channel_max_shift_y)
        max_residual_x = max(max_residual_x, channel_max_shift_x)
        metadata.append(
            {
                "channel_index": channel_idx,
                "is_reference_channel": False,
                "local_block_correction_enabled": True,
                "local_grid_shape": [int(shift_y_grid.shape[0]), int(shift_y_grid.shape[1])],
                "local_patch_size": int(patch_size),
                "local_search_radius": int(search_radius),
                "num_active_blocks": int((confidence_grid > 0).sum().item()),
                "mean_confidence": float(confidence_grid.mean().item()),
                "max_abs_shift_y_pixels": channel_max_shift_y,
                "max_abs_shift_x_pixels": channel_max_shift_x,
                "shift_y_grid_pixels": shift_y_grid.cpu().tolist(),
                "shift_x_grid_pixels": shift_x_grid.cpu().tolist(),
                "correlation_grid": score_grid.cpu().tolist(),
                "confidence_grid": confidence_grid.cpu().tolist(),
            }
        )

    corrected_tile = torch.stack(corrected_channels, dim=0)
    final_margin_y = margin_y + int(math.ceil(max_residual_y))
    final_margin_x = margin_x + int(math.ceil(max_residual_x))
    return corrected_tile, metadata, (final_margin_y, final_margin_x)


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ) / 4.0
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ) / 4.0
    return kernel_x.view(1, 1, 3, 3), kernel_y.view(1, 1, 3, 3)


def _gradient_magnitude(image_2d: torch.Tensor) -> torch.Tensor:
    kernel_x, kernel_y = _sobel_kernels(image_2d.device, image_2d.dtype)
    image = image_2d.unsqueeze(0).unsqueeze(0)
    grad_x = F.conv2d(image, kernel_x, padding=1)
    grad_y = F.conv2d(image, kernel_y, padding=1)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-12).squeeze(0).squeeze(0)


def _luminance(image: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=image.dtype, device=image.device).view(3, 1, 1)
    return (image * weights).sum(dim=0)


def _pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt(a.square().sum() * b.square().sum()).clamp_min(1e-12)
    return float((a * b).sum() / denom)


def _edge_alignment_score(tile: torch.Tensor) -> float:
    gradients = [_gradient_magnitude(tile[channel_idx]) for channel_idx in range(3)]
    return 0.5 * (
        _pearson_corr(gradients[0], gradients[1]) + _pearson_corr(gradients[2], gradients[1])
    )


def _tenengrad_sharpness(tile: torch.Tensor) -> float:
    grad = _gradient_magnitude(_luminance(tile))
    return float(grad.square().mean())


def _edge_chroma_energy(tile: torch.Tensor) -> float:
    luminance = _luminance(tile)
    grad = _gradient_magnitude(luminance)
    threshold = torch.quantile(grad.reshape(-1), 0.8)
    edge_mask = grad >= threshold
    if edge_mask.sum() == 0:
        return 0.0
    channel_mean = tile.mean(dim=0, keepdim=True)
    chroma = torch.sqrt((tile - channel_mean).square().mean(dim=0))
    return float(chroma[edge_mask].mean())


def _make_tile_grid(tiles: List[torch.Tensor]) -> torch.Tensor:
    top = torch.cat([tiles[0], tiles[1]], dim=-1)
    bottom = torch.cat([tiles[2], tiles[3]], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def _crop_center_tensor(tile: torch.Tensor, out_shape: Tuple[int, int]) -> torch.Tensor:
    out_h, out_w = out_shape
    h, w = tile.shape[-2:]
    start_y = max(0, (h - out_h) // 2)
    start_x = max(0, (w - out_w) // 2)
    return tile[..., start_y:start_y + out_h, start_x:start_x + out_w]


def _crop_tiles_to_common_shape(tiles: List[torch.Tensor], out_shape: Tuple[int, int]) -> List[torch.Tensor]:
    return [_crop_center_tensor(tile, out_shape) for tile in tiles]


def _crop_valid_region(tile: torch.Tensor, margin_y: int, margin_x: int) -> torch.Tensor:
    if margin_y <= 0 and margin_x <= 0:
        return tile
    h, w = tile.shape[-2:]
    y0 = min(max(margin_y, 0), max(0, h - 1))
    y1 = max(y0 + 1, h - min(max(margin_y, 0), max(0, h - 1)))
    x0 = min(max(margin_x, 0), max(0, w - 1))
    x1 = max(x0 + 1, w - min(max(margin_x, 0), max(0, w - 1)))
    return tile[..., y0:y1, x0:x1]


def _rectangular_valid_mask(
    shape: Tuple[int, int],
    margin_y: int,
    margin_x: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    h, w = shape
    y0 = min(max(margin_y, 0), max(0, h - 1))
    y1 = max(y0 + 1, h - min(max(margin_y, 0), max(0, h - 1)))
    x0 = min(max(margin_x, 0), max(0, w - 1))
    x1 = max(x0 + 1, w - min(max(margin_x, 0), max(0, w - 1)))
    mask = torch.zeros((1, h, w), device=device, dtype=dtype)
    mask[:, y0:y1, x0:x1] = 1.0
    return mask


def _composite_aligned_preview(
    raw_tile: torch.Tensor,
    aligned_tile: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if raw_tile.shape != aligned_tile.shape or raw_tile.shape != valid_mask.shape:
        raise ValueError("raw_tile, aligned_tile, and valid_mask must have the same shape")
    return aligned_tile * valid_mask + raw_tile * (1.0 - valid_mask)


def _load_tensor(path: str, device: torch.device) -> torch.Tensor:
    try:
        tensor = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        tensor = torch.load(path, map_location=device)
    return tensor.float()


def _maybe_layout_tensor(
    value: Any,
    device: torch.device,
    expected_dim: int,
) -> Optional[torch.Tensor]:
    if value is None:
        return None
    try:
        tensor = torch.tensor(value, dtype=torch.float32, device=device)
    except (TypeError, ValueError):
        return None
    if tensor.dim() != expected_dim or tensor.numel() == 0:
        return None
    if not torch.isfinite(tensor).all():
        return None
    return tensor


def _load_saved_alignment_metadata(result_dir: str, device: torch.device) -> Dict[str, Any]:
    layout_path = os.path.join(result_dir, "branch_layout_stats.yaml")
    if not os.path.isfile(layout_path):
        return {"available": False, "source": "current_config"}

    with open(layout_path, "r", encoding="utf-8") as handle:
        layout_stats = yaml.safe_load(handle) or {}

    return {
        "available": True,
        "source": layout_path,
        "branch_shifts": _maybe_layout_tensor(
            layout_stats.get("branch_channel_shifts_pixels"),
            device=device,
            expected_dim=3,
        ),
        "residual_shifts": _maybe_layout_tensor(
            layout_stats.get("branch_channel_residual_shifts_pixels"),
            device=device,
            expected_dim=3,
        ),
        "wavelengths": _maybe_layout_tensor(
            layout_stats.get("wavelengths_m"),
            device=device,
            expected_dim=1,
        ),
        "design_wavelength": layout_stats.get("design_wavelength_m"),
    }


def _resolve_alignment_parameters(
    simulator: HDRDOESimulator,
    result_dir: str,
) -> Dict[str, Any]:
    config_branch_shifts = simulator._branch_channel_shifts_pixels().detach()
    config_residual_shifts = simulator._branch_channel_residual_shifts_pixels().detach()
    config_wavelengths = simulator.wavelengths.detach()
    config_design_wavelength = float(simulator.config.design_wavelength)

    resolved = {
        "branch_shifts": config_branch_shifts,
        "residual_shifts": config_residual_shifts,
        "wavelengths": config_wavelengths,
        "design_wavelength": config_design_wavelength,
        "source": "current_config",
        "used_saved_branch_shifts": False,
        "used_saved_residual_shifts": False,
        "used_saved_wavelengths": False,
        "used_saved_design_wavelength": False,
    }

    saved = _load_saved_alignment_metadata(result_dir, simulator.device)
    if not saved["available"]:
        return resolved

    saved_branch_shifts = saved["branch_shifts"]
    if saved_branch_shifts is not None and saved_branch_shifts.shape == config_branch_shifts.shape:
        resolved["branch_shifts"] = saved_branch_shifts
        resolved["source"] = saved["source"]
        resolved["used_saved_branch_shifts"] = True

    saved_residual_shifts = saved["residual_shifts"]
    if saved_residual_shifts is not None and saved_residual_shifts.shape == config_residual_shifts.shape:
        resolved["residual_shifts"] = saved_residual_shifts
        resolved["source"] = saved["source"]
        resolved["used_saved_residual_shifts"] = True

    saved_wavelengths = saved["wavelengths"]
    if saved_wavelengths is not None and saved_wavelengths.shape == config_wavelengths.shape:
        resolved["wavelengths"] = saved_wavelengths
        resolved["source"] = saved["source"]
        resolved["used_saved_wavelengths"] = True

    saved_design_wavelength = saved["design_wavelength"]
    if isinstance(saved_design_wavelength, (int, float)) and math.isfinite(float(saved_design_wavelength)):
        resolved["design_wavelength"] = float(saved_design_wavelength)
        resolved["source"] = saved["source"]
        resolved["used_saved_design_wavelength"] = True

    return resolved


def _rgb_to_wavelength_indices(wavelengths: torch.Tensor) -> torch.Tensor:
    return torch.argsort(wavelengths, descending=True)


def _is_simulation_result_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "sensor_mosaic_linear.pt")) and os.path.isfile(
        os.path.join(path, "scene_input_linear.pt")
    )


def _find_result_dirs(results_root: str, result_name: str = "") -> List[str]:
    if not os.path.isdir(results_root):
        return []

    matched = []
    for name in sorted(os.listdir(results_root)):
        candidate = os.path.join(results_root, name)
        if not os.path.isdir(candidate):
            continue
        if result_name and name != result_name:
            continue
        if _is_simulation_result_dir(candidate):
            matched.append(candidate)
    return matched


def _process_result_dir(
    simulator: HDRDOESimulator,
    result_dir: str,
    output_dir: str = "",
    enable_local_block_correction: bool = True,
    local_grid_shape: Tuple[int, int] = (4, 6),
    local_patch_size: int = 31,
    local_search_radius: int = 3,
    local_min_gradient_std: float = 1e-4,
    local_min_correlation: float = 0.15,
    save_branch_previews: bool = False,
) -> dict:
    sensor_path = os.path.join(result_dir, "sensor_mosaic_linear.pt")
    scene_input_path = os.path.join(result_dir, "scene_input_linear.pt")
    if not os.path.exists(sensor_path):
        raise FileNotFoundError(f"Missing sensor mosaic tensor: {sensor_path}")
    if not os.path.exists(scene_input_path):
        raise FileNotFoundError(f"Missing scene input tensor: {scene_input_path}")

    sensor = _load_tensor(sensor_path, simulator.device)
    scene_input = _load_tensor(scene_input_path, simulator.device)

    output_dir = output_dir or _default_validation_output_dir(result_dir)
    os.makedirs(output_dir, exist_ok=True)
    obsolete_output_filenames = [
        "tile_grid_aligned_global.png",
        "tile_grid_aligned_global_context.png",
        "tile_grid_aligned_context.png",
        "tile_grid_raw_valid.png",
        "tile_grid_aligned_global_valid.png",
        "tile_grid_aligned_valid.png",
        "tile_grid_common_valid_mask.png",
        "scene_input_reference.png",
    ]
    for filename in obsolete_output_filenames:
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            os.remove(path)
    for filename in os.listdir(output_dir):
        if filename.startswith("branch_") and filename.endswith(".png"):
            os.remove(os.path.join(output_dir, filename))

    tile_h, tile_w = scene_input.shape[-2:]
    sensor_h, sensor_w = sensor.shape[-2:]
    alignment_params = _resolve_alignment_parameters(simulator, result_dir)
    branch_shifts = alignment_params["branch_shifts"]
    residual_shifts = alignment_params["residual_shifts"]
    wavelengths = alignment_params["wavelengths"]
    rgb_to_wavelength_indices = _rgb_to_wavelength_indices(wavelengths)
    residual_shifts_rgb = torch.index_select(residual_shifts, 1, rgb_to_wavelength_indices)
    design_wavelength = alignment_params["design_wavelength"]
    design_wavelength_inferred_from_saved_residuals = False
    if (
        alignment_params["used_saved_residual_shifts"]
        and alignment_params["used_saved_wavelengths"]
        and not alignment_params["used_saved_design_wavelength"]
    ):
        residual_energy = residual_shifts.abs().sum(dim=(0, 2))
        design_idx = int(torch.argmin(residual_energy).item())
        design_wavelength = float(wavelengths[design_idx].item())
        design_wavelength_inferred_from_saved_residuals = True
    else:
        design_idx = int(torch.argmin(torch.abs(wavelengths - design_wavelength)).item())

    sensor_center = torch.tensor(
        [(sensor_h - 1) / 2.0, (sensor_w - 1) / 2.0],
        dtype=torch.float32,
        device=simulator.device,
    )
    expected_sensor_shape = (
        simulator.config.grid_shape[0] * tile_h,
        simulator.config.grid_shape[1] * tile_w,
    )
    branch_centers = sensor_center.unsqueeze(0) + branch_shifts[:, design_idx]
    branch_alignment_crop_margins_yx: List[Tuple[int, int]] = []
    use_guard_band_centered_tiles = True
    for branch_idx in range(simulator.num_branches):
        global_margin_y = int(torch.ceil(residual_shifts_rgb[branch_idx, :, 0].abs().max()).item())
        global_margin_x = int(torch.ceil(residual_shifts_rgb[branch_idx, :, 1].abs().max()).item())
        extraction_margin_y = global_margin_y + (local_search_radius if enable_local_block_correction else 0)
        extraction_margin_x = global_margin_x + (local_search_radius if enable_local_block_correction else 0)
        branch_alignment_crop_margins_yx.append((extraction_margin_y, extraction_margin_x))
        if not _centered_window_fits(
            (sensor_h, sensor_w),
            center_y=float(branch_centers[branch_idx, 0].item()),
            center_x=float(branch_centers[branch_idx, 1].item()),
            out_h=tile_h + 2 * extraction_margin_y,
            out_w=tile_w + 2 * extraction_margin_x,
        ):
            use_guard_band_centered_tiles = False
    use_fixed_grid_tiles = (sensor_h, sensor_w) == expected_sensor_shape
    tile_extraction_mode = (
        "centered_guard_crop"
        if use_guard_band_centered_tiles
        else ("fixed_grid" if use_fixed_grid_tiles else "theoretical_center_crop")
    )

    raw_tiles = []
    aligned_global_tiles = []
    aligned_tiles = []
    aligned_global_preview_tiles = []
    aligned_preview_tiles = []
    aligned_global_context_tiles = []
    aligned_context_tiles = []
    raw_tiles_valid = []
    aligned_global_tiles_valid = []
    aligned_tiles_valid = []
    valid_mask_tiles = []
    metrics = []

    for branch_idx in range(simulator.num_branches):
        branch_center = branch_centers[branch_idx]
        global_margin_y = int(torch.ceil(residual_shifts_rgb[branch_idx, :, 0].abs().max()).item())
        global_margin_x = int(torch.ceil(residual_shifts_rgb[branch_idx, :, 1].abs().max()).item())
        extraction_margin_y, extraction_margin_x = branch_alignment_crop_margins_yx[branch_idx]

        if tile_extraction_mode == "centered_guard_crop":
            expanded_tile = _crop_centered_window(
                sensor,
                center_y=float(branch_center[0].item()),
                center_x=float(branch_center[1].item()),
                out_h=tile_h + 2 * extraction_margin_y,
                out_w=tile_w + 2 * extraction_margin_x,
            )
            tile = _crop_center_tensor(expanded_tile, (tile_h, tile_w))
            aligned_global_expanded = _align_tile_rgb(expanded_tile, residual_shifts_rgb[branch_idx])
            aligned_expanded, local_alignment_info, _ = _apply_local_block_correction(
                aligned_global_expanded,
                margin_y=extraction_margin_y,
                margin_x=extraction_margin_x,
                grid_shape=local_grid_shape,
                patch_size=local_patch_size,
                search_radius=local_search_radius,
                min_gradient_std=local_min_gradient_std,
                min_correlation=local_min_correlation,
                enabled=enable_local_block_correction,
            )
            aligned_global_tile = _crop_center_tensor(aligned_global_expanded, (tile_h, tile_w))
            aligned_tile = _crop_center_tensor(aligned_expanded, (tile_h, tile_w))
            final_margin_y = 0
            final_margin_x = 0
            global_valid_mask = torch.ones_like(tile)
            final_valid_mask = torch.ones_like(tile)
            aligned_global_preview_tile = aligned_global_tile
            aligned_preview_tile = aligned_tile
            aligned_global_masked_tile = aligned_global_tile
            aligned_masked_tile = aligned_tile
            tile_valid = tile
            aligned_global_tile_valid = aligned_global_tile
            aligned_tile_valid = aligned_tile
        else:
            if tile_extraction_mode == "fixed_grid":
                tile = _extract_sensor_grid_tile(
                    sensor,
                    tile_h=tile_h,
                    tile_w=tile_w,
                    branch_idx=branch_idx,
                    grid_shape=simulator.config.grid_shape,
                )
            else:
                tile = _crop_centered_window(
                    sensor,
                    center_y=float(branch_center[0].item()),
                    center_x=float(branch_center[1].item()),
                    out_h=tile_h,
                    out_w=tile_w,
                )
            aligned_global_tile = _align_tile_rgb(tile, residual_shifts_rgb[branch_idx])
            aligned_tile, local_alignment_info, final_margin_yx = _apply_local_block_correction(
                aligned_global_tile,
                margin_y=global_margin_y,
                margin_x=global_margin_x,
                grid_shape=local_grid_shape,
                patch_size=local_patch_size,
                search_radius=local_search_radius,
                min_gradient_std=local_min_gradient_std,
                min_correlation=local_min_correlation,
                enabled=enable_local_block_correction,
            )
            final_margin_y, final_margin_x = final_margin_yx
            global_valid_mask = _rectangular_valid_mask(
                tile.shape[-2:],
                margin_y=global_margin_y,
                margin_x=global_margin_x,
                device=tile.device,
                dtype=tile.dtype,
            ).expand_as(tile)
            final_valid_mask = _rectangular_valid_mask(
                tile.shape[-2:],
                margin_y=final_margin_y,
                margin_x=final_margin_x,
                device=tile.device,
                dtype=tile.dtype,
            ).expand_as(tile)
            aligned_global_preview_tile = _composite_aligned_preview(tile, aligned_global_tile, global_valid_mask)
            aligned_preview_tile = _composite_aligned_preview(tile, aligned_tile, final_valid_mask)
            aligned_global_masked_tile = aligned_global_tile * global_valid_mask
            aligned_masked_tile = aligned_tile * final_valid_mask
            tile_valid = _crop_valid_region(tile, final_margin_y, final_margin_x)
            aligned_global_tile_valid = _crop_valid_region(aligned_global_tile, final_margin_y, final_margin_x)
            aligned_tile_valid = _crop_valid_region(aligned_tile, final_margin_y, final_margin_x)

        raw_tiles.append(tile)
        aligned_global_tiles.append(aligned_global_masked_tile)
        aligned_tiles.append(aligned_masked_tile)
        aligned_global_preview_tiles.append(aligned_global_masked_tile)
        aligned_preview_tiles.append(aligned_masked_tile)
        aligned_global_context_tiles.append(aligned_global_preview_tile)
        aligned_context_tiles.append(aligned_preview_tile)
        raw_tiles_valid.append(tile_valid)
        aligned_global_tiles_valid.append(aligned_global_tile_valid)
        aligned_tiles_valid.append(aligned_tile_valid)
        valid_mask_tiles.append(final_valid_mask)

        if save_branch_previews:
            raw_preview = simulator.linear_to_srgb(_normalize_for_preview(tile))
            aligned_global_preview = simulator.linear_to_srgb(_normalize_for_preview(aligned_global_tile))
            aligned_preview = simulator.linear_to_srgb(_normalize_for_preview(aligned_tile))
            raw_preview_valid = simulator.linear_to_srgb(_normalize_for_preview(tile_valid))
            aligned_global_preview_valid = simulator.linear_to_srgb(_normalize_for_preview(aligned_global_tile_valid))
            aligned_preview_valid = simulator.linear_to_srgb(_normalize_for_preview(aligned_tile_valid))
            _save_tensor_image(raw_preview, os.path.join(output_dir, f"branch_{branch_idx:02d}_raw.png"))
            _save_tensor_image(
                aligned_global_preview,
                os.path.join(output_dir, f"branch_{branch_idx:02d}_aligned_global.png"),
            )
            _save_tensor_image(aligned_preview, os.path.join(output_dir, f"branch_{branch_idx:02d}_aligned.png"))
            _save_tensor_image(raw_preview_valid, os.path.join(output_dir, f"branch_{branch_idx:02d}_raw_valid.png"))
            _save_tensor_image(
                aligned_global_preview_valid,
                os.path.join(output_dir, f"branch_{branch_idx:02d}_aligned_global_valid.png"),
            )
            _save_tensor_image(aligned_preview_valid, os.path.join(output_dir, f"branch_{branch_idx:02d}_aligned_valid.png"))

        metrics.append(
            {
                "branch_index": branch_idx,
                "crop_center_yx": [float(branch_center[0].item()), float(branch_center[1].item())],
                "residual_shifts_yx_pixels": residual_shifts_rgb[branch_idx].cpu().tolist(),
                "alignment_crop_margin_yx": [extraction_margin_y, extraction_margin_x],
                "global_valid_margin_yx": [global_margin_y, global_margin_x],
                "final_valid_margin_yx": [final_margin_y, final_margin_x],
                "local_block_correction": local_alignment_info,
                "edge_alignment_before": _edge_alignment_score(tile_valid),
                "edge_alignment_after_global": _edge_alignment_score(aligned_global_tile_valid),
                "edge_alignment_after": _edge_alignment_score(aligned_tile_valid),
                "tenengrad_before": _tenengrad_sharpness(tile_valid),
                "tenengrad_after_global": _tenengrad_sharpness(aligned_global_tile_valid),
                "tenengrad_after": _tenengrad_sharpness(aligned_tile_valid),
                "edge_chroma_before": _edge_chroma_energy(tile_valid),
                "edge_chroma_after_global": _edge_chroma_energy(aligned_global_tile_valid),
                "edge_chroma_after": _edge_chroma_energy(aligned_tile_valid),
            }
        )

    common_valid_shape = (
        min(tile.shape[-2] for tile in raw_tiles_valid + aligned_global_tiles_valid + aligned_tiles_valid),
        min(tile.shape[-1] for tile in raw_tiles_valid + aligned_global_tiles_valid + aligned_tiles_valid),
    )
    raw_tiles_valid_common = _crop_tiles_to_common_shape(raw_tiles_valid, common_valid_shape)
    aligned_global_tiles_valid_common = _crop_tiles_to_common_shape(aligned_global_tiles_valid, common_valid_shape)
    aligned_tiles_valid_common = _crop_tiles_to_common_shape(aligned_tiles_valid, common_valid_shape)

    raw_grid = _make_tile_grid(raw_tiles)
    aligned_global_grid = _make_tile_grid(aligned_global_preview_tiles)
    aligned_grid = _make_tile_grid(aligned_preview_tiles)
    aligned_global_context_grid = _make_tile_grid(aligned_global_context_tiles)
    aligned_context_grid = _make_tile_grid(aligned_context_tiles)
    raw_grid_valid = _make_tile_grid(raw_tiles_valid_common)
    aligned_global_grid_valid = _make_tile_grid(aligned_global_tiles_valid_common)
    aligned_grid_valid = _make_tile_grid(aligned_tiles_valid_common)
    valid_mask_grid = _make_tile_grid(valid_mask_tiles)
    raw_grid_preview = simulator.linear_to_srgb(_normalize_for_preview(raw_grid))
    aligned_global_grid_preview = simulator.linear_to_srgb(_normalize_for_preview(aligned_global_grid))
    aligned_grid_preview = simulator.linear_to_srgb(_normalize_for_preview(aligned_grid))
    aligned_global_context_grid_preview = simulator.linear_to_srgb(_normalize_for_preview(aligned_global_context_grid))
    aligned_context_grid_preview = simulator.linear_to_srgb(_normalize_for_preview(aligned_context_grid))
    raw_grid_valid_preview = simulator.linear_to_srgb(_normalize_for_preview(raw_grid_valid))
    aligned_global_grid_valid_preview = simulator.linear_to_srgb(_normalize_for_preview(aligned_global_grid_valid))
    aligned_grid_valid_preview = simulator.linear_to_srgb(_normalize_for_preview(aligned_grid_valid))
    valid_mask_grid_preview = valid_mask_grid.clamp(0.0, 1.0)
    _save_tensor_image(raw_grid_preview, os.path.join(output_dir, "tile_grid_raw.png"))
    _save_tensor_image(aligned_grid_preview, os.path.join(output_dir, "tile_grid_aligned.png"))

    summary = {
        "result_dir": result_dir,
        "tile_size": [tile_h, tile_w],
        "sensor_shape": [sensor_h, sensor_w],
        "tile_extraction_mode": tile_extraction_mode,
        "full_size_alignment_supported_without_post_crop": bool(tile_extraction_mode == "centered_guard_crop"),
        "design_wavelength_m": float(design_wavelength),
        "wavelengths_m": [float(x) for x in wavelengths],
        "alignment_parameter_source": alignment_params["source"],
        "used_saved_branch_shifts": bool(alignment_params["used_saved_branch_shifts"]),
        "used_saved_residual_shifts": bool(alignment_params["used_saved_residual_shifts"]),
        "used_saved_wavelengths": bool(alignment_params["used_saved_wavelengths"]),
        "used_saved_design_wavelength": bool(alignment_params["used_saved_design_wavelength"]),
        "design_wavelength_inferred_from_saved_residuals": bool(
            design_wavelength_inferred_from_saved_residuals
        ),
        "local_block_correction": {
            "enabled": bool(enable_local_block_correction),
            "grid_shape": [int(local_grid_shape[0]), int(local_grid_shape[1])],
            "patch_size": int(local_patch_size),
            "search_radius": int(local_search_radius),
            "min_gradient_std": float(local_min_gradient_std),
            "min_correlation": float(local_min_correlation),
        },
        "save_branch_previews": bool(save_branch_previews),
        "note": (
            "This validation first applies per-branch RGB affine translations using the saved branch layout "
            "metadata when available, then optionally estimates a low-resolution local residual shift field "
            "for the non-green channels using block-wise edge correlation. When the sensor tensor includes enough "
            "guard band around the branch centers, the script uses expanded branch-centered crops, aligns RGB there, "
            "and crops back to the original tile size so no post-alignment border is lost. Otherwise it falls back "
            "to the older strict common-overlap handling, where *_valid.png is smaller because only the shared valid "
            "support remains."
        ),
        "branch_metrics": metrics,
        "mean_edge_alignment_before": sum(item["edge_alignment_before"] for item in metrics) / len(metrics),
        "mean_edge_alignment_after_global": (
            sum(item["edge_alignment_after_global"] for item in metrics) / len(metrics)
        ),
        "mean_edge_alignment_after": sum(item["edge_alignment_after"] for item in metrics) / len(metrics),
        "mean_tenengrad_before": sum(item["tenengrad_before"] for item in metrics) / len(metrics),
        "mean_tenengrad_after_global": sum(item["tenengrad_after_global"] for item in metrics) / len(metrics),
        "mean_tenengrad_after": sum(item["tenengrad_after"] for item in metrics) / len(metrics),
        "mean_edge_chroma_before": sum(item["edge_chroma_before"] for item in metrics) / len(metrics),
        "mean_edge_chroma_after_global": sum(item["edge_chroma_after_global"] for item in metrics) / len(metrics),
        "mean_edge_chroma_after": sum(item["edge_chroma_after"] for item in metrics) / len(metrics),
    }

    with open(os.path.join(output_dir, "alignment_summary.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False, allow_unicode=True)

    print(f"Saved validation outputs to: {output_dir}")
    print(
        "Mean edge-alignment score: "
        f"{summary['mean_edge_alignment_before']:.4f} -> "
        f"{summary['mean_edge_alignment_after_global']:.4f} -> "
        f"{summary['mean_edge_alignment_after']:.4f}"
    )
    print(
        "Mean Tenengrad sharpness: "
        f"{summary['mean_tenengrad_before']:.6f} -> "
        f"{summary['mean_tenengrad_after_global']:.6f} -> "
        f"{summary['mean_tenengrad_after']:.6f}"
    )
    print(
        "Mean edge chroma energy: "
        f"{summary['mean_edge_chroma_before']:.6f} -> "
        f"{summary['mean_edge_chroma_after_global']:.6f} -> "
        f"{summary['mean_edge_chroma_after']:.6f}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate branch-wise RGB pre-alignment for chromatic dispersion.")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "HDRDOESimulator.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--result_dir",
        default="",
        help="Single simulation output directory containing sensor_mosaic_linear.pt and scene_input_linear.pt.",
    )
    parser.add_argument(
        "--results_root",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs_sensor"),
        help="Root directory to scan for RunHDRDOESimulation outputs when --result_dir is not given.",
    )
    parser.add_argument(
        "--result_name",
        default="",
        help="Optional folder name filter under --results_root, e.g. 0010.",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Optional output directory. Only used for single-directory mode.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Optional device override: cpu/cuda/auto.",
    )
    parser.add_argument(
        "--disable_local_block_correction",
        action="store_true",
        help="Disable the local residual block correction stage after the theoretical global pre-alignment.",
    )
    parser.add_argument(
        "--local_grid_rows",
        type=int,
        default=4,
        help="Number of block rows used to estimate the local residual shift field.",
    )
    parser.add_argument(
        "--local_grid_cols",
        type=int,
        default=6,
        help="Number of block columns used to estimate the local residual shift field.",
    )
    parser.add_argument(
        "--local_patch_size",
        type=int,
        default=31,
        help="Odd patch size used for local block matching.",
    )
    parser.add_argument(
        "--local_search_radius",
        type=int,
        default=3,
        help="Maximum integer residual shift searched per block after global alignment.",
    )
    parser.add_argument(
        "--local_min_gradient_std",
        type=float,
        default=1e-4,
        help="Minimum reference edge-texture standard deviation required to trust a local block match.",
    )
    parser.add_argument(
        "--local_min_correlation",
        type=float,
        default=0.15,
        help="Minimum block-wise edge correlation required before applying a local residual shift.",
    )
    parser.add_argument(
        "--save_branch_previews",
        action="store_true",
        help="Also save per-branch preview images in addition to the grid images and summary YAML.",
    )
    args = parser.parse_args()

    _, simulator_config = load_yaml_configs(args.config)
    if args.device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = args.device

    simulator = HDRDOESimulator(simulator_config, device=resolved_device)

    if args.result_dir:
        _process_result_dir(
            simulator,
            args.result_dir,
            args.output_dir,
            enable_local_block_correction=not args.disable_local_block_correction,
            local_grid_shape=(args.local_grid_rows, args.local_grid_cols),
            local_patch_size=args.local_patch_size,
            local_search_radius=args.local_search_radius,
            local_min_gradient_std=args.local_min_gradient_std,
            local_min_correlation=args.local_min_correlation,
            save_branch_previews=args.save_branch_previews,
        )
        return

    result_dirs = _find_result_dirs(args.results_root, args.result_name)
    if not result_dirs:
        raise FileNotFoundError(
            f"No simulation result directories found in {args.results_root} "
            f"with result_name={args.result_name!r}."
        )

    print(f"Found {len(result_dirs)} result directorie(s) to validate.")
    for result_dir in result_dirs:
        print(f"Processing: {result_dir}")
        _process_result_dir(
            simulator,
            result_dir,
            "",
            enable_local_block_correction=not args.disable_local_block_correction,
            local_grid_shape=(args.local_grid_rows, args.local_grid_cols),
            local_patch_size=args.local_patch_size,
            local_search_radius=args.local_search_radius,
            local_min_gradient_std=args.local_min_gradient_std,
            local_min_correlation=args.local_min_correlation,
            save_branch_previews=args.save_branch_previews,
        )


if __name__ == "__main__":
    main()
