import argparse
import os
from typing import List, Optional, Tuple

import torch

from HDRDOESimulator import HDRDOESimulator, HDRDOERunConfig, load_yaml_configs


SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".exr")
SUPPORTED_DEPTH_SUFFIXES = (".npy", ".pt", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr")


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _list_input_images(input_dir: str) -> List[str]:
    if not os.path.isdir(input_dir):
        return []
    image_paths = []
    for name in sorted(os.listdir(input_dir)):
        suffix = os.path.splitext(name)[1].lower()
        if suffix in SUPPORTED_IMAGE_SUFFIXES:
            image_paths.append(os.path.join(input_dir, name))
    return image_paths


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


def _build_output_dir(base_output_dir: str, image_path: str) -> str:
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(base_output_dir, image_name)


def _run_single_image(
    simulator: HDRDOESimulator,
    run_config: HDRDOERunConfig,
    image_path: str,
    output_dir: str,
) -> None:
    depth_path = run_config.depth_path if run_config.depth_path else _find_matching_depth(image_path)
    tile_size = None
    if not run_config.same_sensor_size and run_config.tile_height > 0 and run_config.tile_width > 0:
        tile_size = (run_config.tile_height, run_config.tile_width)

    print(f"Running simulation for: {image_path}")
    if depth_path is not None:
        print(f"Using depth map: {depth_path}")

    with torch.no_grad():
        result = simulator.simulate_from_path(image_path=image_path, depth_path=depth_path, tile_size=tile_size)

    simulator.save_result(result, output_dir)
    print(f"Saved outputs to: {output_dir}")
    print(f"Tile size: {result.tile_size}, mosaic shape: {tuple(result.mosaic_linear.shape)}")


def _run_synthetic(simulator: HDRDOESimulator, run_config: HDRDOERunConfig) -> None:
    print("No input image found. Falling back to a synthetic RGB+depth scene.")
    scene, depth = HDRDOESimulator.generate_synthetic_scene(
        height=run_config.synthetic_height,
        width=run_config.synthetic_width,
        device=simulator.device,
    )
    tile_size = None
    if not run_config.same_sensor_size and run_config.tile_height > 0 and run_config.tile_width > 0:
        tile_size = (run_config.tile_height, run_config.tile_width)

    with torch.no_grad():
        result = simulator.simulate(scene, depthmap=depth, tile_size=tile_size)

    output_dir = os.path.join(run_config.output_dir, "synthetic_scene")
    simulator.save_result(result, output_dir)
    print(f"Saved outputs to: {output_dir}")
    print(f"Tile size: {result.tile_size}, mosaic shape: {tuple(result.mosaic_linear.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 2x2 lens+DOE HDR/depth simulator.")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "HDRDOESimulator.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument("--input_path", default=None, help="Optional single image path override.")
    parser.add_argument("--depth_path", default=None, help="Optional single depth map path override.")
    parser.add_argument("--output_dir", default=None, help="Optional output directory override.")
    parser.add_argument("--device", default=None, help="Optional device override: cpu/cuda/auto.")
    parser.add_argument("--no_noise", action="store_true", help="Disable shot noise for this run.")
    args = parser.parse_args()

    run_config, simulator_config = load_yaml_configs(args.config)

    if args.input_path is not None:
        run_config.input_path = args.input_path
    if args.depth_path is not None:
        run_config.depth_path = args.depth_path
    if args.output_dir is not None:
        run_config.output_dir = args.output_dir
    if args.device is not None:
        run_config.device = args.device
    if args.no_noise:
        run_config.no_noise = True
        simulator_config.add_shot_noise = False
        simulator_config.read_noise_std = 0.0
        simulator_config.dark_current = 0.0

    resolved_device = _resolve_device(run_config.device)
    simulator = HDRDOESimulator(simulator_config, device=resolved_device)

    os.makedirs(run_config.output_dir, exist_ok=True)

    if run_config.input_path:
        output_dir = _build_output_dir(run_config.output_dir, run_config.input_path)
        _run_single_image(simulator, run_config, run_config.input_path, output_dir)
        return

    image_paths = _list_input_images(run_config.input_dir)
    if image_paths:
        for image_path in image_paths:
            output_dir = _build_output_dir(run_config.output_dir, image_path)
            _run_single_image(simulator, run_config, image_path, output_dir)
        return

    _run_synthetic(simulator, run_config)


if __name__ == "__main__":
    main()
