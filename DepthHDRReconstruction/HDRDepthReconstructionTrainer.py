import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import yaml
from lightning_fabric import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from HDRDepthReconstruction import DOEHDRDepthCamera, prepare_data


def _resolve_accelerator(simulator_device: str, requested_accelerator: str) -> str:
    if requested_accelerator != "auto":
        return requested_accelerator
    if simulator_device == "cuda":
        return "gpu"
    return "cpu"


def main() -> None:
    parser = ArgumentParser(add_help=True)
    parser = DOEHDRDepthCamera.add_model_specific_args(parser)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto", help="auto/gpu/cpu")
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--save_top_k", type=int, default=1)
    parser.add_argument("--num_sanity_val_steps", type=int, default=0)
    parser.add_argument("--enable_progress_bar", action="store_true")
    parser.add_argument("--fast_dev_run", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)

    requested_accelerator = args.accelerator
    if requested_accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    else:
        accelerator = requested_accelerator

    resolved_simulator_device = args.simulator_device
    if resolved_simulator_device == "auto":
        resolved_simulator_device = "cuda" if accelerator == "gpu" else "cpu"
        args.simulator_device = resolved_simulator_device
    elif resolved_simulator_device == "cuda" and accelerator == "cpu":
        raise ValueError("simulator_device=cuda is incompatible with trainer accelerator=cpu.")

    logger = TensorBoardLogger(
        save_dir=args.default_root_dir,
        name=args.experiment_name,
        version=None,
    )

    run_dir = logger.log_dir
    os.makedirs(run_dir, exist_ok=True)
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath=run_dir,
        filename="best-epoch={epoch:02d}-val_loss={val_loss:.4f}",
        save_top_k=args.save_top_k,
        save_last=True,
        mode="min",
        every_n_epochs=1,
        auto_insert_metric_name=False,
    )

    model = DOEHDRDepthCamera(args, log_dir=logger.log_dir)
    train_dataloader, val_dataloader = prepare_data(args)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint],
        default_root_dir=args.default_root_dir,
        enable_progress_bar=args.enable_progress_bar or args.fast_dev_run,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=args.num_sanity_val_steps,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    model.export_current_doe(run_dir)

    artifacts = {
        "run_dir": run_dir,
        "experiment_name": args.experiment_name,
        "simulator_config": args.simulator_config,
        "optimize_optics": bool(args.optimize_optics),
        "trainer_accelerator": accelerator,
        "simulator_device": args.simulator_device,
        "checkpoint_last": os.path.join(run_dir, "last.ckpt"),
        "checkpoint_best": checkpoint.best_model_path,
        "doe_fabrication_png": os.path.join(run_dir, "doe_fabrication_heightmap_16bit.png"),
        "doe_fabrication_metadata": os.path.join(run_dir, "doe_fabrication_metadata.yaml"),
    }
    with open(os.path.join(run_dir, "training_artifacts.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(artifacts, handle, sort_keys=False, allow_unicode=True)

    print(f"Saved all training artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
