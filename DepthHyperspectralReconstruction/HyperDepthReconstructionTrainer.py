import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from lightning_fabric import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset
from HyperDepthReconstruction import DOEHyperDepthCamera
from dataset.syn_data import SynDataset

import psutil

def print_memory_usage(step_name=""):
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"{step_name} - Memory usage: {mem_info.rss / 1024 ** 2:.2f} MB")

# 设置随机种子
seed_everything(123)

# 数据准备函数
def prepare_data(args, image_sz):
    """准备训练和验证数据加载器"""
    train_dataset = SynDataset(data_dir="F:\Dataset\syn\syn", dataset='train', image_sz=(image_sz, image_sz), augment=True)
    val_dataset = SynDataset(data_dir="F:\Dataset\syn\syn", dataset='val', image_sz=(image_sz, image_sz), augment=True)

    # 创建 DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,  # 加速 GPU 数据传输
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,  # 验证集通常无需打乱
        pin_memory=True,
    )

    return train_dataloader, val_dataloader

# 主训练函数
def main():
    # 解析命令行参数
    print("开始")
    parser = ArgumentParser(add_help=True)
    parser = DOEHyperDepthCamera.add_model_specific_args(parser)  # 模型特定参数
    parser.add_argument('--experiment_name', type=str, default='DOEHyperDepthReconstruction')
    parser.add_argument('--default_root_dir', type=str, default='data/logs')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    # 初始化日志记录器
    logger = TensorBoardLogger(
        save_dir=args.default_root_dir,
        name=args.experiment_name,
        version=None,  # 自动生成版本号
    )

    # 定义检查点回调
    checkpoint_dir = os.path.join(logger.log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)  # 确保目录存在
    checkpoint = ModelCheckpoint(
        monitor="val_loss",                    # 监控验证损失
        dirpath=checkpoint_dir,                # 保存路径
        filename="best-{epoch:02d}-{val_loss:.2f}",  # 文件名模板
        save_top_k=1,                          # 保存 1 个最佳模型
        save_last=True,                        # 保存最后一个检查点
        mode="min",                            # 最小化 val_loss
        every_n_epochs=1,                      # 每个 epoch 保存
        verbose=True,                          # 打印保存信息
        save_weights_only=False,               # 保存完整状态
    )

    # 初始化 Trainer
    trainer = pl.Trainer(
        accelerator="gpu",                     # 使用 GPU
        devices=1,                             # 单 GPU
        max_epochs=50,                          # 训练 4 个 epoch
        logger=logger,                         # TensorBoard 日志
        callbacks=[checkpoint],                # 检查点回调
        default_root_dir=args.default_root_dir,# 默认根目录
        enable_progress_bar=True,              # 显示进度条
        log_every_n_steps=10,                  # 每10步记录日志
        num_sanity_val_steps=0
        # limit_train_batches=0.01,  # 使用全部训练批次（10个样本）
        # limit_val_batches=0.01  # 使用全部验证批次（10个样本）
    )

    # 初始化模型
    model = DOEHyperDepthCamera(args)
    image_sz = model.camera.image_sz

    # 准备数据
    train_dataloader, val_dataloader = prepare_data(args, image_sz)

    # 开始训练
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

if __name__ == "__main__":
    main()