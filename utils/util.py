import os
import shutil
from pytorch_lightning.callbacks import Callback

class LogManager(Callback):   #一个回调，用于管理训练日志和检查点
    def __init__(self):
        super().__init__()
        self.ckpt_path = None  # 用于存储用户选择的检查点路径

    def on_fit_start(self, trainer, pl_module):
        """
        在训练开始前检查日志和检查点，决定是否删除、恢复或退出。
        """
        ckpt_dir = os.path.join(trainer.logger.log_dir, 'checkpoints')
        print("*" * 30)
        print('checkpoint_dir', ckpt_dir)
        print("*" * 30)

        if os.path.exists(ckpt_dir):
            ckpt_list = [filename for filename in os.listdir(ckpt_dir) if '.ckpt' in filename]
            print(f'Log and the following checkpoint exists:\n Log dir: {trainer.logger.log_dir}\n' + '\n'.join(
                f'[{i}] {filename}' for i, filename in enumerate(ckpt_list)))

            delete = ['delete', 'd']
            resume = ['resume', 'r']
            quit = ['quit', 'q']
            ans = ''
            n_files = len(ckpt_list)
            while not (ans in delete or ans in resume or ans in quit):
                ans = input(f'[Number of ckpt files: {n_files}]\n'
                            f'Delete the existing log and start a new experiment? Resume? Quit? (d/r/q) << ').lower()
                if ans in delete:
                    shutil.rmtree(trainer.logger.log_dir)
                    os.makedirs(trainer.logger.log_dir)
                elif ans in resume:
                    if n_files == 0:
                        print('Any checkpoint files do not exist!')
                        ans = ''
                    else:
                        s = ''
                        if n_files > 1:
                            while not (s.isdigit() and int(s) in range(n_files)):
                                s = input(f'Select which checkpoint to load. [0-{n_files - 1}]<< ').lower()
                            self.ckpt_path = os.path.join(ckpt_dir, ckpt_list[int(s)])
                        else:
                            self.ckpt_path = os.path.join(ckpt_dir, ckpt_list[0])
                        print(f"Selected checkpoint: {self.ckpt_path}")
                elif ans in quit:
                    raise ValueError('Stopped as the log exist for this experiment.')
        else:
            print(f'Starting a new experiment and logging at \n {os.path.expanduser(trainer.logger.log_dir)}')