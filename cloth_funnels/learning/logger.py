import argparse
import os
from typing import Optional
import wandb
from tianshou.utils import WandbLogger, TensorboardLogger

class EnvWandbLogger(WandbLogger):
    def __init__(self,
                 train_interval: int = 1000,
                 test_interval: int = 1,
                 update_interval: int = 1000,
                 save_interval: int = 1000,
                 write_flush: bool = True,
                 project: Optional[str] = None,
                 name: Optional[str] = None,
                 entity: Optional[str] = None,
                 run_id: Optional[str] = None,
                 config: Optional[argparse.Namespace] = None,
                 monitor_gym: bool = True,
                 mode='online',
                 ) -> None:
        self.train_interval = train_interval
        self.test_interval = test_interval
        self.update_interval = update_interval
        self.last_log_train_step = -1
        self.last_log_test_step = -1
        self.last_log_update_step = -1
        self.last_save_step = -1
        self.save_interval = save_interval
        self.write_flush = write_flush
        self.restored = False
        if project is None:
            project = os.getenv("WANDB_PROJECT", "tianshou")

        self.wandb_run = wandb.init(
            project=project,
            name=name,
            id=run_id,
            resume="allow",
            entity=entity,
            sync_tensorboard=True,
            monitor_gym=monitor_gym,
            config=config,  # type: ignore
            # 自定义
            mode=mode,
        ) if not wandb.run else wandb.run
        self.wandb_run._label(repo="tianshou")  # type: ignore
        self.tensorboard_logger: Optional[TensorboardLogger] = None