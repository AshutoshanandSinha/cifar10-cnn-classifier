import os
from ..utils.config import config

def setup_directories():
    dirs = [
        config.CHECKPOINT_DIR,
        config.LOGS_DIR,
        config.DATA_DIR
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
