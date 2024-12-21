import yaml
from pathlib import Path
import torch

class Config:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        # Set base directory
        self.BASE_DIR = Path(__file__).parent.parent.parent

        # Load and set all configurations
        self._set_attributes()

        # Set device
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _set_attributes(self):
        # Training parameters
        self.BATCH_SIZE = self._config['training']['batch_size']
        self.NUM_EPOCHS = self._config['training']['num_epochs']
        self.LEARNING_RATE = self._config['training']['learning_rate']
        self.SAVE_FREQ = self._config['training']['save_freq']
        self.PRINT_FREQ = self._config['training']['print_freq']

        # Model parameters
        self.NUM_CLASSES = self._config['model']['num_classes']

        # Metrics parameters
        self.TARGET_ACCURACY = self._config['metrics']['target_accuracy']

        # Paths
        self.CHECKPOINT_DIR = self.BASE_DIR / self._config['paths']['checkpoint_dir']
        self.DATA_DIR = self.BASE_DIR / self._config['paths']['data_dir']
        self.LOGS_DIR = self.BASE_DIR / self._config['paths']['logs_dir']
        self.CHECKPOINT_FILE = self._config['model']['checkpoint_file']

# Create a global config instance
config = Config()
