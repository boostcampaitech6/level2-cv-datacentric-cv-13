import argparse
import os
from torch import cuda

class Parser(object):

    def __init__(self, description=""):
        self.description = description
        self.parser = argparse.ArgumentParser(description=self.description)
    
    def create_parser(self):
        self.parser.add_argument('--config', default='./config.yml', help='Path to the configuration file in YAML format.')
        self.parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', './data/medical'))
        self.parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                            'trained_models'))

        self.parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
        self.parser.add_argument('--num_workers', type=int, default=8)

        self.parser.add_argument('--image_size', type=int, default=2048)
        self.parser.add_argument('--input_size', type=int, default=1024)
        self.parser.add_argument('--batch_size', type=int, default=8)
        self.parser.add_argument('--learning_rate', type=float, default=1e-3)
        self.parser.add_argument('--max_epoch', type=int, default=150)
        self.parser.add_argument('--save_interval', type=int, default=5)
        self.parser.add_argument('--ignore_tags', type=list, default=['masked', 'excluded-region', 'maintable', 'stamp'])
        self.parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions'))
        self.parser.add_argument('--transform', type=str, default="BaseTransform")
        self.parser.add_argument('--exp_name', type=str)

    def print_args(self, args):
        print("Arguments:")
        for arg in vars(args):
            print("\t{}: {}".format(arg, getattr(args, arg)))