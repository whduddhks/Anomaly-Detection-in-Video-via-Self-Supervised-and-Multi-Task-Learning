import argparse
from config import update_config


parser = argparse.ArgumentParser(description='baseline')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to train.')
parser.add_argument('--level', default='object', type=str, help='Determine level of the Input')
parser.add_argument('--width', default='narrow', type=str, help='Model width [narrow, wide]')
parser.add_argument('--depth', default='shallow', type=str, help='Model depth [shallow, deep]')
parser.add_argument('--iters', default=40000, type=int, help='The total iteration number.')
parser.add_argument('--resume', default=None, type=str,
                    help='The pre-trained model to resume training with, pass \'latest\' or the model name.')
parser.add_argument('--save_interval', default=1000, type=int, help='Save the model every [save_interval] iterations.')
parser.add_argument('--val_interval', default=1000, type=int,
                    help='Evaluate the model every [val_interval] iterations, pass -1 to disable.')

args = parser.parse_args()
train_cfg = update_config(args, mode='train')
train_cfg.print_cfg()


