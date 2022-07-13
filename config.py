from glob import glob
import os

if not os.path.exists('tensorboard_log'):
    os.mkdir('tensorboard_log')
if not os.path.exists('weights'):
    os.mkdir('weights')

share_config = {
    'mode': 'training',
    'dataset': 'avenue',
    'img_size': (64, 64),
    'data_root': 'Data/'
}


class dict2class:
    def __init__(self, config):
        for k, v in config.items():
            self.__setattr__(k, v)
        
    def print_cfg(self):
        print('\n' + '-' * 30 + f'{self.mode} cfg' + '-' * 30)
        for k, v in vars(self).items():
            print(f'{k}: {v}')
        print()


def update_config(args=None, mode=None):
    share_config['mode'] = mode
    assert args.dataset in ('ped2', 'avenue', 'shanghaitech'), 'Dataset error! Check Dataset argument'
    share_config['dataset'] = args.dataset

    if mode == 'train':
        share_config['batch_size'] = args.batch_size
        share_config['train_data'] = share_config['data_root'] + args.dataset + '/training'
        share_config['test_data'] = share_config['data_root'] + args.dataset + '/testing'
        share_config['lr'] = 0.001
        share_config['level'] = args.level
        share_config['width'] = args.width
        share_config['depth'] = args.depth
        share_config['iters'] = args.iters
        share_config['resume'] = glob(f'weights/{args.resume}*')[0] if args.resume else None
        share_config['save_interval'] = args.save_interval
        share_config['val_interval'] = args.val_interval

    elif mode == 'test':
        share_config['test_data'] = share_config['data_root'] + args.dataset + '/testing/'
        share_config['trained_model'] = args.trained_model
        share_config['level'] = args.level
        share_config['width'] = share_config['trained_model'].split('_')[1]
        share_config['depth'] = share_config['trained_model'].split('_')[2]

    return dict2class(share_config)