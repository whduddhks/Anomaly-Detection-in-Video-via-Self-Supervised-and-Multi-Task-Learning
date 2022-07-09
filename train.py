import argparse
import Dataset
import torch
import cv2
import os
import sys
from pathlib import Path
from config import update_config
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from model.conv3D import conv3D
from model.layer import aothead, mihead, mbphead, mdhead
from conv_utils import *
from loss import *
from yolov3.detect_loc import run

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


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

# model 불러오기 또는 초기화
shared_conv = conv3D(train_cfg.width).cuda()
aot_head = aothead(train_cfg.width).cuda()
mi_head = mihead(train_cfg.width).cuda()
mbp_head = mbphead(train_cfg.width).cuda()
md_head = mdhead(train_cfg.width).cuda()

optimizer_shared = torch.optim.Adam(shared_conv.parameters(), lr=train_cfg.lr)
optimizer_aot = torch.optim.Adam(aot_head.parameters(), lr=train_cfg.lr)
optimizer_mi = torch.optim.Adam(mi_head.parameters(), lr=train_cfg.lr)
optimizer_mbp = torch.optim.Adam(mbp_head.parameters(), lr=train_cfg.lr)
optimizer_md = torch.optim.Adam(md_head.parameters(), lr=train_cfg.lr)

if train_cfg.resume:
    shared_conv.load_state_dict(torch.load(train_cfg.resume)['shared'])
    aot_head.load_state_dict(torch.load(train_cfg.resume)['aot'])
    mi_head.load_state_dict(torch.load(train_cfg.resume)['mi'])
    mbp_head.load_state_dict(torch.load(train_cfg.resume)['mbp'])
    md_head.load_state_dict(torch.load(train_cfg.resume)['md'])

    optimizer_shared.load_state_dict(torch.load(train_cfg.resume)['opt_shared'])
    optimizer_aot.load_state_dict(torch.load(train_cfg.resume)['opt_aot'])
    optimizer_mi.load_state_dict(torch.load(train_cfg.resume)['opt_mi'])
    optimizer_mbp.load_state_dict(torch.load(train_cfg.resume)['opt_mbp'])
    optimizer_md.load_state_dict(torch.load(train_cfg.resume)['opt_md'])
else:
    shared_conv.apply(weights_init_normal)
    aot_head.apply(weights_init_normal)
    mi_head.apply(weights_init_normal)
    mbp_head.apply(weights_init_normal)
    md_head.apply(weights_init_normal)

# loss
aot_loss = aotloss().cuda()
mi_loss = miloss().cuda()
mbp_loss = mbploss().cuda()
md_loss =mdloss().cuda()

# Yolo v3 model 불러오기
if train_cfg.level == 'object':
    conf_thres = 0.5 if train_cfg.dataset == 'ped2' else 0.8
    md_lambda = 0.5 if train_cfg.dataset == 'ped2' else 0.2

# Dataloader 정의
train_dataset = Dataset.train_dataset(train_cfg)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_cfg.batch_size,
                              shuffle=True, num_workers=0)

writer = SummaryWriter(f'tensorboard_log/{train_cfg.dataset}_{train_cfg.width}_{train_cfg.depth}')
start_iter = int(train_cfg.resume.split('_')[-1].split('.')[0]) if train_cfg.resume else 0
training = True

shared_conv = shared_conv.train()
aot_head = aot_head.train()
mi_head = mi_head.train()
mbp_head = mbp_head.train()
md_head = md_head.train()

try:
    step = start_iter
    while training:
        for indice, video_clips, random_clips, path_list in train_dataloader:
            
            pred, yolo_cls_prob = run(weights=ROOT / 'yolov3/yolov3.pt', source=path_list[0], imgsz=video_clips.shape[2:4], conf_thres=conf_thres)
                
            video_input_crop = img_crop(video_clips[0], pred[0])
            random_input_crop = img_crop(random_clips[0], pred[0])
            
            video_input_crop = torch.from_numpy(np.array(video_input_crop))
            random_input_crop = torch.from_numpy(np.array(random_input_crop))
            
            aot_input = torch.cat([video_input_crop.clone().detach().requires_grad_(True), torch.flip(video_input_crop.clone().detach().requires_grad_(True), [0, 1])], 0)
            aot_shape = aot_input.shape
            aot_input = aot_input.reshape(aot_shape[0], -1, aot_shape[1], aot_shape[2], aot_shape[3]).cuda()
            aot_target = torch.cat([torch.zeros([video_input_crop.shape[0]]), torch.ones([video_input_crop.shape[0]])], 0).long().cuda()
            
            mi_input = torch.cat([video_input_crop.clone().detach().requires_grad_(True), random_input_crop.clone().detach().requires_grad_(True)], 0)
            mi_shape = mi_input.shape
            mi_input = mi_input.reshape(mi_shape[0], -1, mi_shape[1], mi_shape[2], mi_shape[3]).cuda()
            mi_target = torch.cat([torch.zeros([video_input_crop.shape[0]]), torch.ones([video_input_crop.shape[0]])], 0).long().cuda()
            
            mbp_input = torch.cat([video_input_crop[:, :3, :].clone().detach().requires_grad_(True), video_input_crop[:, 4:, :].clone().detach().requires_grad_(True)], 1)
            mbp_shape = mbp_input.shape
            mbp_input = mbp_input.reshape(mbp_shape[0], -1, mbp_shape[1], mbp_shape[2], mbp_shape[3]).cuda()
            mbp_target = video_input_crop[:, 3, :].clone().detach().requires_grad_(True)
            mbp_target_shape =  mbp_target.shape
            mbp_target = mbp_target.reshape(mbp_target_shape[0], -1, mbp_target_shape[1],mbp_target_shape[2]).cuda()
            
            md_input = video_input_crop[:, 3, :].clone().detach().requires_grad_(True).unsqueeze(dim=1)
            md_shape = md_input.shape
            md_input = md_input.reshape(md_shape[0], -1, md_shape[1], md_shape[2], md_shape[3]).cuda()
            
            aot_shared = shared_conv(aot_input, train_cfg.depth)
            mi_shared = shared_conv(mi_input, train_cfg.depth)
            mbp_shared = shared_conv(mbp_input, train_cfg.depth)
            md_shared = shared_conv(md_input, train_cfg.depth)
            
            
            aot_shared = aot_shared.squeeze()
            mi_shared = mi_shared.squeeze()
            mbp_shared = mbp_shared.squeeze()
            md_shared = md_shared.squeeze()

            aot_output = aot_head(aot_shared, train_cfg.depth)
            mi_output = mi_head(mi_shared, train_cfg.depth)
            mbp_output = mbp_head(mbp_shared, train_cfg.depth)
            md_output_res, md_output_yolo = md_head(md_shared, train_cfg.depth)
            

            aot_l = aot_loss(aot_output, aot_target)
            mi_l = mi_loss(mi_output, mi_target)
            mbp_l = mbp_loss(mbp_output, mbp_target)
            # md_l = aot_loss(md_output_res, md_output_yolo, yolo_cls_prob)
            
            
            

            break
            

        break

except KeyboardInterrupt:
    print('a')