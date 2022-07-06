import argparse
import Dataset
import torch
import cv2
from config import update_config
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from model.conv3D import conv3D
from model.layer import aothead, mihead, mbphead, mdhead
from utils import *
from loss import *


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
    detection_model = torch.hub.load('ultralytics/yolov3', 'yolov3')
    detection_model.conf = 0.5 if train_cfg.dataset == 'Ped2' else 0.8
    md_lambda = 0.5 if train_cfg.dataset == 'Ped2' else 0.5

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
        for indice, video_clips, random_clips in train_dataloader:
            

            object_detect_results = detection_model(video_clips[2][0])
            
            for pos_list in object_detect_results.xyxy[0]:
                pos_list = pos_list.int()
                read_img = cv2.imread(video_clips[2][0])
                crop_img = cv2.resize(read_img[pos_list[1]:pos_list[3], pos_list[0]:pos_list[2]], (64, 64))


except:
