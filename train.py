import argparse
import Dataset
import torch
import os
import sys
import random
import time
import datetime
import glob

from loss import *
from conv_utils import *

from pathlib import Path
from config import update_config
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn

from model.conv3D import conv3D
from model.layer import aothead, mihead, mbphead, mdhead
from yolov3.detect_loc import run
from torchvision.models import resnet50, ResNet50_Weights


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
val_mdloss = valmdloss().cuda()

# Yolo v3 model 불러오기
if train_cfg.level == 'object':
    conf_thres = 0.5 if train_cfg.dataset == 'ped2' else 0.8
    md_lambda = 0.5 if train_cfg.dataset == 'ped2' else 0.2

weights_res = ResNet50_Weights.DEFAULT
res_model = resnet50(weights=weights_res).cuda()
res_model.eval()

preprocess_res = weights_res.transforms().cuda()


# Dataloader 정의
train_dataset = Dataset.train_dataset(train_cfg)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_cfg.batch_size,
                              shuffle=True, num_workers=4)

val_dataset = sorted(glob.glob(f'{train_cfg.train_data}/*'))
val_dataset_len = len(val_dataset)
val_dataset = val_dataset[int(val_dataset_len*0.85+1):]

writer = SummaryWriter(f'tensorboard_log/{train_cfg.dataset}_{train_cfg.width}_{train_cfg.depth}')
start_iter = int(train_cfg.resume.split('_')[-1].split('.')[0]) if train_cfg.resume else 0
training = True

shared_conv = shared_conv.train()
aot_head = aot_head.train()
mi_head = mi_head.train()
mbp_head = mbp_head.train()
md_head = md_head.train()

aot_save = (-1, -1)
mi_save = (-1, -1)
mbp_save = (-1, -1)
md_save = (-1, -1)

try:
    step = start_iter
    while training:
        for indice, video_clips, random_clips, path_list in train_dataloader:
            
            for index in indice:
                train_dataset.all_frames_training[index].pop()
                if len(train_dataset.all_frames_training[index]) == 0:
                    train_dataset.all_frames_training[index] = list(range(3, len(train_dataset.training_videos[index]) - 3))
                    random.shuffle(train_dataset.all_frames_training[index])
                    
            pred, yolo_cls_prob = run(weights=ROOT / 'yolov3/yolov3.pt', source=path_list[0], imgsz=video_clips.shape[2:4], conf_thres=conf_thres)
            if pred == -1:
                continue
                
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
            mbp_target = mbp_target.reshape(mbp_target_shape[0], -1, mbp_target_shape[1], mbp_target_shape[2]).cuda()
            
            md_input = video_input_crop[:, 3, :].clone().detach().requires_grad_(True).unsqueeze(dim=1)
            md_shape = md_input.shape
            md_input = md_input.reshape(md_shape[0], -1, md_shape[1], md_shape[2], md_shape[3]).cuda()
            md_res_input = video_input_crop[:, 3, :].clone().detach().requires_grad_(True).cuda()
            md_res_input = md_input.reshape(md_shape[0], -1, md_shape[2], md_shape[3])
            res_cls_prob = res_prob(res_model, preprocess_res, md_res_input)
            
            aot_shared = shared_conv(aot_input, train_cfg.depth)
            mi_shared = shared_conv(mi_input, train_cfg.depth)
            mbp_shared = shared_conv(mbp_input, train_cfg.depth)
            md_shared = shared_conv(md_input, train_cfg.depth)
            
            aot_shared = aot_shared.squeeze(dim=2)
            mi_shared = mi_shared.squeeze(dim=2)
            mbp_shared = mbp_shared.squeeze(dim=2)
            md_shared = md_shared.squeeze(dim=2)

            aot_output = aot_head(aot_shared, train_cfg.depth)
            mi_output = mi_head(mi_shared, train_cfg.depth)
            mbp_output = mbp_head(mbp_shared, train_cfg.depth)
            md_output_res, md_output_yolo = md_head(md_shared, train_cfg.depth)
            
            aot_l = aot_loss(aot_output, aot_target)
            mi_l = mi_loss(mi_output, mi_target)
            mbp_l = mbp_loss(mbp_output, mbp_target)
            md_l = md_loss(md_output_res, md_output_yolo, res_cls_prob, yolo_cls_prob)

            total_loss = aot_l + mi_l + mbp_l + md_lambda*md_l

            optimizer_aot.zero_grad()
            optimizer_mi.zero_grad()
            optimizer_mbp.zero_grad()
            optimizer_mi.zero_grad()
            optimizer_shared.zero_grad()

            total_loss.backward()

            optimizer_aot.step()
            optimizer_mi.step()
            optimizer_mbp.step()
            optimizer_mi.step()
            optimizer_shared.step()
            
            
            torch.cuda.synchronize()
            time_end = time.time()
            if step > start_iter:  # This doesn't include the testing time during training.
                iter_t = time_end - temp
            temp = time_end
            
            if step != start_iter:
                if step % 20 == 0:
                    time_remain = (train_cfg.iters - step) * iter_t
                    eta = str(datetime.timedelta(seconds=time_remain)).split('.')[0]

                    lr_shared = optimizer_shared.param_groups[0]['lr']
                    lr_aot = optimizer_aot.param_groups[0]['lr']
                    lr_mi = optimizer_mi.param_groups[0]['lr']
                    lr_mbp = optimizer_mbp.param_groups[0]['lr']
                    lr_md = optimizer_md.param_groups[0]['lr']

                    print(f"{step} | aot_l: {aot_l:.3f} | mi_l: {mi_l:.3f} | mbp_l: {mbp_l:.3f} | md_l: {md_l:.3f} | total: {total_loss:.3f} |"
                            f"iter: {iter_t:.3f}s | ETA: {eta} | lr_shared: {lr_shared} | lr_aot: {lr_aot} | lr_mi: {lr_mi} | lr_mbp: {lr_mbp} | lr_md: {lr_md}")

                if step % train_cfg.save_interval == 0:
                    model_dict = {  'shared': shared_conv.state_dict(),
                                    'aot': aot_head.state_dict(),
                                    'mi': mi_head.state_dict(),
                                    'mbp': mbp_head.state_dict(),
                                    'md': md_head.state_dict(),
                                    'opt_shared': optimizer_shared.state_dict(),
                                    'opt_aot': optimizer_aot.state_dict(),
                                    'opt_mi': optimizer_mi.state_dict(),
                                    'opt_mbp': optimizer_mbp.state_dict(),
                                    'opt_md': optimizer_md.state_dict()}
                    torch.save(model_dict, f'weights/{train_cfg.dataset}_{train_cfg.width}_{train_cfg.depth}_{step}.pth')
                    print(f'\nAlready saved: \'{train_cfg.dataset}_{train_cfg.width}_{train_cfg.depth}_{step}.pth\'.')

                if step % train_cfg.val_interval == 0:
                    shared_conv.eval()
                    aot_head.eval()
                    mi_head.eval()
                    mbp_head.eval()
                    md_head.eval()

                    fps = 0

                    aot_score = 0
                    mi_score = 0
                    mbp_score = 0
                    md_score = 0

                    with torch.no_grad():
                        for i, folder in enumerate(val_dataset):
                            val_data = Dataset.val_dataset(folder)

                            for j, (clip, i_path) in enumerate(val_data):
                                
                                pred, val_yolo_cls_prob = run(weights=ROOT / 'yolov3/yolov3.pt', source=i_path, imgsz=clip.shape[1:3], conf_thres=conf_thres)
                                if pred == -1:
                                    continue

                                val_input_crop = img_crop(clip, pred[0])
                                val_input_crop = torch.from_numpy(np.array(val_input_crop))
                                
                                val_aot_input = val_input_crop.clone().detach()
                                val_aot_shape = val_aot_input.shape
                                val_aot_input = val_aot_input.reshape(val_aot_shape[0], -1, val_aot_shape[1], val_aot_shape[2], val_aot_shape[3]).cuda()

                                val_mi_input = val_input_crop.clone().detach()
                                val_mi_shape = val_mi_input.shape
                                val_mi_input = val_mi_input.reshape(val_mi_shape[0], -1, val_mi_shape[1], val_mi_shape[2], val_mi_shape[3]).cuda()

                                val_mbp_input = torch.cat([val_input_crop[:, :3, :].clone().detach(), val_input_crop[:, 4:, :].clone().detach()], 1)
                                val_mbp_shape = val_mbp_input.shape
                                val_mbp_input = val_mbp_input.reshape(val_mbp_shape[0], -1, val_mbp_shape[1], val_mbp_shape[2], val_mbp_shape[3]).cuda()
                                val_mbp_target = val_input_crop[:, 3, :].clone().detach()
                                val_mbp_target_shape =  val_mbp_target.shape
                                val_mbp_target = val_mbp_target.reshape(val_mbp_target_shape[0], -1, val_mbp_target_shape[1], val_mbp_target_shape[2]).cuda()

                                val_md_input = val_input_crop[:, 3, :].clone().detach().unsqueeze(dim=1)
                                val_md_shape = val_md_input.shape
                                val_md_input = val_md_input.reshape(val_md_shape[0], -1, val_md_shape[1], val_md_shape[2], val_md_shape[3]).cuda()

                                val_aot_shared = shared_conv(val_aot_input, train_cfg.depth)
                                val_mi_shared = shared_conv(val_mi_input, train_cfg.depth)
                                val_mbp_shared = shared_conv(val_mbp_input, train_cfg.depth)
                                val_md_shared = shared_conv(val_md_input, train_cfg.depth)

                                val_aot_shared = val_aot_shared.squeeze(dim=2)
                                val_mi_shared = val_mi_shared.squeeze(dim=2)
                                val_mbp_shared = val_mbp_shared.squeeze(dim=2)
                                val_md_shared = val_md_shared.squeeze(dim=2)

                                softmax_loss = nn.Softmax(dim=1)
                                val_aot_output = aot_head(val_aot_shared, train_cfg.depth)
                                val_aot_output = softmax_loss(val_aot_output)
                                val_mi_output = mi_head(val_mi_shared, train_cfg.depth)
                                val_mi_output = softmax_loss(val_mi_output)
                                val_mbp_output = mbp_head(val_mbp_shared, train_cfg.depth)
                                val_md_output_res, val_md_output_yolo = md_head(val_md_shared, train_cfg.depth)

                                aot_score += torch.sum(val_aot_output[:, 1])
                                mi_score += torch.sum(val_mi_output[:, 1])
                                mbp_score += mbp_loss(val_mbp_output, val_mbp_target)
                                md_score += val_mdloss(val_md_output_yolo, val_yolo_cls_prob)

                    print(f"val | aot score: {aot_score:.3f} | mi score: {mi_score:.3f} | mbp score: {mbp_score:.3f} | md score: {md_score:.3f}")

                    if aot_save[0] == -1:
                        aot_save = (aot_score, step)
                        print(f'Save aot model | save: {aot_save[0]} | score: {aot_score}')
                    else:
                        if aot_save[0] < aot_score:
                            aot_head.load_state_dict(torch.load(f'weights/{train_cfg.dataset}_{train_cfg.width}_{train_cfg.depth}_{aot_save[1]}.pth')['aot'])
                            print(f'Load aot model from {aot_save[1]} step | save: {aot_save} | score: {aot_score}')
                        else:
                            aot_save = (aot_score, step)
                            print(f'Save aot model | save: {aot_save[0]} | score: {aot_score}')

                    if mi_save[0] == -1:
                        mi_save = (mi_score, step)
                        print(f'Save mi model | save: {mi_save[0]} | score: {mi_score}')
                    else:
                        if mi_save[0] < mi_score:
                            mi_head.load_state_dict(torch.load(f'weights/{train_cfg.dataset}_{train_cfg.width}_{train_cfg.depth}_{mi_save[1]}.pth')['mi'])
                            print(f'Load mi model from {mi_save[1]} step | save: {mi_save} | score: {mi_score}')
                        else:
                            mi_save = (mi_score, step)
                            print(f'Save mi model | save: {mi_save[0]} | score: {mi_score}')

                    if mbp_save[0] == -1:
                        mbp_save = (mbp_score, step)
                        print(f'Save mbp model | save: {mbp_save[0]} | score: {mbp_score}')
                    else:
                        if mbp_save[0] < mbp_score:
                            mbp_head.load_state_dict(torch.load(f'weights/{train_cfg.dataset}_{train_cfg.width}_{train_cfg.depth}_{mbp_save[1]}.pth')['mbp'])
                            print(f'Load mbp model from {mbp_save[1]} step | save: {mbp_save} | score: {mbp_score}')
                        else:
                            mbp_save = (mbp_score, step)
                            print(f'Save mbp model | save: {mbp_save[0]} | score: {mbp_score}')

                    if md_save[0] == -1:
                        md_save = (md_score, step)
                        print(f'Save md model | save: {md_save[0]} | score: {md_score}')
                    else:
                        if md_save[0] < md_score:
                            md_head.load_state_dict(torch.load(f'weights/{train_cfg.dataset}_{train_cfg.width}_{train_cfg.depth}_{md_save[1]}.pth')['md'])
                            print(f'Load md model from {md_save[1]} step | save: {md_save} | score: {md_score}')
                        else:
                            md_save = (md_score, step)
                            print(f'Save md model | save: {md_save[0]} | score: {md_score}')
                    

                    shared_conv.train()
                    aot_head.train()
                    mi_head.train()
                    mbp_head.train()
                    md_head.train()
                    print('')

            step += 1
            if step > train_cfg.iters:
                training = False
                model_dict = {  'shared': shared_conv.state_dict(),
                                'aot': aot_head.state_dict(),
                                'mi': mi_head.state_dict(),
                                'mbp': mbp_head.state_dict(),
                                'md': md_head.state_dict(),
                                'opt_shared': optimizer_shared.state_dict(),
                                'opt_aot': optimizer_aot.state_dict(),
                                'opt_mi': optimizer_mi.state_dict(),
                                'opt_mbp': optimizer_mbp.state_dict(),
                                'opt_md': optimizer_md.state_dict()}
                torch.save(model_dict, f'weights/{train_cfg.dataset}_{train_cfg.width}_{train_cfg.depth}_{step}.pth')
                break

except KeyboardInterrupt:
    print(f'\nStop early, model saved: \'latest_{train_cfg.dataset}_{train_cfg.width}_{train_cfg.depth}_{step}.pth\'.\n')

    if glob(f'weights/latest*'):
        os.remove(glob(f'weights/latest*')[0])
    
    model_dict = {  'shared': shared_conv.state_dict(),
                    'aot': aot_head.state_dict(),
                    'mi': mi_head.state_dict(),
                    'mbp': mbp_head.state_dict(),
                    'md': md_head.state_dict(),
                    'opt_shared': optimizer_shared.state_dict(),
                    'opt_aot': optimizer_aot.state_dict(),
                    'opt_mi': optimizer_mi.state_dict(),
                    'opt_mbp': optimizer_mbp.state_dict(),
                    'opt_md': optimizer_md.state_dict()}
    torch.save(model_dict, f'weights/latest_{train_cfg.dataset}_{train_cfg.width}_{train_cfg.depth}_{step}.pth')