import argparse
import os
import torch
from pathlib import Path
import sys

import Dataset
from conv_utils import *
from config import update_config
from model.conv3D import conv3D
from model.layer import aothead, mihead, mbphead, mdhead
from loss import *
from yolov3.detect_loc import run


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to train.')
parser.add_argument('--trained_model', default=None, type=str, help='The pre-trained model to evaluate.')


def inference(cfg):
    shared_conv = conv3D(cfg.width).cuda().eval()
    aot_head = aothead(cfg.width).cuda().eval()
    mi_head = mihead(cfg.width).cuda().eval()
    mbp_head = mbphead(cfg.width).cuda().eval()
    md_head = mdhead(cfg.width).cuda().eval()

    shared_conv.load_state_dict(torch.load('weights/' + cfg.trained_model)['shared'])
    aot_head.load_state_dict(torch.load('weights/' + cfg.trained_model)['aot'])
    mi_head.load_state_dict(torch.load('weights/' + cfg.trained_model)['mi'])
    mbp_head.load_state_dict(torch.load('weights/' + cfg.trained_model)['mbp'])
    md_head.load_state_dict(torch.load('weights/' + cfg.trained_model)['md'])

    mbp_loss = mbploss().cuda()
    val_mdloss = valmdloss().cuda()

    video_folders = os.listdir(cfg.test_data)
    video_folders.sort()
    video_folders = [os.path.join(cfg.test_data, aa) for aa in video_folders]

    anomaly_score = []

    with torch.no_grad():
        for i, folder in enumerate(video_folders):
            dataset = Dataset.val_dataset(folder)
        
            score = []

            for j, (clips, i_path) in enumerate(dataset):

                pred, val_yolo_cls_prob = run(weights=ROOT / 'yolov3/yolov3.pt', source=i_path, imgsz=clips.shape[2:4], conf_thres=conf_thres)
                if pred == -1:
                    continue

                val_input_crop = img_crop(clips, pred[0])
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
                val_mbp_target_shape = val_mbp_target.shape
                val_mbp_target = val_mbp_target.reshape(val_mbp_target_shape[0], -1, val_mbp_target_shape[1], val_mbp_target_shape[2]).cuda()

                val_md_input = val_input_crop[:, 3, :].clone().detach().unsqueeze(dim=1)
                val_md_shape = val_md_input.shape
                val_md_input = val_md_input.reshape(val_md_shape[0], -1, val_md_shape[1], val_md_shape[2], val_md_shape[3]).cuda()

                val_aot_shared = shared_conv(val_aot_input, cfg.depth)
                val_mi_shared = shared_conv(val_mi_input, cfg.depth)
                val_mbp_shared = shared_conv(val_mbp_input, cfg.depth)
                val_md_shared = shared_conv(val_md_input, cfg.depth)

                val_aot_shared = val_aot_shared.squeeze(dim=2)
                val_mi_shared = val_mi_shared.squeeze(dim=2)
                val_mbp_shared = val_mbp_shared.squeeze(dim=2)
                val_md_shared = val_md_shared.squeeze(dim=2)

                val_aot_output = aot_head(val_aot_shared, cfg.depth)
                val_mi_output = mi_head(val_mi_shared, cfg.depth)
                val_mbp_output = mbp_head(val_mbp_shared, cfg.depth)
                val_md_output_res, val_md_output_yolo = md_head(val_md_shared, cfg.depth)

                aot_score = torch.sum(val_aot_output[:, 1])
                mi_score = torch.sum(val_mi_output[:, 1])
                mbp_score = mbp_loss(val_mbp_output, val_mbp_target)
                md_score = val_mdloss(val_md_output_yolo, val_yolo_cls_prob)

                total_score = ((aot_score + mi_score + mbp_score + md_score)/4).cpu().detach().numpy()

                score.append(float(total_score))
            
            anomaly_score.append(np.array(score))
    
    print('\nAll frames were detected, begin to compute AUC.')



if __name__ == '__main__':
    args = parser.parse_args()
    test_cfg = update_config(args, mode='test')
    test_cfg.print_cfg()
    inference(test_cfg)