import random
import torch
import numpy as np
import cv2
import glob
import os
import scipy.io as scio
from torch.utils.data import Dataset
from multiprocessing import Process, freeze_support


# frame 불러오기
def np_load_frame(filename):
    img = cv2.imread(filename)
    return img

class train_dataset(Dataset):
    
    def __init__(self, cfg):
        # list마다 각 folder의 이미지들이 넣어져있음
        self.training_videos = []
        self.all_frames_training = []

        # from cfg.train_data : 'data_root/' + dataset name + '/training/' + all folder
        all_folder = sorted(glob.glob(f'{cfg.train_data}/*'))
        all_folder_len = len(all_folder)
        for folder in all_folder[:int(all_folder_len*0.85 + 1)]:
            # root 속에 있는 모든 jpg 파일들 선택
            all_imgs = glob.glob(f'{folder}/*.jpg')
            all_imgs.sort()
            self.training_videos.append(all_imgs)

            frames = list(range(3, len(all_imgs) - 3))
            random.shuffle(frames)
            self.all_frames_training.append(frames)
        
    def __len__(self):  
        return len(self.training_videos)

    def __getitem__(self, idx):
        folder = self.training_videos[idx]
        start = self.all_frames_training[idx][-1]
        i_path = folder[start]

        video_clip = []
        for i in range(start-3, start + 4):
            video_clip.append(np_load_frame(folder[i]))

        random_clip = [np_load_frame(folder[start])]
        
        temp = start
        for i in range(3):
            f = random.randrange(1, 5)
            if temp - (2 - i) - f >= 0:
                random_clip.append(np_load_frame(folder[temp - f]))
                temp -= f
            else:
                random_clip.append(np_load_frame(folder[temp - 1]))
                temp -= 1
    
        random_clip.reverse()

        temp = start
        for i in range(3):
            f = random.randrange(1, 5)
            if temp + (2 - i) + f <= len(folder) - 1:
                random_clip.append(np_load_frame(folder[temp + f]))
                temp += f
            else:
                random_clip.append(np_load_frame(folder[temp + 1]))
                temp += 1

        video_clip = np.array(video_clip)
        random_clip = np.array(random_clip)
        
        return idx, video_clip, random_clip, i_path



class val_dataset(Dataset):
    
    def __init__(self, video_folder):
        self.imgs = glob.glob(video_folder + '/*.jpg')
        self.imgs.sort()
        self.img_idx = range(3, len(self.imgs)-3)
        
    def __len__(self):  
        return len(self.imgs) - 6 

    def __getitem__(self, idx):
        video_clips = []
        start = self.img_idx[idx] - 3
        for i in range(7):
            video_clips.append(np_load_frame(self.imgs[start + i]))
        
        i_path = self.imgs[self.img_idx[idx]]
        
        video_clips = torch.from_numpy(np.array(video_clips))
        return video_clips, i_path


class Label_loader:
    def __init__(self, cfg, video_folders):
        assert cfg.dataset in ('ped2', 'avenue', 'shanghaitech'), f'Did not find the related gt for \'{cfg.dataset}\'.'
        self.cfg = cfg
        self.name = cfg.dataset
        self.frame_path = cfg.test_data
        self.mat_path = f'{cfg.data_root + self.name}/{self.name}.mat'
        self.video_folders = video_folders

    def __call__(self):
        if self.name == 'shanghaitech':
            gt = self.load_shanghaitech()
        else:
            gt = self.load_ucsd_avenue()
        return gt

    def load_ucsd_avenue(self):
        abnormal_events = scio.loadmat(self.mat_path, squeeze_me=True)['gt']

        all_gt = []
        for i in range(abnormal_events.shape[0]):
            length = len(os.listdir(self.video_folders[i]))
            sub_video_gt = np.zeros((length,), dtype=np.int8)

            one_abnormal = abnormal_events[i]
            if one_abnormal.ndim == 1:
                one_abnormal = one_abnormal.reshape((one_abnormal.shape[0], -1))

            for j in range(one_abnormal.shape[1]):
                start = one_abnormal[0, j] - 1
                end = one_abnormal[1, j]

                sub_video_gt[start: end] = 1

            all_gt.append(sub_video_gt)

        return all_gt

    def load_shanghaitech(self):
        np_list = glob.glob(f'{self.cfg.data_root + self.name}/frame_masks/')
        np_list.sort()

        gt = []
        for npy in np_list:
            gt.append(np.load(npy))

        return gt