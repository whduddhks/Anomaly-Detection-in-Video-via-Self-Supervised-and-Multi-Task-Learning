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
        return len(self.videos)

    def __getitem__(self, idx):
        folder = self.videos[idx]
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
            if temp + (2 - i) + f <= len(folder):
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
            video_clips.append(self.imgs[start + i])
        
        i_path = self.imgs[self.img_idx[idx]]
        
        video_clips = np.array(video_clips)
        return video_clips, i_path