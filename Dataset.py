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
        self.videos = []
        self.all_frames = []
        # from cfg.train_data : 'data_root/' + dataset name + '/training/' + all folder
        for folder in sorted(glob.glob(f'{cfg.train_data}/*')):
            # root 속에 있는 모든 jpg 파일들 선택
            all_imgs = glob.glob(f'{folder}/*.jpg')
            all_imgs.sort()
            self.videos.append(all_imgs)

            frames = list(range(3, len(all_imgs) - 2))
            random.shuffle(frames)
            self.all_frames.append(frames)

    def __len__(self):  
        return len(self.videos)

    def __getitem__(self, idx):
        folder = self.videos[idx]
        start = self.all_frames[idx][-1]
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


