import random
import torch
import numpy as np
import cv2
import glob
import os
import scipy.io as scio
from torch.utils.data import Dataset
from multiprocessing import Process, freeze_support


class train_dataset(Dataset):
    
    def __init__(self, cfg):
        self.clip_length = 5

        # list마다 각 folder의 이미지들이 넣어져있음
        self.videos = []
        self.all_frames = []
        # from cfg.train_data : 'data_root/' + dataset name + '/training/' + all folder
        for folder in sorted(glob.glob(f'{cfg.train_data}/*')):
            # root 속에 있는 모든 jpg 파일들 선택
            all_imgs = glob.glob(f'{folder}/*.jpg')
            all_imgs.sort()
            self.videos.append(all_imgs)

            frames = list(range(8, len(all_imgs) - 7))
            random.shuffle(frames)
            self.all_frames.append(frames)

    def __len__(self):  
        return len(self.videos)

    def __getitem__(self, indice):
        folder = self.videos[indice]
        start = self.all_frames[indice][-1]

        video_clip = []
        for i in range(start-2, start + 3):
            video_clip.append(cv2.imread(folder[i]))
        
        random_clip = []
        random_seqs = []
        for i in range(4):
            random_seqs.append(random.randrange(1, 5))
        
        print(random_seqs)

        random_clip.append(cv2.imread(folder[start-(random_seqs[0]+random_seqs[1])]))
        random_clip.append(cv2.imread(folder[start-(random_seqs[1])]))
        random_clip.append(cv2.imread(folder[start]))
        random_clip.append(cv2.imread(folder[i+(random_seqs[2])]))
        random_clip.append(cv2.imread(folder[i+(random_seqs[2]+random_seqs[3])]))

        video_clip = torch.from_numpy(np.array(video_clip))
        random_clip = torch.from_numpy(np.array(random_clip))
        
        return indice, video_clip, random_clip