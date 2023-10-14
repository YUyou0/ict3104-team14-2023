import os
import random
from abc import abstractmethod
import math
import pandas as pd
import av
import cv2
import decord
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from decord import VideoReader, cpu
import torchvision.transforms._transforms_video as transforms_video
from torchvision.transforms.functional import to_tensor
from collections import OrderedDict
import time
import csv

# New imports
import json


class CharadesDataset(Dataset):

    def __init__(self, 
                 video_path, 
                 width=512, 
                 height=512, 
                 n_sample_frames=8, 
                 dataset_set='train', 
                 prompt=None, 
                 sample_frame_rate=2, 
                 sample_start_idx=0, 
                 accelerator=None):
        
        try:
            host_gpu_num = accelerator.num_processes
            host_num = 1
            all_rank = host_gpu_num * host_num
            global_rank = accelerator.local_process_index
        except:
            pass

        print('dataset rank:', global_rank, ' / ',all_rank, ' ')


        # self.data_dir = 'Your dataset path'
        # if dataset_set=='train':
        #     self.text_name = 'caption_rm2048_train.csv'
        # else:
        #     self.text_name = 'caption_2048_val_new.csv'
        # self.meta_path = os.path.join(self.data_dir, self.text_name)
        

        #--------New code -------------------------------------#
    
        # Set metadata path
        self.meta_path = 'training-data/training-data.json'
        # Set video directory
        self.data_dir = video_path

        #-----------------------------------------------------#

        spatial_transform = 'resize_center_crop'
        resolution=width
        load_raw_resolution=True

        video_length= n_sample_frames
        fps_max=None
        load_resize_keep_ratio=False

        self.global_rank = global_rank
        self.all_rank = all_rank
        # self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.frame_stride = sample_frame_rate
        self.load_raw_resolution = load_raw_resolution
        self.fps_max = fps_max
        self.load_resize_keep_ratio = load_resize_keep_ratio
        print('start load meta data')
        self._load_metadata()
        print('load meta data done!!!')
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms_video.RandomCropVideo(crop_resolution)
            elif spatial_transform == "resize_center_crop":
                assert(self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(resolution),
                    transforms_video.CenterCropVideo(resolution),
                    ])
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms_video.CenterCropVideo(resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

        # def _load_metadata(self):
        #     # clip_id frame_id caption 
        #     last_clip_id = ''
        #     self.metadata = []
        #     start_time = time.time()
        #     caption_path = self.meta_path
        #     count=-1
        #     total_count = 8854264 #8856312 - 2048

        #     with open(caption_path, 'r',encoding="utf-8") as csvfile: #41s
        #         reader = csv.DictReader(csvfile)
        #         for row in reader:
        #             if row['clip_id'] != last_clip_id:
        #                 count+=1
        #                 if count >= (total_count // self.all_rank)*self.all_rank: # drop last
        #                     break
        #                 last_clip_id = row['clip_id']
        #                 if count % self.all_rank == self.global_rank:
        #                     self.metadata.append([('%02d'%int(row['part_id']))+row['clip_id']]) 
        #                     self.metadata[-1].append([row['caption']])
        #             else:
        #                 if count % self.all_rank == self.global_rank:
        #                     self.metadata[-1][-1].append(row['caption'])
        #     end_time = time.time()
        #     print('load %d - %d items use time: %.1f;' % (len(self.metadata), count, end_time-start_time))

    #--------New code ----------------------------------------------------------------------------#

    # Change the _load_metadata function to load the metadata from the JSON file
    def _load_metadata(self):
        # Initialize an empty list to store metadata
        self.metadata = []
        # Define the path to the meta data JSON file
        metadata_path = self.meta_path
        # Time
        start_time = time.time()
        
        try:
            # Open the JSON file and load its contents
            with open(metadata_path, 'r', encoding='utf-8') as jsonfile:
                metadata_list = json.load(jsonfile)

                # Iterate through each entry in the JSON file
                for entry in metadata_list:
                    video_id = entry.get("id", "")
                    caption = entry.get("script", "")
                    print("Loading video with ID:", video_id)

                    # Check if the video_id and caption are valid 
                    if video_id and caption:
                        self.metadata.append([video_id, caption])

        except Exception as e:
            print(f"Error loading metadata from JSON: {str(e)}")

        end_time = time.time()
        print(f"Loaded {len(self.metadata)} items in {end_time - start_time} seconds")

    #--------------------------------------------------------------------------------------#

    # def _get_video_path(self, sample):
    #     part_id = int(sample[0][:2])
    #     clip_id = sample[0][2:]
    #     video_path = os.path.join(self.data_dir,'part_%d' % part_id, 'video_clips', clip_id)
    #     return video_path
    
    #--------New code -----------------------------------------------------------------------#

    # Modify the _get_video_path function (Remove the part_id and clip_id ---> video_id)
    def _get_video_path(self, sample):
        video_id = sample[0]
        video_path = os.path.join('input_videos', 'charades', video_id + '.mp4')




        return video_path
    
    #--------------------------------------------------------------------------------------#

    # def __getitem__(self, index):
    #     while True:
            
    #         index = index % len(self.metadata)
    #         sample = self.metadata[index]
    #         video_path = self._get_video_path(sample)
             
    #         try:
    #             if self.load_raw_resolution:
    #                 video_reader = VideoReader(video_path, ctx=cpu(0))
    #             elif self.load_resize_keep_ratio:
    #                 # resize scale is according to the short side
    #                 h, w, c = VideoReader(video_path, ctx=cpu(0))[0].shape
    #                 if h < w:
    #                     scale = h / self.resolution[0]
    #                 else:
    #                     scale = w / self.resolution[1]

    #                 h = math.ceil(h / scale)
    #                 w = math.ceil(w / scale)
    #                 video_reader = VideoReader(video_path, ctx=cpu(0), width=w, height=h)
    #             else:
    #                 video_reader = VideoReader(video_path, ctx=cpu(0), width=self.resolution[1], height=self.resolution[0])
    #             if len(video_reader) < self.video_length:
    #                 print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
    #                 index += 1
    #                 continue
    #             else:
    #                 pass
    #         except:
    #             index += 1
    #             print(f"Load video failed! path = {video_path}")
    #             continue
            
    #         fps_ori = video_reader.get_avg_fps()

    #         fs = self.frame_stride
    #         allf = len(video_reader)
    #         if self.frame_stride != 1:
    #             all_frames = list(range(0, len(video_reader), self.frame_stride))
    #             if len(all_frames) < self.video_length:
    #                 fs = len(video_reader) // self.video_length
    #                 assert(fs != 0)
    #                 all_frames = list(range(0, len(video_reader), fs))
    #         else:
    #             all_frames = list(range(len(video_reader)))
            
    #         # select a random clip
    #         rand_idx = random.randint(0, len(all_frames) - self.video_length)
    #         frame_indices = all_frames[rand_idx:rand_idx+self.video_length]
    #         try:
    #             frames = video_reader.get_batch(frame_indices)
    #             break
    #         except:
    #             print(f"Get frames failed! path = {video_path}")
    #             index += 1
    #             continue

    #     assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
    #     frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
       
    #     if self.spatial_transform is not None:
    #         frames = self.spatial_transform(frames)
    #     assert(frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
    #     frames = frames.byte()
    #     # fps
    #     fps_clip = fps_ori // self.frame_stride
    #     if self.fps_max is not None and fps_clip > self.fps_max:
    #         fps_clip = self.fps_max
        
    #     # caption index
    #     middle_idx = (rand_idx + self.video_length /2 )*fs
    #     big_cap_idx = (middle_idx // 64 +1) *64
    #     small_cap_idx = (middle_idx // 64) *64
    #     if big_cap_idx >= allf or ((big_cap_idx-middle_idx) >= (small_cap_idx-middle_idx)):
    #         cap_idx = small_cap_idx
    #     else:
    #         cap_idx = big_cap_idx
    #     # print(middle_idx, small_cap_idx, big_cap_idx,cap_idx)
    #     caption = sample[1][int(cap_idx//64)]

    #     frames = frames.permute(1,0,2,3)
    #     skeleton_final = torch.zeros_like(frames).byte()
    #     frames = (frames / 127.5 - 1.0)
    #     skeleton_final = (skeleton_final / 127.5 - 1.0)
    #     example = {'pixel_values': frames, 'sentence': caption, 'pose': skeleton_final}

    #     return example

    #--------New code -----------------------------------------------------------------------#

    # Modify the __getitem__ function (Remove the while loop as video only has one frame)
    def __getitem__(self, index):
        index = index % len(self.metadata)
        video = self.metadata[index]
        video_path = self._get_video_path(video)
        
        try:
            if self.load_raw_resolution:
                video_reader = VideoReader(video_path, ctx=cpu(0))
            elif self.load_resize_keep_ratio:
                # resize scale is according to the short side
                h, w, c = VideoReader(video_path, ctx=cpu(0))[0].shape
                if h < w:
                    scale = h / self.resolution[0]
                else:
                    scale = w / self.resolution[1]

                h = math.ceil(h / scale)
                w = math.ceil(w / scale)
                video_reader = VideoReader(video_path, ctx=cpu(0), width=w, height=h)
            else:
                video_reader = VideoReader(video_path, ctx=cpu(0), width=self.resolution[1], height=self.resolution[0])
            if len(video_reader) < self.video_length:
                print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
            else:
                fps_ori = video_reader.get_avg_fps()

                fs = self.frame_stride
                allf = len(video_reader)
                if self.frame_stride != 1:
                    all_frames = list(range(0, len(video_reader), self.frame_stride))
                    if len(all_frames) < self.video_length:
                        fs = len(video_reader) // self.video_length
                        assert(fs != 0)
                        all_frames = list(range(0, len(video_reader), fs))
                else:
                    all_frames = list(range(len(video_reader)))

                # select a random clip
                rand_idx = random.randint(0, len(all_frames) - self.video_length)
                frame_indices = all_frames[rand_idx:rand_idx+self.video_length]
                try:
                    frames = video_reader.get_batch(frame_indices)
                except:
                    print(f"Get frames failed! path = {video_path}")
                    return None

                assert(frames.shape[0] == self.video_length), f'{len(frames)}, self.video_length={self.video_length}'
                frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
            
                if self.spatial_transform is not None:
                    frames = self.spatial_transform(frames)
                assert(frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
                frames = frames.byte()

                # fps
                fps_clip = fps_ori // self.frame_stride
                if self.fps_max is not None and fps_clip > self.fps_max:
                    fps_clip = self.fps_max
                
                # caption index
                middle_idx = (rand_idx + self.video_length / 2) * fs
                big_cap_idx = (middle_idx // 64 + 1) * 64
                small_cap_idx = (middle_idx // 64) * 64
                if big_cap_idx >= allf or ((big_cap_idx-middle_idx) >= (small_cap_idx-middle_idx)):
                    cap_idx = small_cap_idx
                else:
                    cap_idx = big_cap_idx
                caption = video[1][int(cap_idx // 64)]

                frames = frames.permute(1, 0, 2, 3)
                skeleton_final = torch.zeros_like(frames).byte()
                frames = (frames / 127.5 - 1.0)
                skeleton_final = (skeleton_final / 127.5 - 1.0)
                example = {'pixel_values': frames, 'sentence': caption, 'pose': skeleton_final}

                return example
        except:
            print(f"Load video failed! path = {video_path}")
            return None 
        
    #--------------------------------------------------------------------------------------#
        
    def __len__(self):
        return len(self.metadata)