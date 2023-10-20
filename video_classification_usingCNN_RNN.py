import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio
import cv2
import os
from tqdm import tqdm
import shutil
from tabulate import tabulate
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

## make CVS file # #

# Open the .txt file which have names of training videos
# f = open("ucfTrainTestlist/trainlist01.txt", "r")
# temp = f.read()
# videos = temp.split('\n')
#
# # Create a dataframe having video names
# train = pd.DataFrame()
# train['video_name'] = videos
# train = train[:-1]
# train.head()
# # Open the .txt file which have names of test videos
# with open("ucfTrainTestlist/testlist01.txt", "r") as f:
#     temp = f.read()
# videos = temp.split("\n")
#
# # Create a dataframe having video names
# test = pd.DataFrame()
# test["video_name"] = videos
# test = test[:-1]
# test.head()
#
# def extract_tag(video_path):
#     return video_path.split("/")[0]
#
# def separate_video_name(video_name):
#     return video_name.split("/")[1]
#
# def rectify_video_name(video_name):
#     return video_name.split(" ")[0]
#
# def move_videos(df, input_dir,output_dir):
#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)
#     for i in tqdm(range(df.shape[0])):
#         videoFile = df['video_name'][i].split("/")[-1]
#         videoPath = os.path.join(input_dir, df['tag'][i], videoFile)
#         shutil.copy2(videoPath, output_dir)
#     print()
#     print(f"Total videos: {len(os.listdir(output_dir))}")
#
# train["tag"] = train["video_name"].apply(extract_tag)
# train["video_name"] = train["video_name"].apply(separate_video_name)
# train.head()
# train["video_name"] = train["video_name"].apply(rectify_video_name)
# train.head()
# test["tag"] = test["video_name"].apply(extract_tag)
# test["video_name"] = test["video_name"].apply(separate_video_name)
# test.head()
# ## Top n action
# n = 10
# topNActs = train["tag"].value_counts().nlargest(n).reset_index()["index"].tolist()
# train_new = train[train["tag"].isin(topNActs)]
# test_new = test[test["tag"].isin(topNActs)]
# train_new.shape, test_new.shape
# ##
# train_new = train_new.reset_index(drop=True)
# test_new = test_new.reset_index(drop=True)
#
# input_dir = r'E:\UCF-101';
# #move_videos(train_new, input_dir, "train")
# #move_videos(test_new, input_dir, "test")
# train_new.to_csv("train.csv", index=False);
# test_new.to_csv("test.csv", index=False);
##data preparation end

train_df = pd.read_csv("train.csv");
test_df = pd.read_csv("test.csv");

print(f"Total videos for training: {len(train_df)}");
print(f"Total videos for testing: {len(test_df)}");

train_df.sample(10);


## capture the frames from video

def crop_center_square(frame)