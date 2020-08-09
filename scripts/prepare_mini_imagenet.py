"""
Run this script to prepare the miniImageNet dataset.

This script uses the 100 classes of 600 images each used in the Matching Networks paper. The exact images used are
given in data/mini_imagenet.txt which is downloaded from the link provided in the paper (https://goo.gl/e3orz6).

1. Download files from https://drive.google.com/file/d/0B3Irx3uQNoBMQ1FlNXJsZUdYWEE/view and place in
    data/miniImageNet/images
2. Run the script
"""
from tqdm import tqdm as tqdm
import numpy as np
import shutil
import os
import csv

from config import DATA_PATH
from few_shot.utils import mkdir, rmdir


# Clean up folders
rmdir(DATA_PATH + '/miniImageNet/images_train')
rmdir(DATA_PATH + '/miniImageNet/images_eval')
rmdir(DATA_PATH + '/miniImageNet/images_test')
mkdir(DATA_PATH + '/miniImageNet/images_train')
mkdir(DATA_PATH + '/miniImageNet/images_eval')
mkdir(DATA_PATH + '/miniImageNet/images_test')


# Find class identities
classes = []
for root, _, files in os.walk(DATA_PATH + '/miniImageNet/images/'):
    for f in files:
        if f.endswith('.jpg'):
            classes.append(f[:-12])

classes = list(set(classes))


# Train/val/test split
def class_split(data_type):
    filename = DATA_PATH + '/miniImageNet/csv_files/' + data_type + '.csv'
    images = {}

    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader, None)
        print('Reading IDs....')

        for row in tqdm(csv_reader):
            if row[1] in images.keys():
                images[row[1]].append(row[0])
            else:
                images[row[1]] = [row[0]]
    
    return images

split_lists = ['train', 'eval', 'test']
train_dict, eval_dict, test_dict = list(map(class_split, split_lists))


# Create class folders
for c in train_dict.keys():
    mkdir(DATA_PATH + f'/miniImageNet/images_train/{c}/')

for c in eval_dict.keys():
    mkdir(DATA_PATH + f'/miniImageNet/images_eval/{c}/')

for c in test_dict.keys():
    mkdir(DATA_PATH + f'/miniImageNet/images_test/{c}/')


# Move images to correct location
for root, _, files in os.walk(DATA_PATH + '/miniImageNet/images'):
    for f in tqdm(files, total=600*100):
        if f.endswith('.jpg'):
            class_name = f[:-12]
            image_name = f[-12:]
            # Send to correct folder
            if class_name in train_dict.keys():
                subset_folder = 'images_train'
            elif class_name in eval_dict.keys():
                subset_folder = 'images_eval'
            elif class_name in test_dict.keys():
                subset_folder = 'images_test'
            else:
                raise ValueError('no matching csv & data')
            
            src = f'{root}/{f}'
            dst = DATA_PATH + f'/miniImageNet/{subset_folder}/{class_name}/{image_name}'
            shutil.copy(src, dst)
