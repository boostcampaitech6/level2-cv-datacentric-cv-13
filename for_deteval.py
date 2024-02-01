import os
import os.path as osp
import numpy as np
import json
import shutil
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from detect import detect
import cv2


def split_train_val(seed, data_dir, test_size):

    data_path = osp.join(data_dir, 'img/train')
    imgs = os.listdir(data_path)

    des_path = 'data/train_val_split_'+ str(seed)
    des_train_path = osp.join(des_path,'img/train')
    des_val_path = osp.join(des_path,'img/valid')
    des_anno_path = osp.join(des_path, 'ufo')

    train_fnames, val_fnames = train_test_split(imgs, test_size=test_size, random_state=seed)


    if not os.path.exists(des_path):
        os.makedirs(des_train_path)
        os.makedirs(des_val_path)
        os.makedirs(des_anno_path)

        shutil.copyfile(osp.join(data_dir, 'ufo/train.json'), osp.join(des_anno_path,'train.json'))
        for file in train_fnames:
            shutil.copyfile(osp.join(data_path, file), osp.join(des_train_path, file))
        for file in val_fnames:
            shutil.copyfile(osp.join(data_path, file), osp.join(des_val_path, file))
    return des_path, train_fnames, val_fnames
    
'''
{
    "images": {
        "drp.en_ko.in_house.deepnatural_002411.jpg": {
            "paragraphs": {},
            "words": {
                "0001": {
                    "transcription": "",
                    "points": [
                        [
                            363.23,
                            137.85
                        ],
                        [
                            423.18,
                            139.64
                        ],
                        [
                            422.21,
                            189.72
                        ],
                        [
                            362.26,
                            187.93
                        ]
                    ],
}
'''
def get_gt_bbox(data_dir, val_images):

    anno_file = os.path.join(data_dir, 'ufo/train.json')

    with open(anno_file, 'r') as f:
        file = json.load(f)

    file = file['images']

    results = dict()
    for val_img in val_images:
        results[val_img] = []
        for id in file[val_img]['words'].keys():
            results[val_img].append(file[val_img]['words'][id]['points'])
    return results

def get_pred_bbox(model, data_dir, input_size, batch_size, split='valid'):
    image_fnames, by_sample_bboxes = [], [] 

    images = []
    for image_fpath in tqdm(glob(osp.join(data_dir, 'img/{}/*'.format(split)))):
        image_fnames.append(osp.basename(image_fpath))

        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    results = dict()
    '''
    results = {
            "drp.en_ko.in_house.deepnatural_003650.jpg" : [
                            [좌표, 좌표, 좌표, 좌표],
                            [좌표, 좌표, 좌표, 좌표],
                            [좌표, 좌표, 좌표, 좌표],
                            [좌표, 좌표, 좌표, 좌표],
                            ...
                        ],
            "drp.en_ko.in_house.deepnatural_003502.jpg" : [
                            [좌표, 좌표, 좌표, 좌표],
                            [좌표, 좌표, 좌표, 좌표],
                            [좌표, 좌표, 좌표, 좌표],
                            [좌표, 좌표, 좌표, 좌표],
                            ...
                        ],
            ...
            }
    '''
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        results[image_fname] = bboxes
    return results

