import os
import sys
import json
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
from PIL import Image, ImageDraw

import yaml
from sklearn.model_selection import train_test_split

from east_dataset import EASTDataset
from dataset_unfinished import SceneTextDataset
from model import EAST
from argparser import Parser
from augmentation import BaseTransform
from importlib import import_module
from logger import WeightAndBiasLogger
from optimizer import CustomOptimizer
from scheduler import CustomScheduler
from detect import get_bboxes
from deteval import calc_deteval_metrics
from inference import inference_main

import random
import numpy as np
import wandb


def do_training(
    args, config, data_dir, model_dir, device, image_size, input_size, 
    num_workers, batch_size, learning_rate, max_epoch, save_interval, 
    ignore_tags, output_dir, transform, exp_name, seed, optimizer, optim_hparams, 
    scheduler, sched_hparams,
):
    wb_logger = WeightAndBiasLogger(args, exp_name)
    transform = getattr(import_module("augmentation"), transform)

    try:
        assert osp.isfile("data/medical/ufo/new_train.json") is True
        assert osp.isfile("data/medical/ufo/new_valid.json") is True
    except:
        image_fnames = sorted(os.listdir(os.path.join(data_dir, 'img/train')))
        train_fnames, val_fnames = train_test_split(image_fnames, test_size=0.2, random_state=seed)
        
        with open("data/medical/ufo/train.json", "r") as f:
            train_json = json.load(f)
        
        new_train_json = [x for x in train_json['images'].items() if x[0] in train_fnames]    
        new_valid_json = [x for x in train_json['images'].items() if x[0] in val_fnames]
        new_train_json = {"images": dict(new_train_json)}
        new_valid_json = {"images": dict(new_valid_json)}
        
        with open("data/medical/ufo/new_train.json", "w") as f:
            json.dump(new_train_json, f)
        with open("data/medical/ufo/new_valid.json", "w") as f:
            json.dump(new_valid_json, f)
        
    train_dataset = SceneTextDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags,
        transform = transform
    )
    train_dataset = EASTDataset(train_dataset)
    num_train_batches = math.ceil(len(train_dataset) / batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_dataset = SceneTextDataset(
        data_dir,
        split='new_valid',
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags,
    )
    val_dataset = EASTDataset(val_dataset)
    num_val_batches = math.ceil(len(val_dataset) / batch_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = CustomOptimizer(optim_hparams)(model, optimizer)
    scheduler = CustomScheduler(sched_hparams)(optimizer, max_epoch, scheduler)
    
    model.train()
    for epoch in range(max_epoch):
        train_epoch_loss, valid_epoch_loss= 0, 0
        f1, precision, recall = 0, 0, 0
        
        for _, (img, gt_score_map, gt_geo_map, roi_mask) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
            wb_logger.log(
                {
                    "train_cls_loss": extra_info['cls_loss'],
                    "train_angle_loss": extra_info['angle_loss'],
                    "train_iou_loss": extra_info['iou_loss'],
                }
            )
        
        if not osp.exists(model_dir):
            os.makedirs(model_dir)

        ckpt_fpath = osp.join(model_dir, f'epoch_{epoch}.pth')
        torch.save(model.state_dict(), ckpt_fpath)
        torch.save(model.state_dict(), osp.join(model_dir,"latest.pth"))
        
        for _, (img, gt_score_map, gt_geo_map, roi_mask) in tqdm(enumerate(val_loader)):           
            with torch.no_grad():
                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
            
            valid_epoch_loss += loss.item()            
            pred_bbox_dict, gt_bbox_dict = {}, {}
            
            for i in range(len(gt_score_map)):
                pred_bbox = get_bboxes(
                    extra_info['score_map'][i].detach().cpu().numpy(), 
                    extra_info['geo_map'][i].detach().cpu().numpy()
                )
                gt_bbox = get_bboxes(
                    gt_score_map[i].detach().cpu().numpy(), 
                    gt_geo_map[i].detach().cpu().numpy()
                )
                pred_bbox = pred_bbox[:, :8].reshape(-1, 4, 2)
                gt_bbox = gt_bbox[:, :8].reshape(-1, 4, 2)
                pred_bbox_dict[i] = pred_bbox
                gt_bbox_dict[i] = gt_bbox
            
            metric = calc_deteval_metrics(pred_bbox_dict, gt_bbox_dict)
            f1 += metric['total']['hmean']
            precision += metric['total']['precision']
            recall += metric['total']['recall']
            
            _img = img[0].detach().cpu().numpy()
            _img = np.transpose(_img, (1, 2, 0))
            _img = _img * 255
            _img = _img.astype(np.uint8)
            _gt_pil_img = Image.fromarray(_img)        
            _pred_pil_img = Image.fromarray(_img)
            
            gt_box = metric['per_sample'][0]['gt_bboxes']
            pred_box = metric['per_sample'][0]['det_bboxes']
            
            gt_draw = ImageDraw.Draw(_gt_pil_img)
            pred_draw = ImageDraw.Draw(_pred_pil_img)
            
            for box in gt_box:
                box = np.array(box).astype(np.int32).tolist()
                gt_draw.rectangle(box, outline=(255, 0, 0))
            for box in pred_box:
                box = np.array(box).astype(np.int32).tolist()
                pred_draw.rectangle(box, outline=(0, 255, 0))
                
            _gt_pil_img.save(f'./gt.jpg')
            _pred_pil_img.save(f'./pred.jpg')            
            wb_logger.log(
                {
                    "valid_cls_loss": extra_info['cls_loss'],
                    "valid_angle_loss": extra_info['angle_loss'],
                    "valid_iou_loss": extra_info['iou_loss'],
                    "gt_img": wandb.Image(_gt_pil_img, caption="GT"),
                    "pred_img": wandb.Image(_pred_pil_img, caption="Pred"),
                }
            )

        wb_logger.log(
            {
                "Train Loss": train_epoch_loss / num_train_batches,
                "Val Loss": valid_epoch_loss / num_val_batches,
                "F1_score": f1 / num_val_batches,
                "Precision": precision / num_val_batches,
                "Recall": recall / num_val_batches,
            }
        )
    
        if (epoch+1)%5 == 0:
            inference_main(args, epoch)


def main(args):

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')
    
    # seed 고정
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    do_training(args, **args.__dict__)


if __name__ == '__main__':

    p = Parser()
    p.create_parser()
    
    pargs = p.parser.parse_args()
    try:
        with open(pargs.config, 'r') as fp:
            load_args = yaml.load(fp, Loader=yaml.FullLoader)
        key = vars(pargs).keys()
        for k in load_args.keys():
            if k not in key:
                print("Wrong argument: ", k)
                assert(k in key)
            p.parser.set_defaults(**load_args)
    except FileNotFoundError:
        print("Invalid filename. Check your file path or name.")
        
    args = p.parser.parse_args() 
    p.print_args(args)

    main(args)
