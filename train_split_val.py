import os
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

import yaml
import wandb

from east_dataset import EASTDataset
from dataset import SceneTextDataset, CustomSceneTextDataset
from model import EAST
from argparser import Parser
from augmentation import BaseTransform
from importlib import import_module
from logger import WeightAndBiasLogger
from optimizer import CustomOptimizer
from scheduler import CustomScheduler
from for_deteval import *
from deteval import calc_deteval_metrics

import random
import numpy as np


def do_training(
    args, config, data_dir, model_dir, device, image_size, input_size, 
    num_workers, batch_size, learning_rate, max_epoch, save_interval, 
    ignore_tags, output_dir, transform, exp_name, seed, optimizer, optim_hparams, 
    scheduler, sched_hparams,
):
    wb_logger = WeightAndBiasLogger(args, exp_name)
    transform = getattr(import_module("augmentation"), transform)

    new_data_dir, train_fnames, val_fnames = split_train_val(seed=seed, data_dir=data_dir, test_size=0.2)

    dataset = CustomSceneTextDataset(
        new_data_dir,
        train_fnames,
        split='train',
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags,
        transform = transform
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset,
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
    best_f1_score = 0
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir, exp_name):
                os.makedirs(model_dir, exp_name)

            ckpt_fpath = osp.join(model_dir, exp_name, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)
        
        wb_logger.log(
                {
                    "Mean Loss": epoch_loss / num_batches,
                }
            )
        
        # validation

        with torch.no_grad():
            print('Calc_deteval_metrics...')
                
            pred_bbox = get_pred_bbox(model, new_data_dir, input_size, batch_size, split='valid')
            gt_bbox = get_gt_bbox(new_data_dir, val_fnames)

            result = calc_deteval_metrics(pred_bbox, gt_bbox)
            precision = result['total']['precision']
            recall = result['total']['recall']
            f1_score = result['total']['hmean']

            print('F1_score : {:.4f} | Epoch : {}'.format(f1_score, epoch))

            wb_logger.log(
                {
                    "Precision" : precision,
                    "Recall" : recall,
                    "f1_score" : f1_score
                }
            )
            
            if best_f1_score < f1_score:
                if not osp.exists(osp.join(model_dir, exp_name)):
                    os.makedirs(osp.join(model_dir, exp_name))

                best_ckpt_fpath = osp.join(model_dir, exp_name,'best.pth')
                torch.save(model.state_dict(), best_ckpt_fpath)

        
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
