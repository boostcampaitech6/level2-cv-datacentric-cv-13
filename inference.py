import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob

import torch
import cv2
from torch import cuda
from model import EAST
from tqdm import tqdm

import yaml

from detect import detect
from argparser import Parser


CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']


def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, split='test'):
    model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
    model.eval()

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

    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    return ufo_result


def inference(args):
    # Initialize model
    # model = EAST(pretrained=False).to(args.device)
    model = EAST(pretrained=False).to('cuda')

    # Get paths to checkpoint files
    ckpt_fpath = osp.join(args.model_dir, 'latest.pth')

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Inference in progress')

    ufo_result = dict(images=dict())
    split_result = do_inference(model, ckpt_fpath, args.data_dir, args.input_size,
                                args.batch_size, split='test')
    ufo_result['images'].update(split_result['images'])

    output_fname = f'output.csv'
    with open(osp.join(args.output_dir, output_fname), 'w') as f:
        json.dump(ufo_result, f, indent=4)


def inference_main(args, epoch):
    # Initialize model
    # model = EAST(pretrained=False).to(args.device)
    model = EAST(pretrained=False).to('cuda')

    # Get paths to checkpoint files
    ckpt_fpath = osp.join(args.model_dir, 'latest.pth')

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Inference in progress')

    ufo_result = dict(images=dict())
    split_result = do_inference(model, ckpt_fpath, args.data_dir, args.input_size,
                                args.batch_size, split='test')
    ufo_result['images'].update(split_result['images'])

    output_fname = f'output_{epoch}.csv'
    with open(osp.join(args.output_dir, output_fname), 'w') as f:
        json.dump(ufo_result, f, indent=4)


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

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    inference(args)
    inference_main(args, 73)
