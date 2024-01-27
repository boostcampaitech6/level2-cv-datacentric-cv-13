import subprocess
import os
import numpy as np
import torch
from torchvision.ops import nms

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# def merge_quadrangle_n9(polys, thres=0.3, precision=10000):
#     from .adaptor import merge_quadrangle_n9 as nms_impl
#     if len(polys) == 0:
#         return np.array([], dtype='float32')
#     p = polys.copy()
#     p[:,:8] *= precision
#     ret = np.array(nms_impl(p, thres), dtype='float32')
#     ret[:,:8] /= precision
#     return ret

#     ## shape of polys: (n, 9) x1, ..., scoremap
    

def merge_quadrangle_n9(polys, thres=0.3, precision=10000):
    # from .adaptor import merge_quadrangle_n9 as nms_impl
    if len(polys) == 0:
        return np.array([], dtype='float32')
    p = polys.copy()
    p[:, :8] *= precision
    _boxes = torch.from_numpy(p[:, [0, 1, 4, 5]]).cuda()
    _scores = torch.from_numpy(p[:, 8]).cuda()
    keep = nms(_boxes, _scores, thres)
    ret = p[keep.cpu().numpy(), :]
    ret[:, :8] /= precision
    
    return ret

    # box -> (x1, y1, x2, y2), score map, thre
    # return => keep 할 box의 idx
    
    