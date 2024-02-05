import torch
from lion_pytorch  import Lion

from model import EAST


class CustomOptimizer:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        
    def __call__(self, model: EAST, method: str="SGD"):
        _cfg = self.cfg[method]
        if method == "SGD":
            return torch.optim.SGD(
                model.parameters(), 
                lr=_cfg['lr'], 
                momentum=_cfg['momentum'], 
                weight_decay=_cfg['weight_decay']
            )
        elif method == "Adam":
            return torch.optim.Adam(
                model.parameters(), 
                lr=_cfg['lr'], 
                betas=(_cfg['beta1'], _cfg['beta2']), 
                eps=_cfg['eps'], 
                weight_decay=_cfg['weight_decay']
            )
        elif method == "AdamW":
            return torch.optim.AdamW(
                model.parameters(), 
                lr=_cfg['lr'], 
                betas=(_cfg['beta1'], _cfg['beta2']), 
                eps=_cfg['eps'], 
                weight_decay=_cfg['weight_decay']
            )
        elif method == "Lion":
            return Lion(
                model.parameters(), 
                lr=_cfg['lr'], 
                betas=(_cfg['beta1'], _cfg['beta2']),
                weight_decay=_cfg['weight_decay']
            )
                