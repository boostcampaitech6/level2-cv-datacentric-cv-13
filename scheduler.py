from torch.optim import lr_scheduler

class CustomScheduler:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        
    def __call__(self, optimizer, max_epochs, method: str):
        _cfg = self.cfg[method]
        if method == "cyclic":
            return lr_scheduler.CyclicLR(
                optimizer, 
                base_lr=_cfg['base_lr'],
                max_lr=_cfg['max_lr'],
                mode=_cfg['mode'],
                gamma=_cfg['gamma'],
            )
        elif method == "cosine":
            return lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=max_epochs,
                eta_min=_cfg['eta_min'], 
            )
        elif method == "step":
            return lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=[max_epochs // 2], 
                gamma=_cfg['gamma'],
            )
