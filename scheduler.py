from torch.optim import lr_scheduler

class CustomScheduler:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        
    def __call__(self, optimizer, max_epochs, method: str):
        _cfg = self.cfg[method]
        if method == "plateau":
            return lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode=_cfg['mode'],
                factor=_cfg['factor'],
                patience=_cfg['patience'],
                min_lr=_cfg['min_lr']
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
