import torch
from torch.optim.lr_scheduler import StepLR
###############################
#
# 以下为冻结代码
#
############################
def freeze_backbone(model):
    # 冻结 diffusion backbone
    for p in model.diff_model.parameters():
        p.requires_grad = False

    # 冻结条件编码
    for p in model.cond_mgr.parameters():
        p.requires_grad = False

    # 冻结 self-conditioning
    for p in model.selfcond_film.parameters():
        p.requires_grad = False

    print('[INFO] Backbone frozen. Train decoders only.')

def build_decoder_optimizer(model, base_lr, wd=1e-4):
    params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("traj_decoder") or name.startswith("tim_decoder"):
            params.append(p)

    optimizer = torch.optim.AdamW(
        params,
        lr=base_lr,
        weight_decay=wd
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.5
    )
    return optimizer, scheduler



##########################################################################################
#
# 全部优化器
#
##########################################################################################
# utils/optimizer_utils.py
def build_full_optimizer(model, base_lr, wd):
    overrides = {
        "diff_model": 0.5,
        "tim_decoder": 1.0,
        "traj_decoder": 1.0,
    }
    param_groups = []
    selected = set()

    for key, mult in overrides.items():
        group = []
        for name, p in model.named_parameters():
            if p.requires_grad and id(p) not in selected and name.startswith(key):
                group.append(p)
                selected.add(id(p))
        if group:
            param_groups.append({
                "params": group,
                "lr": base_lr * mult,
                "weight_decay": wd
            })

    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in selected]
    if rest:
        param_groups.append({
            "params": rest,
            "lr": base_lr,
            "weight_decay": wd
        })

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
    return optimizer, scheduler
