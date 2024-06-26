from pathlib import Path

import torch


# Used only for training purposes.
def load_params_from_model_path(model_path, model, optimizer=None):
    model_name = model.__class__.__name__
    model_file = Path(model_path)
    if not (model_file.exists() and model_file.is_file()):
        print("model file in general_config is not a file or does not exist")
        return
    print(f'Loading model from {model_file}')
    ckpt = torch.load(model_file)
    start_epoch = None
    lr_now = None
    err_best = {}
    if model_name == 'CISTGCN' or model_name == 'PGBIG' or model_name == 'HRI' or model_name == 'MMA':
        start_epoch = ckpt['epoch']
        err_best = ckpt['err'] if model.__class__.__name__ == 'CISTGCN' else {"mpjpe": ckpt['err']}
        lr_now = ckpt['lr']
        if optimizer is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            lr_now = None
        print(f'model loaded at epoch {start_epoch} with test error of {err_best["mpjpe"]}')
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
        err_best["mpjpe"] = None

    return {'epoch': start_epoch,
            'lr': lr_now,
            'err': err_best,
            'model': model,
            'optimizer': optimizer}
