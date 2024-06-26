import numpy as np
import torch
from torch.optim import lr_scheduler


class LearningRateWarmUP(object):
    def __init__(self, optimizer, warmup_iteration, target_lr, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_iteration = warmup_iteration
        self.target_lr = np.array(target_lr)
        self.after_scheduler = after_scheduler
        self.iterator = 0
        self.step()

    def warmup_learning_rate(self):
        warmup_lr = self.target_lr * float(self.iterator) / float(self.warmup_iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def step(self):
        if self.iterator <= self.warmup_iteration:
            self.warmup_learning_rate()  # self.iterator
        else:
            self.after_scheduler.step()  # self.iterator - self.warmup_iteration not necessary
        self.iterator = self.iterator + 1

    def load_state_dict(self, state_dict):
        self.after_scheduler.load_state_dict(state_dict)


def scheduler(optimizer, opt, dataset_iterations=None, epochs=None):
    if opt.type == "StepLR":
        operator = lr_scheduler.StepLR(optimizer, **opt.params.__dict__)
    elif opt.type == "MultiStepLR":
        operator = lr_scheduler.MultiStepLR(optimizer, **opt.params.__dict__)
    elif opt.type == "CosineAnnealingLR":
        if opt.params.T_max == "end":
            opt.params.T_max = dataset_iterations * epochs
        operator = lr_scheduler.CosineAnnealingLR(optimizer, **opt.params.__dict__)
    else:
        operator = None
        NotImplementedError
    return operator


def lr_decay(optimizer, lr_now, gamma):
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def set_optimizer(model, opt):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(opt.learning_config.lr),
                                 weight_decay=float(opt.learning_config.weight_decay))
    return optimizer


def save_ckpt(state, is_best=True, save_all=False, file_name='ckpt.pth.tar'):
    torch.save(state, str(file_name).replace(".pth.", "_last.pth."))
    if is_best:
        print("Saving a new BEST model")
        torch.save(state, str(file_name).replace(".pth.", "_best.pth."))
    if save_all:
        torch.save(state, str(file_name).replace(".pth.", f'_epoch_{state["epoch"]:05}.pth.'))


def load_ckpt(model, file_name):
    print("Loading model")
    arch = torch.load(file_name)
    model.load_state_dict(arch['state_dict'])
    return model


if __name__ == '__main__':
    lr = 1e-2
    batches_per_epoch = 704
    epochs = 50
    model = torch.nn.Sequential(torch.nn.Linear(1, 1))
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=0.)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.85)
    print("Testing the LR schedule")
    for epoch in range(epochs):
        for _ in range(batches_per_epoch):
            optimizer.step()
            scheduler.step()
        print(f'epoch:{epoch}/{epochs}, lr:{optimizer.param_groups[0]["lr"]:.4E}')
