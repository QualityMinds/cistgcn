import re

from .. import losses
import torch
from tqdm.auto import tqdm

torch.manual_seed(0)


def gradient_control(model, max_norm):
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_norm))
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=float(max_norm))


def get_loss(loader, kwargs):
    data = iter(loader)._next_data()["target"]
    if kwargs.loss.type == "weighted_mpjpe" or kwargs.loss.type == "w_mpjpe":
        loss_func = losses.weighted_mpjpe
    elif kwargs.loss.type == "mpjpe":
        loss_func = losses.mpjpe
    elif kwargs.loss.type == "rmpjpe":
        loss_func = losses.rmpjpe
    elif kwargs.loss.type == "mpjpe_soft":
        loss_func = losses.mpjpe_soft
    elif kwargs.loss.type == "weighted_mpjpe_soft" or kwargs.loss.type == "w_mpjpe_soft":
        loss_func = losses.weighted_mpjpe_soft
    else:
        NotImplemented
    weights = torch.arange(1, data.shape[1] + 1)
    w_vals = kwargs.loss.weights
    if "linear" in w_vals:
        weights = weights
    if "sqrt" in w_vals:
        weights = torch.sqrt(weights)
    elif "exp" in w_vals:
        weights = torch.exp(weights / (weights.max() / 5))
    elif "square" in w_vals:
        weights = torch.pow(weights / (weights.max() / 5), 2)
    else:
        NotImplemented
    # weights = weights / weights.max()
    weights = weights.unsqueeze(0).unsqueeze(2).tile(1, 1, data.shape[2]).cuda()
    return loss_func, weights


def train(loader, model, optimizer, scheduler, writer=None, loss_names=["pose", "vel", "norm_vel"], **kwargs):
    full_loss_list = losses.LossOperator()

    loss_func, weights = get_loss(loader, kwargs.get("learning_config"))
    if "speed" in kwargs.get("learning_config").loss.weights:
        arg_speed = kwargs.get("learning_config").loss.weights
        elems = re.findall(r'\d+', arg_speed)
        factor = float(elems[0] if len(elems) > 0 else 1)
    model.train()
    # tdqm: https://stackoverflow.com/questions/41707229/tqdm-printing-to-newline
    for i, data in tqdm(enumerate(loader), total=loader.__len__()):
        inputs = data["sample"].cuda()
        targets = [data["target"].cuda(), data["target_vel"].cuda(), data["target_gvel"].cuda()]
        outputs = model(inputs)

        # Losses computation.
        w = weights.tile(targets[0].shape[0], 1, 1).float()
        if "speed" in kwargs.get("learning_config").loss.weights:
            speeds = targets[2][:, :, :, 0]
            # speeds = targets[2].mean(1, keepdims=True)[:, :, :, 0]
            speeds /= (speeds.max(2, keepdims=True)[0] + 1e-6)
            if arg_speed == "speed":
                w = speeds * factor
            else:
                w += (speeds * factor)
            # w /= w.max(0)[0]

        all_losses = {}
        avg_loss = 0
        for tar, out, name, wloss in zip(targets, outputs, loss_names, [1, 1, 0.5]):
            all_losses[name] = wloss * loss_func(tar, out, w=w)
            avg_loss += all_losses[name]

        # calculate loss and backward
        optimizer.zero_grad()
        avg_loss.backward()

        # Training metrics.
        global_step = kwargs.get("epoch") * loader.__len__() + i
        for loss in all_losses:
            writer.add_scalar(f'losses/loss_{loss}', all_losses[loss].item(), global_step)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
        if isinstance(kwargs.get("save_grads"), (float, int)) and not isinstance(kwargs.get("save_grads"), bool):
            if global_step % int(kwargs.get("save_grads")) == 0:
                # Record all weights and gradients per epoch in histograms.
                for name, weight in model.named_parameters():
                    writer.add_histogram(name, weight, global_step)
                    writer.add_histogram(f'{name}.grad', weight.grad, global_step)
                    writer.add_scalar(f'grads/{name}.grad', weight.grad.norm().item(), global_step)
                    writer.add_scalar(f'values/{name}', weight.norm().item(), global_step)

        if hasattr(kwargs.get("learning_config"), "max_norm"):
            gradient_control(model, kwargs.get("learning_config").max_norm)

        if isinstance(kwargs.get("save_grads"), (float, int)) and not isinstance(kwargs.get("save_grads"), bool):
            if global_step % int(kwargs.get("save_grads")) == 0:
                # Record all weights and gradients per epoch in histograms.
                for name, weight in model.named_parameters():
                    writer.add_scalar(f'clip_grads/{name}.grad', weight.grad.norm().item(), global_step)

        optimizer.step()
        scheduler.step()

        loss_list = []
        for loss in all_losses:
            loss_list.append(all_losses[loss].item())
        full_loss_list.append(loss_list)
        # if i == 1: break;  # Remove when debug finishes

    torch.cuda.empty_cache()

    return {"loss": full_loss_list.mean(0),
            "loss_names": loss_names,
            "pred": outputs[0].detach().cpu().numpy(),  # n, seq_len=20?, 25, 3
            "target": targets[0].detach().cpu().numpy(), }
