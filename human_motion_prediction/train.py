import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from tensorboardX import SummaryWriter

from . import environment  # train, test, load_params_from_model_path
from . import loaders
from .environment import utils
from .models import choose_net as net
from .utils import yaml_utils, body_utils

np.random.seed(0)


def save_metrics(metrics, name, writer, epoch, num_mesh, action="", db="cmu", dim_used=None):
    if action != "": action = action + "_"
    pred = metrics["pred"]
    target = metrics["target"]
    # pred = (pred - pred.mean(0)) / pred.std(0)
    # target = (target - target.mean(0)) / target.std(0)
    target = body_utils.create_symmetic_3d_edges(target, db=db, dim_used=dim_used)
    pred = body_utils.create_symmetic_3d_edges(pred, db=db, dim_used=dim_used)
    pcl_plot = body_utils.convert_points_to_plot(target, pred, get_color=True)
    for i in range(num_mesh):
        writer.add_mesh(f'{name}/sample{i}', pcl_plot["pcl"][i:i + 1],
                        colors=pcl_plot["colors"][i:i + 1],
                        global_step=epoch)

    # Record numerical results.
    temp_err = None
    if action != "": print(f'{action:21s}', end="")
    print(f'{name}', end=": ")
    for k in metrics:
        if metrics[k] is None or k in ["loss_names", "pred", "target", "inputs"]:
            continue
        if metrics[k].shape and (name == "test" or action != ""):  # If vector
            out_name = "action_" if action != "" else "sequence_"
            if k == 'mpjpe_seq':
                idx = [1, 4, 9, 13, 17, 24] if len(metrics[k]) > 10 else [1, 4, 9]
                temp_err = np.array([f'{40 * (i + 1)}:{v:.2f},' for i, v in enumerate(metrics[k])])[idx]
                temp_err = "mpjpe: " + " ".join(temp_err)
            for i in np.arange(0, len(metrics[k])):
                writer.add_scalar(f'{out_name}metrics/{action}-{k.replace("seq", "")}_{40 * (i + 1)}-',
                                  metrics[k][i], epoch)
        elif metrics[k].shape and name == "train":  # If vector
            for i in np.arange(0, len(metrics[k])):
                print(f'{metrics["loss_names"][i]}: {metrics[k][i]:.6f}', end=" - ")
                writer.add_scalar(f'epoch_losses/{metrics["loss_names"][i]}', metrics[k][i], epoch)
        else:  # If number
            print(f'{k}: {metrics[k]:.2f}', end=" - ")
            out_name = "" if action != "" else "global_"
            writer.add_scalar(f'{out_name}metrics/-{k}-{action}-', metrics[k], epoch)
    if temp_err is not None:
        print("\n" + temp_err, end="")
    print()


def main(yaml_file, opt):
    curr_time = datetime.utcnow().strftime('%Y%m%d_%H%M-id%f')[:-2]
    start_epoch = 0
    err_best = 10000
    is_best = False
    architecture = opt.architecture_config.model
    return_all_joints = opt.environment_config.return_all_joints
    use_actions = True
    if "cmu" in opt.general_config.data_dir:
        db = "cmu"
    elif "h3" in opt.general_config.data_dir:
        db = "h36m"
    elif "3d" in opt.general_config.data_dir:
        db = "3dpw"
        use_actions = False
    elif "amass" in opt.general_config.data_dir:
        db = "amass"
    elif "expi" in opt.general_config.data_dir:
        db = "expi"
    print(f'folder name: {curr_time}')

    ############################################
    # Create Model
    print(">>> creating model")
    print(">>> architecture:", architecture)
    model = net.choose_net(architecture, opt).cuda()
    print(">>> total params: {:.2f}K".format(sum(p.numel() for p in model.parameters()) / 1000.0))
    # import analysis
    # model.eval()
    # flops = analysis.compute_flops_pytorch_model(model,
    #                                              input_sz=(1, opt.architecture_config.model_params.input_n,
    #                                                        opt.architecture_config.model_params.joints, 3))
    # print(f'total flops: {flops["total"] / 1e6:.1f}M')
    # data loadig
    print(">>> loading data")
    train_loader = loaders.get_loader(opt, split="train", model=architecture,
                                      return_all_joints=False, )
    test_loader = loaders.get_loader(opt, split="test", model=architecture,
                                     return_all_joints=return_all_joints, )

    if use_actions:
        action_loader, actions = loaders.get_loader_divided_by_actions(db, "test", opt, shuffle=False, )

    # Learning process
    optimizer = utils.set_optimizer(model, opt)

    if opt.general_config.load_model_path:
        updates = environment.load_params_from_model_path(opt.general_config.load_model_path, model, optimizer)
        start_epoch = updates["epoch"]
        err_best = updates["err"]
        model = updates["model"]
        optimizer = updates["optimizer"]

    scheduler = utils.scheduler(optimizer,
                                opt.learning_config.scheduler,
                                train_loader.__len__(),  # number of batches per epoch (not dataset length)
                                opt.learning_config.epochs)  # epochs
    if hasattr(opt.learning_config, "WarmUp"):
        if opt.learning_config.WarmUp > 0:
            scheduler = utils.LearningRateWarmUP(optimizer=optimizer,
                                                 warmup_iteration=opt.learning_config.WarmUp,
                                                 target_lr=optimizer.param_groups[0]['lr'],
                                                 after_scheduler=scheduler)
    ############################################

    # Set a Tensorboard X for pytorch :)
    # Save configuration and model files.
    ckpt_path = Path(opt.general_config.log_path, opt.general_config.experiment_name, curr_time)
    ckpt_path.mkdir(parents=True, mode=0o770, exist_ok=True)
    writer = SummaryWriter(logdir=str(ckpt_path), comment=opt.meta_config.comment)
    ckpt_path = ckpt_path.joinpath("files")
    ckpt_path.mkdir(parents=True, mode=0o770, exist_ok=True)
    model_file = str(Path(net.__file__).parent.parent.joinpath(*(model.__module__.split(".")[1:]))) + ".py"
    shutil.copyfile(model_file, ckpt_path.joinpath(f'model.py'))
    shutil.copyfile(yaml_file, ckpt_path.joinpath(f'config-{curr_time}.yaml'))

    # Save graph from model.
    writer.add_graph(model, iter(train_loader)._next_data()["sample"].cuda())  # Save graph model, architecture explanation.

    transforms = train_loader.dataset.transform.transforms
    # training process
    for epoch in range(start_epoch, opt.learning_config.epochs):
        print(f'\nepoch:{epoch}/{opt.learning_config.epochs}, lr:{optimizer.param_groups[0]["lr"]:.4E}')

        # per epoch
        train_metrics = environment.train(train_loader, model, optimizer, scheduler, writer, epoch=epoch,
                                          save_grads=opt.environment_config.save_grads,
                                          learning_config=opt.learning_config)
        test_metrics = environment.test(test_loader, model, get_all_samples=opt.environment_config.get_all_samples,
                                        unnormalize=loaders.get_dataset_stats(
                                            db) if opt.learning_config.normalize else None,
                                        db=db,
                                        output_n=opt.architecture_config.model_params.output_n,
                                        joints_n=opt.architecture_config.model_params.joints)
        if use_actions:
            action_metrics = {}
            for a in actions:
                action_metrics[a] = environment.test(action_loader[a], model,
                                                     get_all_samples=opt.environment_config.get_all_samples,
                                                     unnormalize=loaders.get_dataset_stats(
                                                         db) if opt.learning_config.normalize else None,
                                                     db=db,
                                                     output_n=opt.architecture_config.model_params.output_n,
                                                     joints_n=opt.architecture_config.model_params.joints,
                                                     )

        # Save metrics into a log Writer Tensorboard.
        # Plot 3D mesh.
        for metrics, name in [[train_metrics, "train"], [test_metrics, "test"]]:
            save_metrics(metrics, name, writer, epoch, opt.general_config.tensorboard.num_mesh,
                         db=db, dim_used=test_loader.dataset.dim_used)
        if use_actions:
            for a in actions:
                save_metrics(action_metrics[a], "metrics", writer, epoch,
                             opt.general_config.tensorboard.num_mesh, a,
                             db=db, dim_used=test_loader.dataset.dim_used)

        # save ckpt
        # TODO: This is still fixed but must be changed for general estimation
        metric_used_to_save = "mpjpe"
        if test_metrics[metric_used_to_save] <= err_best:
            err_best = test_metrics[metric_used_to_save]
            is_best = True

        if opt.general_config.save_models:
            m_path = ckpt_path.joinpath(f'{opt.general_config.model_name_rel_path}-{curr_time}.pth.tar')
            utils.save_ckpt({'epoch': epoch,
                             'lr': optimizer.param_groups[0]['lr'],
                             'err': test_metrics,
                             'metric_used_to_save': metric_used_to_save,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()},
                            is_best=is_best,
                            save_all=opt.general_config.save_all_intermediate_models,
                            file_name=m_path)
        if np.isnan(np.array(test_metrics["mpjpe"])):
            m_path = ckpt_path.joinpath(f'{opt.general_config.model_name_rel_path}-{curr_time}_nan.pth.tar')
            utils.save_ckpt({'epoch': epoch,
                             'lr': optimizer.param_groups[0]['lr'],
                             'err': test_metrics,
                             'metric_used_to_save': metric_used_to_save,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()},
                            is_best=False,
                            save_all=True,
                            file_name=m_path)
        is_best = False
        print("=========================")


if __name__ == "__main__":
    import argparse
    import torch

    torch.cuda.set_device(0)  # VERY IMPORTANT RUN THIS CODE ONLY WITH ONE VISIBLE GPU
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='path to config via YAML file')
    args = parser.parse_args()
    option = yaml_utils.load_yaml(args.data_dir, class_mode=True)
    main(args.data_dir, option)
    # python3 main.py "config/cmu_remote.yaml"