#!/usr/bin/env python3
import os
import time
from pathlib import Path

import numpy as np

from . import analysis
from . import environment  # train, test, load_params_from_model_path
from . import loaders
from .models import choose_net as net
from .utils import yaml_utils

CURR_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


def main(args, opt):
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
    ############################################
    # Call Model
    architecture = opt.architecture_config.model
    model = net.choose_net(architecture, opt).cuda()
    model.eval()
    # Learning process
    print(">>> total params: {:.2f}K".format(sum(p.numel() for p in model.parameters()) / 1000.0))
    # Learning process
    if Path(opt.general_config.load_model_path).exists():
        updates = environment.load_params_from_model_path(opt.general_config.load_model_path, model)
        start_epoch = updates["epoch"]
        err_best = updates["err"]
        model = updates["model"]
        print("model loaded...")
        print(f'last best epoch was {start_epoch} with error {err_best["mpjpe"]}')
        # import torch
        # opt.general_config.load_model_path = "/home/eme/Projects/STSGCN/checkpoints/CKPT_3D_H36M/h36_3d_25frames_ckpt"
        # ckpt = torch.load(opt.general_config.load_model_path)
        # model.load_state_dict(ckpt)
    else:
        print(f'model path: {opt.general_config.load_model_path}')
        raise ValueError("Invalid model path!! It does not exist")

    if args.compute_flops:
        flops = analysis.compute_flops_pytorch_model(model,
                                                     input_sz=(1, opt.architecture_config.model_params.input_n,
                                                               opt.architecture_config.model_params.joints, 3))
        print(f'total flops: {flops["total"] / 1e6:.1f}M')

    # security level
    root_folder = CURR_DIR.joinpath(opt.general_config.load_model_path).parent.parent
    figures_path = root_folder.joinpath("figures")
    massive_tests_path = root_folder.joinpath("massive_tests")
    robustness_test_path = massive_tests_path.joinpath("robustness_test")
    if hasattr(opt.evaluation_config, "outputs_path"):
        if opt.evaluation_config.outputs_path != "":
            root_folder = CURR_DIR.joinpath(opt.evaluation_config.outputs_path)
            print(f'root folder: {root_folder}')
            root_folder.mkdir(parents=True, mode=0o770, exist_ok=True)
            figures_path = root_folder.joinpath("figures")
            massive_tests_path = root_folder.joinpath("massive_tests")
            robustness_test_path = massive_tests_path.joinpath("robustness_test")
    print(f'save files in output folder: {root_folder}')
    figures_path.mkdir(parents=True, mode=0o770, exist_ok=True)
    robustness_test_path.mkdir(parents=True, mode=0o770, exist_ok=True)
    if list(massive_tests_path.glob("*.xlsx")):
        print("Warning: This folder is not empty. All figures and excels will be replaced")
    # data loadig
    print(">>> loading data")
    times = []
    # CAREFUL Check TRAINset augmentations when run this code. MUST BE EMPTY/CLEAN.
    if not isinstance(opt.evaluation_config.sets, list):
        raise ValueError("opt.evaluation_config.sets is not a list type")
    db_sets = [set.__dict__ for set in opt.evaluation_config.sets]
    times.append(time.time())
    for typ in db_sets:
        db_set = list(typ.keys())[0]  # must have only one set.
        typ[db_set].name = db_set  # PATCH TO HAVE THE SPLIT NAME
        typ = typ[db_set]
        actions = "all" if typ.classes == ["all"] else typ.classes
        if isinstance(actions, str):  ### CASE 1 # ALL NOT SUGGESTED TO USE HERE
            loader, actions = loaders.get_loader_divided_by_actions(db, typ, opt,
                                                                    shuffle=False,
                                                                    return_class=True)
        elif isinstance(actions, list):  ### CASE 2 # BY MOTION CLASS
            loader = {}
            for act in actions:
                loader[act] = loaders.get_loader(opt, split=typ,
                                                 model=opt.architecture_config.model,
                                                 return_all_joints=return_all_joints,
                                                 actions=act, shuffle=False, return_class=True)
        else:
            raise ValueError(f'Input format not recognized:', actions)
        metrics = {}
        for a in actions:
            print(f'=========== db_set:{db_set}, action:{a} ===========')
            metrics[a] = environment.test(loader[a], model,
                                          get_all_samples=opt.environment_config.evaluate_from,
                                          compute_joint_error=True,
                                          unnormalize=loaders.get_dataset_stats(
                                              db) if opt.learning_config.normalize else None,
                                          db=db,
                                          output_n=opt.architecture_config.model_params.output_n,
                                          joints_n=opt.architecture_config.model_params.joints,
                                          adversarial_attacks=typ.adversarial_attack if \
                                              hasattr(typ, "adversarial_attack") else None,
                                          )
        print("\n\n")

        if opt.evaluation_config.outputs_path != "":
            root_folder_t = root_folder
        else:
            root_folder_t = root_folder.parent.parent
        # Record results. # Write excel here.
        if args.robustness_test:
            save_folder_path = robustness_test_path
        else:
            save_folder_path = root_folder.parent#.parent
        for typi in typ.evaluate:
            file_name = f'{save_folder_path.joinpath(typi)}_{db_set}.xlsx'
            if hasattr(typ, "extension_path"):
                if typ.extension_path:
                    file_name = f'{save_folder_path.joinpath(typi)}_{db_set}_{typ.extension_path}.xlsx'
                else:
                    file_name = f'{save_folder_path.joinpath(typi)}_{db_set}.xlsx'
            analysis.record_sheet(metrics, file_name, compute=typi, skeleton_type=db)
        times.append(time.time())

        # Drawing data
        if hasattr(typ, "visualization"):
            # batches = metrics[a]["pred"].shape[0]
            samples = typ.visualization.action_batch_samples
            args = typ.visualization.__dict__
            args["db"] = db
            del typ.visualization.action_batch_samples
            for b in range(samples):
                for a in actions:
                    print(f'action:{a} , batch:{b}/{samples}', end=" - ")
                    gif_path = f'{a.replace("/", ".")}_{b:03}.gif'
                    pred = metrics[a]["pred"]  # This metrics is the last one processed in the previous stage (above).
                    target = metrics[a]["target"]
                    inputs = metrics[a]["inputs"]
                    # To plot full sequence sample
                    ##############
                    target = np.concatenate((inputs, target), axis=1)
                    pred = np.concatenate((np.zeros_like(inputs), pred), axis=1)
                    ##############
                    # Plot 3D mesh. # scaling previous version.
                    analysis.create_animation(figures_path.joinpath(gif_path),
                                              [target[b], pred[b]],
                                              **args)
                    print(f'gif generated on: {figures_path.joinpath(gif_path)}')

    for i, db_set in enumerate(db_sets):
        db_set_key = [key for key in db_set][0]
        print(f'{db_set_key}: {times[i + 1] - times[i]}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data_config', type=str, help='path to config via YAML file')
    parser.add_argument('--compute-flops', action='store_true', help='compute flops')
    parser.add_argument('--online-plot', action='store_true', help='plot figure in local computer')
    parser.add_argument('--robustness_test', action='store_true', help='robustness test')
    args = parser.parse_args()
    option = yaml_utils.load_yaml(args.data_config, class_mode=True)
    main(args, option)
    # python3 main.py "config/cmu_remote.yaml"