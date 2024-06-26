import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from . import environment  # train, test, load_params_from_model_path
from . import loaders
from .analysis import analysis_utils
from .models import choose_net as net
from .utils import yaml_utils


def main(yaml_file, opt):
    curr_time = datetime.utcnow().strftime('%Y%m%d_%H%M-id%f')[:-2]
    if "cmu" in opt.general_config.data_dir:
        db = "cmu"
    elif "h3" in opt.general_config.data_dir:
        db = "h36m"
    elif "3d" in opt.general_config.data_dir:
        db = "3dpw"
    elif "amas" in opt.general_config.data_dir:
        db = "amass"
    elif "ex" in opt.general_config.data_dir:
        db = "expi"
    dim_used = json.load(open(f'human-motion-prediction/stats/{db}_train_stats.json'))["dim_used"]
    print(f'folder name: {curr_time}')

    if hasattr(opt, "architecture_config"):
        if hasattr(opt.architecture_config, "model"):
            if opt.architecture_config.model is not None:
                if hasattr(opt.general_config, "load_model_path"):
                    input_n = opt.architecture_config.model_params.input_n
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
                    else:
                        raise ValueError("Invalid model path!! It does not exist")
    ############################################
    # Output path in data analysis is mandatory.
    if not hasattr(opt.evaluation_config, "outputs_path"):
        raise NotImplementedError(f'outputs_path variable must be defined')
    outout_path = Path(opt.evaluation_config.outputs_path)  # Output folder # logdir is suggested.
    if not outout_path.exists():
        outout_path.mkdir(parents=True, mode=0o770, exist_ok=True)
    # data loadig
    print(">>> loading data")
    ###### train set
    for typ in opt.evaluation_config.sets:
        ######### Setting section #########
        db_set = list(typ.__dict__.keys())[0]  # must have only one set.
        typ = typ.__dict__[db_set]
        actions = "all" if typ.classes == ["all"] else typ.classes
        valid_j = typ.joints  # joint names to analyze.
        indices_to_eval = typ.index  # samples to analyze - Of course they change the order when different set are read
        print(f'analysis on: {db_set}')
        figs_path = outout_path.joinpath(f'{db_set}')
        if not figs_path.exists():
            figs_path.mkdir(parents=True, mode=0o770, exist_ok=True)
        ######### Setting section #########

        if isinstance(actions, str):  ### CASE 1 # ALL NOT SUGGESTED TO USE HERE
            loader, actions = loaders.get_loader_divided_by_actions(db, db_set, opt,
                                                                    shuffle=False,
                                                                    return_class=True)
        elif isinstance(actions, list):  ### CASE 2 # BY MOTION CLASS
            loader = {}
            for act in actions:
                loader[act] = loaders.get_loader(opt, split=db_set, model=opt.architecture_config.model,
                                                 return_all_joints=opt.environment_config.return_all_joints,
                                                 actions=act,
                                                 shuffle=False, return_class=True)
        else:
            raise ValueError(f'Input format not recognized')

        ###### By actions in test set.
        model_prediction = False
        if hasattr(typ, "plot_model_prediction"):
            if typ.plot_model_prediction:
                model_prediction = True
        for act in actions:
            plot = analysis_utils.SequenceAnalytics(loader[act],
                                                    db=db,
                                                    dim_used=dim_used,
                                                    remove_temporal_data=True)
            if hasattr(typ, "visualization"):
                act_fig_path = figs_path.joinpath(act, "gifs")
                if not act_fig_path.exists():
                    act_fig_path.mkdir(parents=True, mode=0o770, exist_ok=True)
                print(f'plotting GIF figure')
                for idx in indices_to_eval:
                    if idx >= len(loader[act].dataset.target):
                        print(f'Current index is not valid for this dataset -> idx:{idx} - db:{db_set} - act:{act}')
                        continue
                    args = typ.visualization.__dict__
                    args["db"] = db
                    args["dim_used"] = dim_used
                    plot.plotGIF_sequence(name=act_fig_path.joinpath(f'{idx}_{act}.gif'),
                                          idx=idx,
                                          fig_args=args)
                    plt.close()

            act_fig_path = figs_path.joinpath(act, "physics")
            if not act_fig_path.exists():
                act_fig_path.mkdir(parents=True, mode=0o770, exist_ok=True)
            print(f'plotting Physics representation')
            for idx in indices_to_eval:
                if idx >= len(loader[act].dataset.target):
                    print(f'Current index is not valid for this dataset -> idx:{idx} - db:{db_set} - act:{act}')
                    continue
                plot_type = typ.evaluate.index.physical.__dict__
                global_config = None
                if hasattr(typ.evaluate.index.physical, "global_config"):
                    global_config = typ.evaluate.index.physical.global_config
                    plot_type.pop("global_config")
                if hasattr(typ.evaluate.index.physical, "fig_size"):
                    fig_size = typ.evaluate.index.physical.fig_size
                    plot_type.pop("fig_size")
                for name, conf in plot_type.items():
                    plot.init_figure(size=fig_size)
                    plot.Plot2D_joint_physics(eval_physical_config=conf, idx=idx,
                                              global_config=global_config,
                                              mode=name, joints=valid_j)
                    if model_prediction:
                        outputs = model(torch.from_numpy(plot.db.data[idx:idx + 1, :input_n]).cuda())
                        if isinstance(outputs, (tuple, list)):
                            outputs = outputs[0]
                        temp_data_idx = plot.db.data[idx].copy()
                        plot.db.data[idx, input_n:] = outputs.detach().cpu().numpy()
                        plot.Plot2D_joint_physics(eval_physical_config=conf, idx=idx,
                                                  global_config=global_config, input_n=input_n,
                                                  mode=name, joints=valid_j, prediction=True)
                        plot.db.data[idx] = temp_data_idx
                    plot.show(act_fig_path.joinpath(f'{idx}_{act.replace("/", "-")}_{name}_norm.png'))
                    plt.close()
            # Quick testing PoC
            # valid_joints = plot._get_index_joints_given_names_list(valid_j)
            # for idx in idxs:
            #     plt.figure()
            #     plt.plot(plot.db.angles[idx, :, valid_joints].T)
            # plt.show()


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
