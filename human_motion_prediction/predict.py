from pathlib import Path

import numpy as np

from . import analysis
from . import environment  # train, test, load_params_from_model_path
from . import loaders
from .models import choose_net as net
from .utils import yaml_utils


def main(args, opt):
    if "cmu" in opt.general_config.data_dir:
        db = "cmu"
    elif "h3" in opt.general_config.data_dir:
        db = "h36m"
    elif "3d" in opt.general_config.data_dir:
        db = "3dpw"
        use_actions = False
    # create model
    print(">>> creating model")

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
        raise ValueError("Invalid model path!! It does not exist")

    # security level
    root_folder = Path(opt.general_config.load_model_path)
    predict_path = root_folder.parent.parent.joinpath("predict")
    if hasattr(opt.evaluation_config, "outputs_path"):
        if opt.evaluation_config.outputs_path != "":
            root_folder = Path(opt.evaluation_config.outputs_path)
            predict_path = root_folder.joinpath("predict")
    predict_path.mkdir(parents=True, mode=0o770, exist_ok=True)
    if list(root_folder.glob("*.xlsx")):
        print("Warning: This folder is not empty. All figures and excels will be replaced")

    # data loadig
    print(">>> loading data")
    ###### train set
    db_sets = [set.__dict__ for set in opt.evaluation_config.sets]
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
                loader[act] = loaders.get_loader(opt, split=typ, model=opt.architecture_config.model,
                                                 return_all_joints=opt.environment_config.return_all_joints,
                                                 actions=act,
                                                 shuffle=False, return_class=True)
        else:
            AssertionError(f'Input format not recognized')

        idxs = typ.index
        print(">>> performing inference")
        metrics = {}
        for act in actions:
            print(f'=========== db_set:{db_set}, action:{act} ===========')
            metrics[act] = environment.test(loader[act], model,
                                            get_all_samples=opt.environment_config.evaluate_from,
                                            compute_joint_error=True,
                                            unnormalize=loaders.get_dataset_stats(
                                                db) if opt.learning_config.normalize else None,
                                            db=db,
                                            output_n=opt.architecture_config.model_params.output_n,
                                            joints_n=opt.architecture_config.model_params.joints,
                                            get_interpretation=opt.evaluation_config.interpretation.layers if \
                                                hasattr(opt.evaluation_config, "interpretation") else None,
                                            adversarial_attacks=typ.adversarial_attack if \
                                                hasattr(typ, "adversarial_attack") else None,
                                            idx=idxs)

        # save raw outputs to analyze data processing (ongoing)
        # metrics["all"]["classes"] = loader["all"].dataset.class_seq
        # np.save("20221111_1223-id0734__original_test.npy", metrics)
        print("\n\n")
        print(">>> saving interpretation figures")
        for act in actions:
            for idx, sample_idx in enumerate(idxs):
                pred = metrics[act]["pred"][idx]
                target = metrics[act]["target"][idx]
                inputs = metrics[act]["inputs"][idx]
                # Plot interpretations:
                if hasattr(opt.evaluation_config, "interpretation"):
                    for k in metrics[act]["interpretation"]:
                        print(k, end="")
                        internal_layer = np.array(metrics[act]["interpretation"][k])
                        batch_rsz = internal_layer.shape[0] * internal_layer.shape[1]
                        internal_layer = np.reshape(internal_layer, (batch_rsz, *internal_layer.shape[2:]))[idx]
                        analysis.plot_interpretations(internal_layer,
                                                      predict_path.joinpath(f'{act}_{sample_idx:06}__{k}.png'),
                                                      title=k,
                                                      db=db,
                                                      dim_used=loader[act].dataset.dim_used)
                        print(k, end="[X]  ")
                print()
                # To plot full sequence sample
                ##############
                target = np.concatenate((inputs, target), axis=0)
                pred = np.concatenate((np.zeros_like(inputs), pred), axis=0)
                ##############
                if hasattr(typ, "visualization"):
                    args = typ.visualization.__dict__
                    args["db"] = db
                    # Plot 3D mesh.
                    analysis.create_animation(predict_path.joinpath(f'{act}_{sample_idx:06}.gif'),
                                              [target, pred],
                                              **args)
                    output_model = {"full_data": pred,
                                    "target": target,
                                    }
                    if hasattr(metrics[act], "interpretation"):
                        output_model["interpretation"] = metrics[act]["interpretation"]  # is always a list [0].
                    np.save(f'{predict_path}_{act}_{sample_idx}', output_model)

                    ################ PAPER GRAPHICS ################
                    if hasattr(opt.evaluation_config, "mode"):
                        if opt.evaluation_config.mode.type == "paper":
                            for idx, sample_idx in enumerate(idxs):
                                pred = metrics[act]["pred"][
                                    idx]  # This metrics is the last one processed in the previous stage (above).
                                target = metrics[act]["target"][idx]
                                inputs = metrics[act]["inputs"][idx]
                                ##############
                                target = np.concatenate((inputs, target), axis=0)
                                pred = np.concatenate((np.zeros_like(inputs), pred), axis=0)
                                ##############
                                args = opt.evaluation_config.mode.visualization.__dict__
                                args["db"] = db
                                # Plot 3D mesh. # scaling previous version.
                                analysis.create_animation(predict_path.joinpath(f'{act}_{sample_idx:06}_paper.gif'),
                                                          [target, pred],
                                                          **args)
        #### Stop here or could fail is paper gif were not correctly generated.
        # This is still on evaluation.
        if hasattr(opt.evaluation_config, "mode"):
            if opt.evaluation_config.mode.type == "paper":
                n_input = opt.evaluation_config.mode.input_n
                time_ms = opt.evaluation_config.mode.times
                generated_gifs = [str(f) for f in list(predict_path.rglob('*.gif')) if "_paper" in str(f)]
                print("please select only 4 for this app, this is still on validation")
                print(f'size of the total _paper.gif is: {len(generated_gifs)}')
                for gif in generated_gifs:
                    print(gif)
                    images = analysis.extract_images_from_gif(gif, return_images=True)
                    # This is pure speculation to remove white regions. normal values are: 220:-220, 320:-320
                    images = np.array(images)[n_input:, 220:-220, 320:-320]  # 320 + 100:-320 + 100
                    images = images[time_ms]
                    images = np.transpose(images, (1, 0, 2, 3)).reshape(images.shape[1], -1, 3)
                    fig = analysis.Image.fromarray(images)
                    IX, IY = fig.size
                    fig = fig.resize((IX // 2, IY // 2), analysis.Image.Resampling.LANCZOS)
                    fig.save((gif[:-4] + ".eps").replace("_paper", ""), optimize=True, quality=95)
                    fig.close()
    print("finished")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='path to config via YAML file')
    parser.add_argument('--online-plot', action='store_true', help='plot figure in local computer')
    parser.add_argument('--robustness_test', action='store_true', help='robustness test')
    args = parser.parse_args()
    option = yaml_utils.load_yaml(args.data_dir, class_mode=True)
    main(args, option)
    # python3 main.py "config/cmu_remote.yaml"