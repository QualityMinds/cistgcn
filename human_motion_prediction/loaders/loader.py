import json
from pathlib import Path
import os
from ..environment import custom_transforms as trs
import numpy as np
import torch
from ..loaders.amass_motion_3d import Amass_Motion3D
from ..loaders.cmu_motion_3d import CMU_Motion3D, CMU_Motion3D_specific_mgcn
from ..loaders.d3pw_motion_3d import D3PW_Motion3D
from ..loaders.expi_motion_3d import ExPI_Motion3D
from ..loaders.h36m_motion_3d import H36m_Motion3D
from torch.utils.data import DataLoader
from torchvision import transforms
from ..utils import data_utils


np.random.seed(0)
torch.manual_seed(0)

CURR_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent


def save_metrics(data_dataset, db):
    data_stats = {}
    data_stats["data_std"] = np.float32(data_dataset.data_std).tolist()
    data_stats["data_mean"] = np.float32(data_dataset.data_mean).tolist()
    data_stats["dim_used"] = data_dataset.dim_used.tolist()
    # Saving stats
    with open(CURR_DIR.joinpath(f'stats/{db}_train_stats.json'), 'w') as outfile:
        json.dump(data_stats, outfile, indent=4, sort_keys=True)


def get_dataset_stats(data):
    path = CURR_DIR.joinpath(f'stats/{data}_train_stats.json')
    print(f'getting stats from: {path}')
    with open(path) as json_file:
        data_stats = json.load(json_file)
    return {"data_mean": np.array(data_stats["data_mean"]),
            "data_std": np.array(data_stats["data_std"]),
            "dim_used": np.array(data_stats["dim_used"]), }


def get_transformations(opt_trs):
    transformations = []
    transformations.append(trs.ToTensor())
    if opt_trs is not None:  # All augmentations we implemented work only in Tensor types. NO OTHERS.
        if hasattr(opt_trs, "random_flip"):
            if opt_trs.random_flip != '':
                transformations.append(trs.RandomFlip(opt_trs.random_flip.x,
                                                      opt_trs.random_flip.y,
                                                      opt_trs.random_flip.z))
        if hasattr(opt_trs, "random_rotation"):
            if opt_trs.random_rotation.x != '' or \
                opt_trs.random_rotation.y != '' or \
                opt_trs.random_rotation.z != '':
                transformations.append(trs.RandomRotation(opt_trs.random_rotation.x,
                                                          opt_trs.random_rotation.y,
                                                          opt_trs.random_rotation.z))
        if hasattr(opt_trs, "random_scale"):
            if opt_trs.random_scale.x != '' or \
                opt_trs.random_scale.y != '' or \
                opt_trs.random_scale.z != '':
                transformations.append(trs.RandomScale(opt_trs.random_scale.x,
                                                       opt_trs.random_scale.y,
                                                       opt_trs.random_scale.z))
        if hasattr(opt_trs, "random_noise"):
            if opt_trs.random_noise != '':
                transformations.append(trs.RandomNoise(opt_trs.random_noise))
        if hasattr(opt_trs, "random_translation"):
            if opt_trs.random_translation.x != '' or \
                opt_trs.random_translation.y != '' or \
                opt_trs.random_translation.z != '':
                transformations.append(trs.RandomTranslation(opt_trs.random_translation.x,
                                                             opt_trs.random_translation.y,
                                                             opt_trs.random_translation.z))
        if hasattr(opt_trs, "rotation"):
            if opt_trs.rotation.x != '' or \
                opt_trs.rotation.y != '' or \
                opt_trs.rotation.z != '':
                transformations.append(trs.RandomRotation(opt_trs.rotation.x,
                                                          opt_trs.rotation.y,
                                                          opt_trs.rotation.z,
                                                          opt_trs.rotation.prob_threshold,
                                                          opt_trs.rotation.seq_idx,
                                                          opt_trs.rotation.continuous,
                                                          opt_trs.rotation.keep))
        if hasattr(opt_trs, "scale"):
            if opt_trs.scale.x != '' or \
                opt_trs.scale.y != '' or \
                opt_trs.scale.z != '':
                transformations.append(trs.RandomScale(opt_trs.scale.x,
                                                       opt_trs.scale.y,
                                                       opt_trs.scale.z,
                                                       opt_trs.scale.prob_threshold,
                                                       opt_trs.scale.seq_idx,
                                                       opt_trs.scale.continuous,
                                                       opt_trs.scale.keep))
        if hasattr(opt_trs, "noise"):
            if opt_trs.noise != '':
                transformations.append(trs.RandomNoise(opt_trs.noise.noise,
                                                       opt_trs.noise.prob_threshold,
                                                       opt_trs.noise.seq_idx,
                                                       opt_trs.noise.continuous,
                                                       opt_trs.noise.keep))
        if hasattr(opt_trs, "translation"):
            if opt_trs.translation.x != '' or \
                opt_trs.translation.y != '' or \
                opt_trs.translation.z != '':
                transformations.append(trs.RandomTranslation(opt_trs.translation.x,
                                                             opt_trs.translation.y,
                                                             opt_trs.translation.z,
                                                             opt_trs.translation.prob_threshold,
                                                             opt_trs.translation.seq_idx,
                                                             opt_trs.translation.continuous,
                                                             opt_trs.translation.keep))
        if hasattr(opt_trs, "flip"):
            if opt_trs.flip != '':
                transformations.append(trs.RandomFlip(opt_trs.flip.x,
                                                      opt_trs.flip.y,
                                                      opt_trs.flip.z,
                                                      opt_trs.flip.prob_threshold,
                                                      opt_trs.flip.seq_idx,
                                                      opt_trs.flip.keep))
        if hasattr(opt_trs, "pose_invers"):
            transformations.append(trs.RandomPoseInvers("h36m",
                                                        opt_trs.pose_invers.prob_threshold,
                                                        opt_trs.pose_invers.seq_idx,
                                                        opt_trs.pose_invers.keep))
    # TODO: More augmentations should be added for more robust analysis (later versions).
    transformations = transforms.Compose(transformations)
    return transformations


def get_meta_from_inputs(db, split, opt, actions, model):
    if hasattr(opt, "architecture_config"):
        if hasattr(opt.architecture_config, "model"):
            if opt.architecture_config.model_params is not None:
                input_n = opt.architecture_config.model_params.input_n
                output_n = opt.architecture_config.model_params.output_n
            else:
                input_n = 10
                output_n = 25
    if actions is not None:
        if isinstance(actions, str):
            actions = [actions]
        else:
            actions = actions
    else:
        actions = opt.environment_config.actions

    if CURR_DIR.joinpath(f'stats/{db}_train_stats.json').exists():
        data_stats = get_dataset_stats(db)
    # run train split to compute stats before running testing. Recursion wins :)
    elif not CURR_DIR.joinpath(f'stats/{db}_train_stats.json').exists() and split == "test":
        get_loader(opt, model=model, split="train")
        data_stats = get_dataset_stats(db)
    else:
        data_stats = {"data_mean": None,
                      "data_std": None,
                      "dim_used": None, }
    return data_stats, input_n, output_n, actions


def get_expi_dataset(split, transformations, opt, model, actions, return_all_joints=True, return_class=False):
    data_stats, input_n, output_n, actions = get_meta_from_inputs("expi", split, opt, actions, model)
    data_dataset = ExPI_Motion3D(path_to_data=opt.general_config.data_dir,
                                 actions=actions,
                                 input_n=input_n, output_n=output_n,
                                 split=split, normalize=opt.learning_config.normalize,  # not done in online manner.
                                 transform=transformations,
                                 data_mean=data_stats["data_mean"],
                                 data_std=data_stats["data_std"],
                                 dim_used=data_stats["dim_used"],
                                 return_all_joints=return_all_joints,
                                 return_class=return_class,
                                 config=opt.environment_config.protocol,
                                 )
    return data_dataset, data_stats, input_n, output_n


def get_amass_dataset(split, transformations, opt, model, actions, return_all_joints=True, return_class=False):
    data_stats, input_n, output_n, actions = get_meta_from_inputs("amass", split, opt, actions, model)
    data_dataset = Amass_Motion3D(path_to_data=opt.general_config.data_dir,
                                  actions=actions,
                                  input_n=input_n, output_n=output_n,
                                  split=split, normalize=opt.learning_config.normalize,  # not done in online manner.
                                  transform=transformations,
                                  data_mean=data_stats["data_mean"],
                                  data_std=data_stats["data_std"],
                                  dim_used=data_stats["dim_used"],
                                  return_all_joints=return_all_joints,
                                  return_class=return_class,
                                  )
    return data_dataset, data_stats, input_n, output_n


def get_pw3d_dataset(split, transformations, opt, model, actions, return_all_joints=True, return_class=False):
    data_stats, input_n, output_n, actions = get_meta_from_inputs("3dpw", split, opt, actions, model)
    data_dataset = D3PW_Motion3D(path_to_data=opt.general_config.data_dir,
                                 actions=actions,
                                 input_n=input_n, output_n=output_n,
                                 split=split, normalize=opt.learning_config.normalize,  # not done in online manner.
                                 transform=transformations,
                                 data_mean=data_stats["data_mean"],
                                 data_std=data_stats["data_std"],
                                 dim_used=data_stats["dim_used"],
                                 return_all_joints=return_all_joints,
                                 return_class=return_class,
                                 )
    return data_dataset, data_stats, input_n, output_n


def get_h36m_dataset(split, transformations, opt, model, actions, return_all_joints=True, return_class=False):
    data_stats, input_n, output_n, actions = get_meta_from_inputs("h36m", split, opt, actions, model)
    data_dataset = H36m_Motion3D(path_to_data=opt.general_config.data_dir,
                                 actions=actions,
                                 input_n=input_n, output_n=output_n,
                                 split=split, normalize=opt.learning_config.normalize,  # not done in online manner.
                                 transform=transformations,
                                 data_mean=data_stats["data_mean"],
                                 data_std=data_stats["data_std"],
                                 dim_used=data_stats["dim_used"],
                                 return_all_joints=return_all_joints,
                                 return_class=return_class,
                                 )
    return data_dataset, data_stats, input_n, output_n


def get_cmu_dataset(split, transformations, opt, model, actions, return_all_joints=True, return_class=False):
    data_stats, input_n, output_n, actions = get_meta_from_inputs("cmu", split, opt, actions, model)
    if model == "mgcn":
        dct_n = opt.architecture_config.model_params.dct_n
        data_dataset = CMU_Motion3D_specific_mgcn(path_to_data=opt.general_config.data_dir,
                                                  actions=actions,
                                                  input_n=input_n, output_n=output_n,
                                                  normalize=opt.learning_config.normalize,
                                                  split="train", dct_n=dct_n,
                                                  data_mean=data_stats["data_mean"],
                                                  data_std=data_stats["data_std"],
                                                  dim_used=data_stats["dim_used"],
                                                  )
    else:
        data_dataset = CMU_Motion3D(path_to_data=opt.general_config.data_dir,
                                    actions=actions,
                                    input_n=input_n, output_n=output_n,
                                    split=split, normalize=opt.learning_config.normalize,  # not done in online manner.
                                    transform=transformations,
                                    data_mean=data_stats["data_mean"],
                                    data_std=data_stats["data_std"],
                                    dim_used=data_stats["dim_used"],
                                    return_all_joints=return_all_joints,
                                    return_class=return_class,
                                    )
    return data_dataset, data_stats, input_n, output_n


def get_loader(opt, split, model=None, **kwargs):
    # Get train split
    if not isinstance(split, str):
        split_yaml = split
        split = split.name
    else:
        split_yaml = None
    # BUGFIX: Multiple lines does not work on our oficial pipeline. Remove comments later if needed.
    # else:
    #     if isinstance(opt.evaluation_config.sets, list):
    #         if len(opt.evaluation_config.sets) > 1:
    #             print("WARNING: THERE MORE THAN 1 SET INSIDE")
    #         split = list(opt.evaluation_config.sets[0].__dict__.keys())[0]
    #         split_yaml = getattr(opt.evaluation_config.sets[0], split)
    #     else:
    #         AssertionError(f'Input format not recognized')
    if split == "train":
        if hasattr(opt.learning_config, "augmentations"):
            opt_trs = opt.learning_config.augmentations
        else:
            opt_trs = None
        transformations = get_transformations(opt_trs)
        batch_size = opt.environment_config.train_batch
    elif split == "original_test":
        opt_trs = None
        if split_yaml is not None:
            if kwargs.get("actions") in split_yaml.classes or "all" in split_yaml.classes:
                if hasattr(split_yaml, "robustness_test"):
                    opt_trs = split_yaml.robustness_test
                else:
                    opt_trs = None  # TODO: WHICH AUGMENTATION WILL BE APPLIED HERE
        transformations = get_transformations(opt_trs)
        batch_size = opt.environment_config.test_batch
    # Get test split
    else:  # if test or anything else.
        transformations = get_transformations(None)
        batch_size = opt.environment_config.test_batch
    db = "no-data-"
    return_all_joints = True
    return_class = False
    if kwargs.get("return_all_joints") is not None:
        return_all_joints = kwargs.get("return_all_joints")
    if kwargs.get("return_class") is not None:
        return_class = kwargs.get("return_class")
    if "cmu" in opt.general_config.data_dir.lower():
        data_dataset, data_stats, input_n, output_n = get_cmu_dataset(split,
                                                                      transformations,
                                                                      opt,
                                                                      model,
                                                                      kwargs.get("actions"),
                                                                      return_all_joints=return_all_joints,
                                                                      return_class=return_class)
        db = "cmu"
    elif "h3.6m" in opt.general_config.data_dir.lower() or "h36m" in opt.general_config.data_dir.lower():
        data_dataset, data_stats, input_n, output_n = get_h36m_dataset(split,
                                                                       transformations,
                                                                       opt,
                                                                       model,
                                                                       kwargs.get("actions"),
                                                                       return_all_joints=return_all_joints,
                                                                       return_class=return_class)
        db = "h36m"
    elif "3dpw" in opt.general_config.data_dir.lower() or "pw3d" in opt.general_config.data_dir.lower():
        data_dataset, data_stats, input_n, output_n = get_pw3d_dataset(split,
                                                                       transformations,
                                                                       opt,
                                                                       model,
                                                                       kwargs.get("actions"),
                                                                       return_all_joints=return_all_joints,
                                                                       return_class=return_class)
        db = "pw3d"
    elif "amass" in opt.general_config.data_dir.lower():
        data_dataset, data_stats, input_n, output_n = get_amass_dataset(split,
                                                                        transformations,
                                                                        opt,
                                                                        model,
                                                                        kwargs.get("actions"),
                                                                        return_all_joints=return_all_joints,
                                                                        return_class=return_class)
        db = "amass"
    elif "expi" in opt.general_config.data_dir.lower():
        data_dataset, data_stats, input_n, output_n = get_expi_dataset(split,
                                                                       transformations,
                                                                       opt,
                                                                       model,
                                                                       kwargs.get("actions"),
                                                                       return_all_joints=return_all_joints,
                                                                       return_class=return_class)
        db = "expi"
    else:
        raise ValueError(f'data_dir variable was not correctly defined or has an invalid value. '
                         f'Valid values are: cmu, h36m, amass, 3dpw, expi')

    if split == "train":
        save_metrics(data_dataset, db)
    # load datasets for training or testing
    shuffle = True if split == "train" else False,
    if kwargs.get("shuffle") is not None:
        shuffle = kwargs.get("shuffle")
    loader = DataLoader(dataset=data_dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=opt.environment_config.job,
                        pin_memory=True)

    print("data loaded successfully!!!")
    print(f'{split} data: {data_dataset.__len__()} | batches:{loader.__len__()}')
    print(f'supported input:{input_n}, '
          f'output:{output_n}')
    return loader


def get_loader_divided_by_actions(db, split, opt, **kwargs):
    if db == "cmu":
        actions = data_utils.define_actions_cmu(opt.environment_config.actions)
        actions.remove("walking_extra")
    elif db == "h36m":
        actions = data_utils.define_actions_h36m(opt.environment_config.actions)
    elif db == "3dpw" or db == "pw3d":
        actions = data_utils.define_actions_pw3d(opt.environment_config.actions)
    elif db == "amass":
        actions = data_utils.define_actions_amass(opt.environment_config.actions)
    elif db == "expi":
        actions = data_utils.define_actions_expi(opt.environment_config.actions, opt.environment_config.protocol, split)
    else:
        raise ValueError(f'data_dir variable was not correctly defined or has an invalid value. '
                         f'Valid values are: cmu, h36m, amass, 3dpw, expi')
    return_class = False
    if kwargs.get("return_class") is not None:
        return_class = kwargs.get("return_class")
    shuffle = False
    if kwargs.get("shuffle") is not None:
        shuffle = kwargs.get("shuffle")
    action_loader = {}
    remove_actions = []
    for a in actions:
        print(f'action: {a}')
        action_loader[a] = get_loader(opt, split=split, model=opt.architecture_config.model,
                                      return_all_joints=opt.environment_config.return_all_joints,
                                      actions=a, shuffle=shuffle, return_class=return_class)
        if action_loader[a].__len__() == 0:
            del action_loader[a]
            remove_actions.append(a)
    for val in remove_actions: actions.remove(val)
    return action_loader, actions
