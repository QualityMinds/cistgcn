import yaml


class Struct(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [Struct(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, Struct(b) if isinstance(b, dict) else b)


def RemoveStruct(opt):
    opt1 = {}
    if not isinstance(opt, (dict, Struct)):
        if not isinstance(opt, (int, str, float)):
            opt = opt.tolist()
        return opt
    for k in list(opt.__dict__.keys()):
        if isinstance(getattr(opt, k), Struct):
            opt1[k] = RemoveStruct(getattr(opt, k))
        elif isinstance(getattr(opt, k), list):
            opt_list = []
            for i in range(len(getattr(opt, k))):
                opt_list.append(RemoveStruct(getattr(opt, k)[i]))
            opt1[k] = opt_list
        else:
            opt1[k] = getattr(opt, k)
    return opt1


# Read YAML file
def load_yaml(path, class_mode=False):
    """ A function to load YAML file"""
    with open(path, 'r') as stream:
        data = yaml.safe_load(stream)
    if class_mode:
        data = Struct(data)
    return data


# Write YAML file
def write_yaml(data, path="default_output.yaml", remote_struct=True):
    """ A function to write YAML file"""
    if remote_struct:
        data = RemoveStruct(data)
    with open(path, 'w') as f:
        yaml.dump(data, f)


if __name__ == '__main__':
    # Original values with some modifications for testing.
    meta_config = {"version": "0.1.1",
                   "task": "3d keypoint prediction",
                   "comment": "Testing a new architecture based on JDM paper.",
                   "project": "Attention"}
    general_config = {"data_dir": "../data/ann_cmu_mocap",
                      "experiment_name": "architecture-tests",
                      "log_path": "logdir/",
                      "model_name_rel_path": "SpatiotemporalGCN-benchmark",
                      "save_all_intermediate_models": False,
                      "save_models": True,
                      "load_model_path": "",
                      "tensorboard": {"num_mesh": 4,
                                      }
                      }
    architecture_config = {"model": "SpatiotemporalGCN",
                           "model_params": {"input_n": 10,
                                            "output_n": 10,
                                            "joints": 25,
                                            "temporal_hidden_units": 15,
                                            }
                           }
    learning_config = {"augmentations": {"random_rotation": {"x": [-180, 180],
                                                             "y": "",
                                                             "z": ""},
                                         "random_translation": {"x": [-0.01, 0.01],
                                                                "y": [-0.01, 0.01],
                                                                "z": [-0.01, 0.01]},
                                         "random_Scale": {"x": [-0.05, 0.05],
                                                          "y": [-0.05, 0.05],
                                                          "z": [-0.05, 0.05]},
                                         "random_noise": 0.025,
                                         },
                       "dropout": 0.5,
                       "max_norm": True,
                       "lr": 1.0e-3,
                       "scheduler": {"type": "StepLR",
                                     "params": {"step_size": 5000,
                                                "gamma": 0.96, }
                                     },
                       "WarmUp": 2500,
                       }
    environment_config = {"evaluate_from": 0,
                          "actions": "all",
                          "train_batch": 16,
                          "test_batch": 64,
                          "job": 16,
                          "sample_rate": 2,
                          "is_norm": True, }
    conversion_from_init_config = {"general_config": general_config,
                                   "architecture_config": architecture_config,
                                   "learning_config": learning_config,
                                   "environment_config": environment_config,
                                   "meta_config": meta_config, }
    write_yaml(conversion_from_init_config)
    print("done")
