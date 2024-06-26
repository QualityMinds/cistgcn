from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from ..utils import data_utils

torch.manual_seed(0)


class ExPI_Motion3D(Dataset):  # This class must be cleaned, e.g. dct_n was removed.
    def __init__(self, path_to_data, actions, **kwargs):
        output_n = 10
        split = "train"
        data_mean = 0
        data_std = 0
        dim_used = None
        normalize = False
        return_all_joints = False
        return_class = False
        config = "pro3"
        self.class_seq = None
        if kwargs.get("input_n"): input_n = kwargs.get("input_n")
        if kwargs.get("output_n"): output_n = kwargs.get("output_n")
        if kwargs.get("split"): split = kwargs.get("split")
        if kwargs.get("data_mean") is not None: data_mean = kwargs.get("data_mean")
        if kwargs.get("data_std") is not None: data_std = kwargs.get("data_std")
        if kwargs.get("dim_used") is not None: dim_used = kwargs.get("dim_used")
        if kwargs.get("normalize") is not None: normalize = kwargs.get("normalize")
        if kwargs.get("transform"): transform = kwargs.get("transform")
        if kwargs.get("return_all_joints"): return_all_joints = kwargs.get("return_all_joints")
        if kwargs.get("return_class"): return_class = kwargs.get("return_class")
        if kwargs.get("config"): config = kwargs.get("config")

        if not (split == "train" or split == "test" or split == "original_test"):
            raise ValueError("Undefined value was given for split")

        self.path_to_data = path_to_data
        self.split = split
        self.input_n = input_n
        self.dim_used = dim_used
        if isinstance(actions, list):
            if len(actions) == 1: actions = actions[0]
        is_test = False if split == "train" else True

        self.path_to_data = Path(self.path_to_data)
        all_seqs, dim_ignore, dim_use, class_seq, data_mean, data_std = data_utils.load_data_expi(self.path_to_data,
                                                                                                  actions,
                                                                                                  input_n, output_n,
                                                                                                  data_std=data_std,
                                                                                                  data_mean=data_mean,
                                                                                                  is_test=self.split,
                                                                                                  config=config)
        # if all_seqs is None:
        #     self.target = np.empty(0)
        #     return None
        if not is_test:
            dim_used = dim_use

        self.transform = transform
        self.dim_used = dim_used
        self.data_mean = data_mean
        self.data_std = data_std
        if normalize:
            all_seqs = (all_seqs - data_mean) / data_std

        n, seq_len, dim_full_len = all_seqs.shape
        self.target = np.float32(all_seqs.reshape(n, seq_len, -1, 3))
        if not return_all_joints:
            self.target = self.target[:, :, self.dim_used, :]
            # self.dim_used = np.arange(self.target.shape[2])  # validate this, generates problems.
        del all_seqs
        if return_class:
            self.class_seq = np.array(class_seq)

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, item):  # preprocess_seq and all_seqs are the same but sample is dct preprocessed.
        data = self.target[item]
        if self.transform:
            self.proc_data = self.transform(data).float()
            self.velocities = np.diff(self.proc_data, axis=0)
            self.global_vel = np.linalg.norm(self.velocities, axis=-1, keepdims=True)
        return {"sample": self.proc_data[:self.input_n],
                "target": self.proc_data[self.input_n:],
                "target_vel": self.velocities[self.input_n - 1:].cumsum(0),
                "target_gvel": self.global_vel[self.input_n - 1:].cumsum(0),
                "original": data,
                "processed": self.proc_data,
                "item": item,
                }


if __name__ == "__main__":
    # Stop a debug breakpoint in return {"sample":
    data = []  # comes from code above
    self = {}  # comes from code above
    import analysis

    root_folder = Path("./")
    ##############
    # Plot 3D mesh. # scaling previous version.
    analysis.create_animation(root_folder.joinpath(f'test1.gif'),
                              [data, self.proc_data],
                              mode="test",  # also it is possible to change to train to get only 1 view.
                              plot_joints=True,
                              # [-2, 2],
                              online_plot=True)
