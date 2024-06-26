from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from ..utils import data_utils, body_utils

torch.manual_seed(0)


class H36m_Motion3D(Dataset):  # This class must be cleaned, e.g. dct_n was removed.
    def __init__(self, path_to_data, actions, **kwargs):
        output_n = 10
        split = "train"
        data_mean = 0
        data_std = 0
        dim_used = 0
        normalize = False
        return_all_joints = False
        return_class = False
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

        if not (split == "train" or split == "test" or split == "original_test"):
            raise ValueError("Undefined value was given for split")

        self.path_to_data = path_to_data
        self.split = split
        self.input_n = input_n
        self.dim_used = dim_used
        if isinstance(actions, list):
            if len(actions) == 1: actions = actions[0]
        actions = data_utils.define_actions_h36m(actions)  # actions = ['walking']
        is_test = False if split == "train" else True

        self.path_to_data = Path(self.path_to_data, "dataset")
        all_seqs, dim_ignore, dim_use, class_seq, data_mean, data_std = data_utils.load_data_h36m(self.path_to_data,
                                                                                                  actions,
                                                                                                  input_n, output_n,
                                                                                                  data_std=data_std,
                                                                                                  data_mean=data_mean,
                                                                                                  is_test=self.split)
        if not is_test:
            dim_used = dim_use

        self.dim_repeat_22 = [9, 9, 14, 16, 19, 21]  # for post-processing
        self.dim_repeat_32 = [16, 24, 20, 23, 28, 31]  # for post-processing

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

        idxs = self.__detect_pose_inversion_in_batch()
        if len(idxs) > 0:
            print(f'-- inverted poses: {len(idxs)} poses, {len(idxs) / self.target.shape[0]:.2f} --')
            Ycentroid_batch = self.target[idxs].mean((1, 2))[:, 1]
            self.target[idxs, :, :, 1] = Ycentroid_batch[:, None, None] - self.target[idxs, :, :, 1]
        del all_seqs
        if return_class:
            self.class_seq = np.array(class_seq)

    def __detect_pose_inversion_in_batch(self, data=None):
        if data is None:
            data = self.target
        # Check if poses are inverted in the Y-axis. head-spine vector
        # TODO: Check if this is a bug from dataset or from source where data was got.
        _, joints_names = body_utils.get_reduced_skeleton(skeleton_type="h36m")
        head_joint = np.where(["Head" in j for j in joints_names])[0][0]
        hips_joint = np.where(["Site" in j for j in joints_names])[0][0]
        signs = np.sign(data[:, 0, head_joint, 1] - data[:, 0, hips_joint, 1])
        idxs = np.where(signs == -1)[0]
        return idxs

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, item):  # preprocess_seq and all_seqs are the same but sample is dct preprocessed.
        data = self.target[item]
        if self.transform:
            self.proc_data = self.transform(data).float()
            self.velocities = np.diff(self.proc_data, axis=0)
            self.global_vel = np.linalg.norm(self.velocities, axis=-1, keepdims=True)
        return {"sample": self.proc_data[:self.input_n],
                "sample_vel": self.velocities[:self.input_n],
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
    n, seq_len, dim_full_len, _ = self.target.shape
    analysis.create_animation(f'test_mine.gif',
                              [self.target.reshape(n, seq_len, -1, 3)[258],
                               self.target.reshape(n, seq_len, -1, 3)[0]],
                              mode="single",  # also it is possible to change to train to get only 1 view.
                              plot_joints=True,
                              db="h36m",
                              times=1,
                              # [-2, 2],
                              online_plot=False)
