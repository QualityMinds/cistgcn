import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from ..utils import body_utils

np.random.seed(1000)  # MUST BE SELECTED BY ARGUMENTS IN THE FUTURE.


class RandomRotation:
    """
    Perform a random rotation in the full 3D sequence. input format Seq x Joints x 3
    rotation parameters are in degrees to make life easier. :)
    """

    def __init__(self, rot_x, rot_y, rot_z, prob_threshold=0.5, seq_idx=[], continuous=False, keep=True) -> None:
        self.prob_threshold = prob_threshold
        self.seq_idx = seq_idx
        self.continuous = continuous
        self.keep = keep
        if rot_x:
            if isinstance(rot_x, int) or isinstance(rot_x, float):
                self.rot_x = np.array([rot_x, rot_x])
            else:
                self.rot_x = np.array(rot_x)
        else:
            self.rot_x = [0, 0]
        if rot_y:
            if isinstance(rot_y, int) or isinstance(rot_y, float):
                self.rot_y = np.array([rot_y, rot_y])
            else:
                self.rot_y = np.array(rot_y)
        else:
            self.rot_y = [0, 0]
        if rot_z:
            if isinstance(rot_z, int) or isinstance(rot_z, float):
                self.rot_z = np.array([rot_z, rot_z])
            else:
                self.rot_z = np.array(rot_z)
        else:
            self.rot_z = [0, 0]

    def __call__(self, data):
        """
        Args:
            data (Seq x Joints x 3): Input Data 3D Sequence.
        Returns:
            Tensor: Converted 3D sequential data.
        """
        if np.random.uniform() > self.prob_threshold:
            seq, joints, dim = data.shape
            rotx = np.float32(np.random.uniform(self.rot_x[0], self.rot_x[1]))
            roty = np.float32(np.random.uniform(self.rot_y[0], self.rot_y[1]))
            rotz = np.float32(np.random.uniform(self.rot_z[0], self.rot_z[1]))

            if self.seq_idx:
                seq_len = self.seq_idx[1] - self.seq_idx[0]
            else:
                seq_len = seq
            if self.continuous:
                rot_seq_x = torch.linspace(0, rotx, steps=seq_len)
                rot_seq_y = torch.linspace(0, roty, steps=seq_len)
                rot_seq_z = torch.linspace(0, rotz, steps=seq_len)
            else:
                rot_seq_x = torch.linspace(rotx, rotx, steps=seq_len)
                rot_seq_y = torch.linspace(roty, roty, steps=seq_len)
                rot_seq_z = torch.linspace(rotz, rotz, steps=seq_len)
            self.angles = torch.stack([rot_seq_x, rot_seq_y, rot_seq_z], dim=1)
            transformation_mat = torch.from_numpy(R.from_rotvec(self.angles, degrees=True).as_matrix()).float()
            if self.seq_idx:  # [3,7]
                pre_mat = torch.eye(3).unsqueeze(0).repeat((self.seq_idx[0], 1, 1))
                if self.keep:
                    post_mat = transformation_mat[-1].repeat((seq - self.seq_idx[1], 1, 1))
                else:
                    post_mat = torch.eye(3).unsqueeze(0).repeat((seq - self.seq_idx[1], 1, 1))
                transformation_mat = torch.cat([pre_mat, transformation_mat, post_mat])
            datac = data.mean((0, 1))
            output = torch.bmm(data - datac, transformation_mat) + datac
        else:
            output = data.clone()
        return output

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomScale:
    """
    Perform a random scale in the full 3D sequence. input format Seq x Joints x 3
    rotation parameters are in degrees to make life easier. :)
    """

    def __init__(self, scale_x, scale_y, scale_z, prob_threshold=0.5, seq_idx=[], continuous=False, keep=True) -> None:
        self.prob_threshold = prob_threshold
        self.seq_idx = seq_idx
        self.continuous = continuous
        self.keep = keep
        if scale_x:
            if isinstance(scale_x, float):
                self.scale_x = np.array([scale_x, scale_x])
            else:
                self.scale_x = np.array(scale_x)
        else:
            self.scale_x = [0, 0]
        if scale_y:
            if isinstance(scale_y, float):
                self.scale_y = np.array([scale_y, scale_y])
            else:
                self.scale_y = np.array(scale_y)
        else:
            self.scale_y = [0, 0]
        if scale_z:
            if isinstance(scale_z, float):
                self.scale_z = np.array([scale_z, scale_z])
            else:
                self.scale_z = np.array(scale_z)
        else:
            self.scale_z = [0, 0]

    def __call__(self, data):
        """
        Args:
            data (Seq x Joints x 3): Input Data 3D Sequence.
        Returns:
            Tensor: Converted 3D sequential data.
        """
        if np.random.uniform() > self.prob_threshold:
            seq, joints, dim = data.shape
            scalex = np.float32(np.random.uniform(self.scale_x[0], self.scale_x[1]))
            scaley = np.float32(np.random.uniform(self.scale_y[0], self.scale_y[1]))
            scalez = np.float32(np.random.uniform(self.scale_z[0], self.scale_z[1]))

            if self.seq_idx:
                seq_len = self.seq_idx[1] - self.seq_idx[0]
            else:
                seq_len = seq
            if self.continuous:
                scale_seq_x = torch.linspace(1, scalex, steps=seq_len)
                scale_seq_y = torch.linspace(1, scaley, steps=seq_len)
                scale_seq_z = torch.linspace(1, scalez, steps=seq_len)
            else:
                scale_seq_x = torch.linspace(scalex, scalex, steps=seq_len)
                scale_seq_y = torch.linspace(scaley, scaley, steps=seq_len)
                scale_seq_z = torch.linspace(scalez, scalez, steps=seq_len)
            self.scales = torch.stack([scale_seq_x, scale_seq_y, scale_seq_z], dim=1)[:, None, :]
            if self.seq_idx:
                pre_scales = torch.ones(3).repeat(self.seq_idx[0], 1, 1)
                if self.keep:
                    post_scales = self.scales[-1].repeat((seq - self.seq_idx[1], 1, 1))
                else:
                    post_scales = torch.ones(3).repeat(seq - self.seq_idx[1], 1, 1)
                self.scales = torch.cat([pre_scales, self.scales, post_scales])
            output = data * self.scales
        else:
            output = data.clone()
        return output

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomTranslation:
    """
    Perform a random translation (based on rates) in the full 3D sequence. input format Seq x Joints x 3
    rotation parameters are in degrees to make life easier. :)
    """

    def __init__(self, tx, ty, tz, prob_threshold=0.5, seq_idx=[], continuous=False, keep=True) -> None:
        self.prob_threshold = prob_threshold
        self.seq_idx = seq_idx
        self.continuous = continuous
        self.keep = keep
        if tx:
            if isinstance(tx, float):
                self.tx = np.array([tx, tx])
            else:
                self.tx = np.array(tx)
        else:
            self.tx = [0, 0]
        if ty:
            if isinstance(ty, float):
                self.ty = np.array([ty, ty])
            else:
                self.ty = np.array(ty)
        else:
            self.ty = [0, 0]
        if tz:
            if isinstance(tz, float):
                self.tz = np.array([tz, tz])
            else:
                self.tz = np.array(tz)
        else:
            self.tz = [0, 0]

    def __call__(self, data):
        """
        Args:
            data (Seq x Joints x 3): Input Data 3D Sequence.
        Returns:
            Tensor: Converted 3D sequential data.
        """
        if np.random.uniform() > self.prob_threshold:
            seq, joints, dim = data.shape
            tx = np.float32(np.random.uniform(self.tx[0], self.tx[1]))
            ty = np.float32(np.random.uniform(self.ty[0], self.ty[1]))
            tz = np.float32(np.random.uniform(self.tz[0], self.tz[1]))

            dist = data.max(0).values.max(0).values - data.min(0).values.min(0).values

            if self.seq_idx:
                seq_len = self.seq_idx[1] - self.seq_idx[0]
            else:
                seq_len = seq
            if self.continuous:
                t_seq_x = torch.linspace(0, tx, steps=seq_len)
                t_seq_y = torch.linspace(0, ty, steps=seq_len)
                t_seq_z = torch.linspace(0, tz, steps=seq_len)
            else:
                t_seq_x = torch.linspace(tx, tx, steps=seq_len)
                t_seq_y = torch.linspace(ty, ty, steps=seq_len)
                t_seq_z = torch.linspace(tz, tz, steps=seq_len)
            self.translation = torch.stack([t_seq_x, t_seq_y, t_seq_z], dim=1) * dist
            if self.seq_idx:
                pre_translation = torch.zeros(self.seq_idx[0], 3)
                if self.keep:
                    post_translation = self.translation[-1].repeat((seq - self.seq_idx[1], 1))
                else:
                    post_translation = torch.zeros(seq - self.seq_idx[1], 3)
                self.translation = torch.cat([pre_translation, self.translation, post_translation])
            self.translation = self.translation[:, None, :]
            output = data + self.translation
        else:
            output = data.clone()
        return output

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomFlip:
    """
    (ON VALIDATION) X AXIS CHANGES IN CMU AND H36M. TAKE CARE IN SEVERAL DATASETS.
    Perform a random flip in the full 3D sequence. input format Seq x Joints x 3
    rotation parameters are in degrees to make life easier. :)
    """

    def __init__(self, fx, fy, fz, prob_threshold=0.5, seq_idx=[], keep=True) -> None:
        self.fx = fx
        self.fy = fy
        self.fz = fz
        self.prob_threshold = prob_threshold
        self.seq_idx = seq_idx
        self.keep = keep

    def __call__(self, data):
        """
        Args:
            data (Seq x Joints x 3): Input Data 3D Sequence.
        Returns:
            Tensor: Converted 3D sequential data.
        """
        seq, joints, dim = data.shape
        centroid = data.mean((0, 1))
        output = data.clone()
        if self.fx and np.random.uniform() > self.prob_threshold:
            if self.seq_idx:
                output[self.seq_idx[0]:self.seq_idx[1], :, 0] = centroid[0] - \
                                                                (data[self.seq_idx[0]:self.seq_idx[1], :, 0] - centroid[
                                                                    0])
                if self.keep:
                    output[self.seq_idx[1]:, :, 0] = centroid[0] - \
                                                     (data[self.seq_idx[1]:, :, 0] - centroid[0])
            else:
                output[:, :, 0] = centroid[0] - (data[:, :, 0] - centroid[0])
        if self.fy and np.random.uniform() > self.prob_threshold:
            if self.seq_idx:
                output[self.seq_idx[0]:self.seq_idx[1], :, 1] = centroid[1] - \
                                                                (data[self.seq_idx[0]:self.seq_idx[1], :, 1] - centroid[
                                                                    1])
                if self.keep:
                    output[self.seq_idx[1]:, :, 1] = centroid[1] - \
                                                     (data[self.seq_idx[1]:, :, 1] - centroid[1])
            else:
                output[:, :, 1] = centroid[1] - (data[:, :, 1] - centroid[1])
        if self.fz and np.random.uniform() > self.prob_threshold:
            if self.seq_idx:
                output[self.seq_idx[0]:self.seq_idx[1], :, 2] = centroid[2] - \
                                                                (data[self.seq_idx[0]:self.seq_idx[1], :, 2] - centroid[
                                                                    2])
                if self.keep:
                    output[self.seq_idx[1]:, :, 2] = centroid[2] - \
                                                     (data[self.seq_idx[1]:, :, 2] - centroid[2])
            else:
                output[:, :, 2] = centroid[2] - (data[:, :, 2] - centroid[2])
        return output

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomPoseInvers:
    """
    Perform a pose inversion in the full 3D sequence. input format Seq x Joints x 3
    pose inversion: left pose change to right pose and vice versa.
    """

    def __init__(self, skeleton_type, prob_threshold=0.5, seq_idx=[], keep=True) -> None:
        self.skeleton_type = skeleton_type
        self.prob_threshold = prob_threshold
        self.seq_idx = seq_idx
        self.keep = keep
        self.inverse_mapping, _ = body_utils.get_reduced_skeleton(skeleton_type, inverse=True)

    def __call__(self, data):
        """
        Args:
            data (Seq x Joints x 3): Input Data 3D Sequence.
        Returns:
            Tensor: Converted 3D sequential data.
        """
        if np.random.uniform() > self.prob_threshold:
            if self.seq_idx:
                for x, y in self.inverse_mapping:
                    if self.keep:
                        data_temp_x = data[self.seq_idx[0]:, x, :].clone()
                        data_temp_y = data[self.seq_idx[0]:, y, :].clone()
                        data[self.seq_idx[0]:, x, :] = data_temp_y
                        data[self.seq_idx[0]:, y, :] = data_temp_x
                    else:
                        data_temp_x = data[self.seq_idx[0]:self.seq_idx[1], x, :].clone()
                        data_temp_y = data[self.seq_idx[0]:self.seq_idx[1], y, :].clone()
                        data[self.seq_idx[0]:self.seq_idx[1], x, :] = data_temp_y
                        data[self.seq_idx[0]:self.seq_idx[1], y, :] = data_temp_x
                output = data
            else:
                for x, y in self.inverse_mapping:
                    data_temp_x = data[:, x, :].clone()
                    data_temp_y = data[:, y, :].clone()
                    data[:, x, :] = data_temp_y
                    data[:, y, :] = data_temp_x
                output = data
        else:
            output = data.clone()
        return output

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomNoise:
    """
    Perform a random noise in the full 3D sequence. input format Seq x Joints x 3
    rotation parameters are in degrees to make life easier. :)
    """

    def __init__(self, noise, prob_threshold=0.5, seq_idx=[], continuous=False, keep=True) -> None:
        self.noise = noise
        self.prob_threshold = prob_threshold
        self.seq_idx = seq_idx
        self.continuous = continuous
        self.keep = keep

    def __call__(self, data):
        """
        Args:
            data (Seq x Joints x 3): Input Data 3D Sequence.
        Returns:
            Tensor: Converted 3D sequential data.
        """
        if np.random.uniform() > self.prob_threshold:
            seq, joints, dim = data.shape
            noise = torch.from_numpy(np.random.uniform(-1, 1, (joints, dim)))

            dist = data.max(0).values.max(0).values - data.min(0).values.min(0).values

            if self.seq_idx:
                seq_len = self.seq_idx[1] - self.seq_idx[0]
            else:
                seq_len = seq
            if self.continuous:
                noise_seq = torch.linspace(0, self.noise, steps=seq_len)
            else:
                noise_seq = torch.linspace(self.noise, self.noise, steps=seq_len)
            self.noise_dist = noise_seq[:, None, None].repeat((1, 1, dim)) * noise * dist
            if self.seq_idx:
                pre_noise = torch.zeros(self.seq_idx[0], joints, dim)
                if self.keep:
                    post_noise = self.noise_dist[-1].repeat((seq - self.seq_idx[1], 1, 1))
                else:
                    post_noise = torch.zeros(seq - self.seq_idx[1], joints, dim)
                self.noise_dist = torch.cat([pre_noise, self.noise_dist, post_noise])
            output = data + self.noise_dist
        else:
            output = data.clone()
        return output

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ToTensor:
    """
    Convert a numpy to Torch tensor
    """

    def __init__(self) -> None:
        pass

    def __call__(self, data):
        """
        Args:
            data (Seq x Joints x 3): Input Data 3D Sequence.
        Returns:
            Tensor: Converted 3D sequential data.
        """
        return torch.from_numpy(data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
