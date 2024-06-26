"""
copied from https://github.com/Arthur151/SOTA-on-monocular-3D-pose-and-shape-estimation/blob/master/evaluation_matrix.py
modified by QualityMinds GmbH
"""
import numpy as np
import torch
from ..utils import body_utils
from ..utils import data_utils

SmoothL1Loss = torch.nn.SmoothL1Loss(reduction="none")


class LossOperator():
    def __init__(self):
        self.loss = []

    def average(self, seq_len):
        self.loss[-seq_len:] = [sum(self.loss[-seq_len:]) / seq_len]

    def append(self, val):
        self.loss.append(val)

    def mean(self, axis=None):
        return np.mean(np.vstack(self.loss), axis)

    def __len__(self):
        return len(self.loss)

    def get_all(self):
        loss = np.vstack(self.loss)
        if len(loss.shape) > 3:  # Temporal information must be included in the loss.
            loss = loss.reshape(-1, *loss.shape[2:])
        return loss


def rmpjpe(predicted, target, w=None, dim=-1, reduce_axis=[]):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    # update loss and testing errors
    if isinstance(reduce_axis, (list, tuple, int)):
        error = torch.sqrt(torch.mean(torch.norm(predicted - target, 2, dim=dim), reduce_axis))
    else:
        error = torch.sqrt(torch.norm(predicted - target, 2, dim=dim))
    return error


def mpjpe(predicted, target, w=None, dim=-1, reduce_axis=[]):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    # update loss and testing errors
    if isinstance(reduce_axis, (list, tuple, int)):
        error = torch.mean(torch.norm(predicted - target, 2, dim=dim), reduce_axis)
    else:
        error = torch.norm(predicted - target, 2, dim=dim)
    return error


def weighted_mpjpe(predicted, target, w=None, dim=-1, reduce_axis=[]):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]

    # update loss and testing errors
    if isinstance(reduce_axis, (list, tuple, int)):
        error = torch.mean(w * torch.norm(predicted - target, 2, dim=dim), reduce_axis)
    else:
        error = w * torch.norm(predicted - target, 2, dim=dim)
    return error


def pa_mpjpe(predicted, target, w=None, dim=-1, reduce_axis=[], return_conversion=False):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    PA-MPJPE: MPJPE after rigid alignment of the prediction with ground truth
    using Procrustes Analysis (MPJPE-PA)
    Modified to work for sequences. BxSxNx3
    """
    assert predicted.shape == target.shape

    muX = torch.mean(target, dim=2, keepdim=True)
    muY = torch.mean(predicted, dim=2, keepdim=True)

    X0 = target - muX
    Y0 = predicted - muY
    X0[X0 ** 2 < 1e-6] = 1e-3

    normX = torch.sqrt(torch.sum(X0 ** 2, dim=(-1, -2), keepdim=True))
    normY = torch.sqrt(torch.sum(Y0 ** 2, dim=(-1, -2), keepdim=True))

    normX[normX < 1e-3] = 1e-3

    X0 /= normX
    Y0 /= normY

    H = torch.matmul(X0.transpose(-1, -2), Y0)

    U, s, Vt = torch.linalg.svd(H)
    V = Vt.permute((0, 1, 3, 2)).cuda()
    R = torch.matmul(V, U.transpose(3, 2))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    R = R.cpu()
    R = torch.det(R)
    sign_detR = torch.sign(R)
    sign_detR = sign_detR.cuda()
    torch.cuda.empty_cache()

    V[:, :, -1] *= sign_detR.unsqueeze(-1)
    s[:, :, -1] *= sign_detR.view(s.shape[0], -1)
    R = torch.matmul(V, U.transpose(3, 2))  # Rotation

    tr = torch.unsqueeze(torch.sum(s, dim=2, keepdim=True), 3)

    a = tr * normX / normY  # Scale
    t = muX - a * torch.matmul(muY, R)  # Translation

    if (a != a).sum() > 0:
        print('NaN Error!!')
        # print('UsV:', U, s, Vt)
        # print('aRt:', a, R, t)
    a[a != a] = 1.
    R[R != R] = 0.
    t[t != t] = 0.

    # Perform rigid transformation on the input
    predicted_aligned = a * torch.matmul(predicted, R) + t

    # update loss and testing errors
    if isinstance(reduce_axis, (list, tuple, int)):
        error = torch.sqrt(((predicted_aligned - target) ** 2).sum(dim)).mean(reduce_axis)
    else:
        error = torch.sqrt(((predicted_aligned - target) ** 2).sum(dim))
    if return_conversion:
        return error, predicted_aligned, (a, R, t)
    return error


def n_mpjpe(predicted, target, w=None, dim=-1, reduce_axis=[]):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape

    norm_predicted = torch.mean(torch.sum(predicted ** 2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target * predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted

    # update loss and testing errors
    n_mpjpe = mpjpe(scale * predicted, target, dim=dim, reduce_axis=reduce_axis)
    return n_mpjpe


def mean_velocity_error(predicted, target, w=None, seq_dim=1, dim=-1, reduce_axis=[]):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape

    velocity_predicted = torch.diff(predicted, axis=seq_dim)
    velocity_target = torch.diff(target, axis=seq_dim)

    # update loss and testing errors
    if isinstance(reduce_axis, (list, tuple, int)):
        mean_velocity_error = torch.mean(torch.linalg.norm(velocity_predicted - velocity_target, dim=dim), reduce_axis)
    else:
        mean_velocity_error = torch.linalg.norm(velocity_predicted - velocity_target, dim=dim)
    return mean_velocity_error


def mean_angles_error(predicted, target, w=None, dim=-1, reduce_axis=[]):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    b, seq, N, _ = target.shape
    # get euler angles from expmap
    pred_eul = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(predicted.view(-1, 3)))
    pred_eul = pred_eul.view(-1, seq, N, 3)
    targ_eul = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(target.view(-1, 3)))
    targ_eul = targ_eul.view(-1, seq, N, 3)

    # update loss and testing errors
    if isinstance(reduce_axis, (list, tuple, int)):
        mean_angle_error = torch.mean(torch.norm(pred_eul - targ_eul, 2, dim=dim), reduce_axis)
    else:
        mean_angle_error = torch.norm(pred_eul - targ_eul, 2, dim=dim)
    return mean_angle_error


def bone_length_error(predicted, target, w=None, dim=-1, reduce_axis=[], skeleton_type="cmu", dim_used=None):
    assert predicted.shape == target.shape

    bones, joint_names = body_utils.get_reduced_skeleton(skeleton_type, dim_used=dim_used)
    bones_predicted = predicted[:, :, bones, :]
    dist_predicted = torch.norm(bones_predicted[:, :, :, 0, :] - bones_predicted[:, :, :, 1, :], p=2, dim=-1)
    bones_target = target[:, :, bones, :]
    dist_target = torch.norm(bones_target[:, :, :, 0, :] - bones_target[:, :, :, 1, :], p=2, dim=-1)
    dist_predicted = dist_predicted.unsqueeze(-1)
    dist_target = dist_target.unsqueeze(-1)

    # update loss and testing errors
    if isinstance(reduce_axis, (list, tuple, int)):
        bone_length_error = torch.mean(torch.linalg.norm(dist_predicted - dist_target, dim=dim), reduce_axis)
    else:
        bone_length_error = torch.linalg.norm(dist_predicted - dist_target, dim=dim)
    return bone_length_error


def weighted_bone_length_error(predicted, target, w=None, dim=-1, reduce_axis=[], skeleton_type="cmu", dim_used=None):
    assert predicted.shape == target.shape

    bones, joint_names = body_utils.get_reduced_skeleton(skeleton_type, dim_used=dim_used)
    bones_predicted = predicted[:, :, bones, :]
    dist_predicted = torch.norm(bones_predicted[:, :, :, 0, :] - bones_predicted[:, :, :, 1, :], p=2, dim=-1)
    bones_target = target[:, :, bones, :]
    dist_target = torch.norm(bones_target[:, :, :, 0, :] - bones_target[:, :, :, 1, :], p=2, dim=-1)
    dist_predicted = dist_predicted.unsqueeze(-1)
    dist_target = dist_target.unsqueeze(-1)
    if w is not None:
        w = w.unsqueeze(0).unsqueeze(2).tile(dist_target.shape[0], 1, dist_target.shape[2])
    else:
        w = torch.ones_like(dist_target[:, :, :, 0])
    # update loss and testing errors
    if isinstance(reduce_axis, (list, tuple, int)):
        bone_length_error = torch.mean(
            w[:, :, :dist_target.shape[2]] * torch.linalg.norm(dist_predicted - dist_target, dim=dim), reduce_axis)
    else:
        bone_length_error = w[:, :, :dist_target.shape[2]] * torch.linalg.norm(dist_predicted - dist_target, dim=dim)
    return bone_length_error


def mpjpe_soft(predicted, target, w=None, dim=-1, reduce_axis=[]):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    # update loss and testing errors
    if isinstance(reduce_axis, (list, tuple, int)):
        error = torch.mean(torch.norm(SmoothL1Loss(predicted, target), 2, dim=dim), reduce_axis)
    else:
        error = torch.norm(SmoothL1Loss(predicted, target), 2, dim=dim)
    return error


def weighted_mpjpe_soft(predicted, target, w=None, dim=-1, reduce_axis=[]):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]

    # update loss and testing errors
    if isinstance(reduce_axis, (list, tuple, int)):
        error = torch.mean(w * torch.norm(SmoothL1Loss(predicted, target), 2, dim=dim), reduce_axis)
    else:
        error = w * torch.norm(SmoothL1Loss(predicted, target), 2, dim=dim)
    return error


def test():
    r1 = np.random.rand(3, 14, 3)
    r2 = np.random.rand(3, 14, 3)
    pmpjpe = pa_mpjpe(torch.from_numpy(r1), torch.from_numpy(r2), with_sRt=False)


if __name__ == '__main__':
    test()
