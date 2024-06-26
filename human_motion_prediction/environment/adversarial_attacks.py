import numpy as np
import torch
from torch import nn

from .. import losses
from ..utils import body_utils


def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


def convert_to_dists(data1, data2, bins):
    data1 = data1.flatten(1, -1)
    data2 = data2.flatten(1, -1)
    data = torch.cat([data1, data2], 1)
    bins = tensor_linspace(data.min(1)[0], data.max(1)[0], bins).cpu()
    px = [torch.histogram(x, bins=bins[i], density=True).hist.detach().numpy() for i, x in enumerate(data1.cpu())]
    px = torch.from_numpy(np.array(px))
    qx = [torch.histogram(x, bins=bins[i], density=True).hist.detach().numpy() for i, x in enumerate(data2.cpu())]
    qx = torch.from_numpy(np.array(qx))
    return px, qx


def compute_entropy(px, qx, eps=1e-8):
    return (px * (torch.log(px + eps) - torch.log(qx + eps))).sum(1)


class CustomJSD(nn.Module):
    def __init__(self, bins, eps=1e-8):
        super().__init__()
        self.bins = bins + 1
        self.eps = eps

    def forward(self, data1, data2, dim=0):
        data1 = torch.sqrt(torch.pow(data1[..., None, :] - data1[..., None, :, :], 2).sum(-1))
        data2 = torch.sqrt(torch.pow(data2[..., None, :] - data2[..., None, :, :], 2).sum(-1))
        if dim > 0:  # B (samples) x T x J x 3 => B x N
            data1 = torch.swapaxes(data1, 0, dim)
            data2 = torch.swapaxes(data2, 0, dim)
        px, qx = convert_to_dists(data1, data2, self.bins)
        mx = (px + qx) / 2
        return (compute_entropy(px, mx, eps=self.eps) + compute_entropy(qx, mx, eps=self.eps)) / 2


class CustomKLD(nn.Module):
    def __init__(self, bins, eps=1e-8):
        super().__init__()
        self.bins = bins + 1
        self.eps = eps

    def forward(self, data1, data2, dim=0):
        data1 = torch.sqrt(torch.pow(data1[..., None, :] - data1[..., None, :, :], 2).sum(-1))
        data2 = torch.sqrt(torch.pow(data2[..., None, :] - data2[..., None, :, :], 2).sum(-1))
        if dim > 0:  # B (samples) x T x J x 3 => B x N
            data1 = torch.swapaxes(data1, 0, dim)
            data2 = torch.swapaxes(data2, 0, dim)
        px, qx = convert_to_dists(data1, data2, self.bins)
        return compute_entropy(px, qx, eps=self.eps)


class CustomKolmogorovSmirnovTest(nn.Module):
    def __init__(self, bins, eps=1e-8):
        super().__init__()
        self.bins = bins + 1
        self.eps = eps

    def forward(self, data1, data2, dim=0):
        data1 = torch.sqrt(torch.pow(data1[..., None, :] - data1[..., None, :, :], 2).sum(-1))
        data2 = torch.sqrt(torch.pow(data2[..., None, :] - data2[..., None, :, :], 2).sum(-1))
        if dim > 0:  # B (samples) x T x J x 3 => B x N
            data1 = torch.swapaxes(data1, 0, dim)
            data2 = torch.swapaxes(data2, 0, dim)
        px, qx = convert_to_dists(data1, data2, self.bins)
        px = px.cumsum(1)
        qx = qx.cumsum(1)
        return torch.max((px - qx).abs(), 1)[0]


class HausdorffDistance(nn.Module):
    def __init__(self, func_type="mean", dist_type=False):
        super().__init__()
        self.dist_type = dist_type
        if func_type == "mean":
            self.func_type = torch.mean
        elif func_type == "max":
            self.func_type = torch.max
        elif func_type == "std":
            self.func_type = torch.std
        else:
            NotImplementedError

    def forward(self, data1, data2, dim=0):
        """
        Compute the Hausdorff Distance function
         between two unordered sets of points (the function is symmetric).
        :param set1: Tensor where each row is an N-dimensional point.
        :param set2: Tensor where each row is an N-dimensional point.
        :return: The Hausdorff Distance between set1 and set2.
        """
        if self.dist_type:
            b, t, j, _ = data1.shape
            mask = [(data1[0, 0, :, 0] == a).max(0)[1].item() for a in torch.unique(data1[0, 0, :, 0])]
            mask.sort()
            data1 = torch.sqrt(torch.pow(data1[..., mask, None, :] - data1[..., None, mask, :], 2).sum(-1))
            data1 = data1.reshape([b, t, -1, 1])
            data2 = torch.sqrt(torch.pow(data2[..., mask, None, :] - data2[..., None, mask, :], 2).sum(-1))
            data2 = data2.reshape([b, t, -1, 1])
        if dim > 0:  # B (samples) x T x J x 3 => B x N
            data1 = torch.swapaxes(data1, 0, dim)
            data2 = torch.swapaxes(data2, 0, dim)
        d2_matrix = ((data1[..., None, :] - data2[..., None, :, :]) ** 2).sum(-1).sqrt()

        if self.func_type == torch.max or self.func_type == torch.min:
            Hausdorff_distance = self.func_type(self.func_type(torch.min(d2_matrix, -1)[0], -1)[0], -1)
        else:
            Hausdorff_distance = self.func_type(torch.min(d2_matrix, -1)[0], (-2, -1))

        if isinstance(Hausdorff_distance, tuple): Hausdorff_distance = Hausdorff_distance[0]

        return Hausdorff_distance


class ComputeAttackMetrics:
    def __init__(self, typ_eval="time_spatial_std_dim"):
        self.raxis = [0]
        self.typ_eval = typ_eval
        self.queries = 0  # 0 means no iterative algorithms
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.cosine_similarity_spec = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.KLD64 = CustomKLD(bins=64)
        self.JSD64 = CustomJSD(bins=64)
        self.KSTest = CustomKolmogorovSmirnovTest(bins=64)
        self.mse = nn.MSELoss(reduction='none')
        self.hausdorff_mean = HausdorffDistance("mean")
        self.hausdorff_max = HausdorffDistance("max")
        self.hausdorff_dist_mean = HausdorffDistance("mean", True)
        self.hausdorff_dist_max = HausdorffDistance("max", True)

    def _init_func(self, seq, seq_vel=None, model=None, pred_func=None, params=None, reduce_axis=[]):
        params["model"] = model
        params["inputs"] = seq
        params["inputs_vel"] = seq_vel
        outputs = pred_func(**params)  # Predict using normal inputs.

        if model.__class__.__name__ == "PGBIG":
            loss_list = []
            for output in outputs:
                loss_list.append(losses.mpjpe(output, params["target"], reduce_axis=reduce_axis))
            loss_ = torch.stack(loss_list).mean(dim=0)
        else:
            loss_ = losses.mpjpe(outputs, params["target"], reduce_axis=reduce_axis)
        if loss_.shape != []:
            loss_to_adv = loss_.mean()
        else:
            loss_to_adv = loss_
        model.zero_grad()
        loss_to_adv.backward()
        return outputs, loss_

    def _get_metrics(self, in_seq, adv_seq, in_seq_vel=None, adv_seq_vel=None):

        self.full_mpjpe = losses.mpjpe(in_seq, adv_seq, reduce_axis=self.raxis)
        self.mpjpe_spatial = self.full_mpjpe.mean(0).cpu().data.numpy()
        self.mpjpe_temporal = self.full_mpjpe.mean(1).cpu().data.numpy()
        self.mpjpe = self.full_mpjpe.mean().cpu().data.numpy()
        self.mpjpe_sample = losses.mpjpe(in_seq, adv_seq, reduce_axis=[2, 1]).cpu().data.numpy()

        self.full_n_mpjpe = losses.n_mpjpe(in_seq, adv_seq, reduce_axis=self.raxis)
        self.n_mpjpe_spatial = self.full_n_mpjpe.mean(0).cpu().data.numpy()
        self.n_mpjpe_temporal = self.full_n_mpjpe.mean(1).cpu().data.numpy()
        self.n_mpjpe = self.full_n_mpjpe.mean().cpu().data.numpy()
        self.n_mpjpe_sample = losses.n_mpjpe(in_seq, adv_seq, reduce_axis=[2, 1]).cpu().data.numpy()

        self.full_pa_mpjpe = losses.pa_mpjpe(in_seq, adv_seq, reduce_axis=self.raxis)
        self.pa_mpjpe_spatial = self.full_pa_mpjpe.mean(0).cpu().data.numpy()
        self.pa_mpjpe_temporal = self.full_pa_mpjpe.mean(1).cpu().data.numpy()
        self.pa_mpjpe = self.full_pa_mpjpe.mean().cpu().data.numpy()
        self.pa_mpjpe_sample = losses.pa_mpjpe(in_seq, adv_seq, reduce_axis=[2, 1]).cpu().data.numpy()

        self.cosine_similarity_error = self.cosine_similarity(in_seq.reshape(in_seq.shape[0], -1),
                                                              adv_seq.reshape(in_seq.shape[0], -1)).cpu().data.numpy()
        self.cosine_similarity_temporal = self.cosine_similarity_spec(in_seq, adv_seq).mean((1, 2)).cpu().data.numpy()
        self.cosine_similarity_spatial = self.cosine_similarity_spec(in_seq, adv_seq).mean((0, 2)).cpu().data.numpy()

        self.hausdorff_mean_error = self.hausdorff_mean(in_seq, adv_seq, 0).cpu().data.numpy()
        self.hausdorff_mean_temporal = self.hausdorff_mean(in_seq, adv_seq, 1).cpu().data.numpy()
        self.hausdorff_mean_spatial = self.hausdorff_mean(in_seq, adv_seq, 2).cpu().data.numpy()

        self.hausdorff_max_error = self.hausdorff_max(in_seq, adv_seq, 0).cpu().data.numpy()
        self.hausdorff_max_temporal = self.hausdorff_max(in_seq, adv_seq, 1).cpu().data.numpy()
        self.hausdorff_max_spatial = self.hausdorff_max(in_seq, adv_seq, 2).cpu().data.numpy()

        self.KLD_error = self.KLD64(in_seq, adv_seq, 0).cpu().data.numpy()
        self.KLD_temporal = self.KLD64(in_seq, adv_seq, 1).cpu().data.numpy()
        self.KLD_spatial = self.KLD64(in_seq, adv_seq, 2).cpu().data.numpy()

        self.JSD_error = self.JSD64(in_seq, adv_seq, 0).cpu().data.numpy()
        self.JSD_temporal = self.JSD64(in_seq, adv_seq, 1).cpu().data.numpy()
        self.JSD_spatial = self.JSD64(in_seq, adv_seq, 2).cpu().data.numpy()

        self.KSTest_error = self.KSTest(in_seq, adv_seq, 0).cpu().data.numpy()
        self.KSTest_temporal = self.KSTest(in_seq, adv_seq, 1).cpu().data.numpy()
        self.KSTest_spatial = self.KSTest(in_seq, adv_seq, 2).cpu().data.numpy()

        self.mse_error = self.mse(in_seq, adv_seq).mean((1, 2, 3)).cpu().data.numpy()
        self.mse_temporal = self.mse(in_seq, adv_seq).mean((0, 2, 3)).cpu().data.numpy()
        self.mse_spatial = self.mse(in_seq, adv_seq).mean((0, 1, 3)).cpu().data.numpy()

        # compute average error for MlpMixer
        # MlpMixer has 2 inputs
        if in_seq_vel is not None and adv_seq_vel is not None:
            self.full_mpjpe = (self.full_mpjpe + losses.mpjpe(in_seq_vel, adv_seq_vel, reduce_axis=self.raxis)) / 2
            self.mpjpe_spatial = self.full_mpjpe.mean(0).cpu().data.numpy()
            self.mpjpe_temporal = self.full_mpjpe.mean(1).cpu().data.numpy()
            self.mpjpe = self.full_mpjpe.mean().cpu().data.numpy()
            self.mpjpe_sample = (self.mpjpe_sample + losses.mpjpe(in_seq_vel, adv_seq_vel,
                                                                  reduce_axis=[2, 1]).cpu().data.numpy()) / 2

            self.full_n_mpjpe = (self.full_n_mpjpe + losses.n_mpjpe(in_seq, adv_seq, reduce_axis=self.raxis)) / 2
            self.n_mpjpe_spatial = self.full_n_mpjpe.mean(0).cpu().data.numpy()
            self.n_mpjpe_temporal = self.full_n_mpjpe.mean(1).cpu().data.numpy()
            self.n_mpjpe = self.full_n_mpjpe.mean().cpu().data.numpy()
            self.n_mpjpe_sample = (self.n_mpjpe_sample + losses.n_mpjpe(in_seq, adv_seq,
                                                                        reduce_axis=[2, 1]).cpu().data.numpy()) / 2

            self.full_pa_mpjpe = (self.full_pa_mpjpe + losses.pa_mpjpe(in_seq, adv_seq, reduce_axis=self.raxis)) / 2
            self.pa_mpjpe_spatial = self.full_pa_mpjpe.mean(0).cpu().data.numpy()
            self.pa_mpjpe_temporal = self.full_pa_mpjpe.mean(1).cpu().data.numpy()
            self.pa_mpjpe = self.full_pa_mpjpe.mean().cpu().data.numpy()
            self.pa_mpjpe_sample = (self.pa_mpjpe_sample + losses.pa_mpjpe(in_seq, adv_seq,
                                                                           reduce_axis=[2, 1]).cpu().data.numpy()) / 2

            self.cosine_similarity_error = (self.cosine_similarity_error + self.cosine_similarity(
                in_seq_vel.reshape(in_seq_vel.shape[0], -1),
                adv_seq_vel.reshape(in_seq_vel.shape[0], -1)).cpu().data.numpy()) / 2
            self.cosine_similarity_temporal = (self.cosine_similarity_temporal + self.cosine_similarity_spec(in_seq_vel,
                                                                                                             adv_seq_vel).mean(
                (1, 2)).cpu().data.numpy()) / 2
            self.cosine_similarity_spatial = (self.cosine_similarity_spatial + self.cosine_similarity_spec(in_seq_vel,
                                                                                                           adv_seq_vel).mean(
                (0, 2)).cpu().data.numpy()) / 2

            self.hausdorff_mean_error = (self.hausdorff_mean_error + self.hausdorff_mean(in_seq, adv_seq,
                                                                                         0).cpu().data.numpy()) / 2
            self.hausdorff_mean_temporal = (self.hausdorff_mean_temporal + self.hausdorff_mean(in_seq, adv_seq,
                                                                                               1).cpu().data.numpy()) / 2
            self.hausdorff_mean_spatial = (self.hausdorff_mean_spatial + self.hausdorff_mean(in_seq, adv_seq,
                                                                                             2).cpu().data.numpy()) / 2

            self.hausdorff_max_error = (self.hausdorff_max_error + self.hausdorff_max(in_seq, adv_seq,
                                                                                      0).cpu().data.numpy()) / 2
            self.hausdorff_max_temporal = (self.hausdorff_max_temporal + self.hausdorff_max(in_seq, adv_seq,
                                                                                            1).cpu().data.numpy()) / 2
            self.hausdorff_max_spatial = (self.hausdorff_max_spatial + self.hausdorff_max(in_seq, adv_seq,
                                                                                          2).cpu().data.numpy()) / 2

            self.KLD_error = (self.KLD_error + self.KLD64(in_seq_vel, adv_seq_vel, 0).cpu().data.numpy()) / 2
            self.KLD_temporal = (self.KLD_temporal + self.KLD64(in_seq_vel, adv_seq_vel, 1).cpu().data.numpy()) / 2
            self.KLD_spatial = (self.KLD_spatial + self.KLD64(in_seq_vel, adv_seq_vel, 2).cpu().data.numpy()) / 2

            self.JSD_error = (self.JSD_error + self.JSD64(in_seq_vel, adv_seq_vel, 0).cpu().data.numpy()) / 2
            self.JSD_temporal = (self.JSD_temporal + self.JSD64(in_seq_vel, adv_seq_vel, 1).cpu().data.numpy()) / 2
            self.JSD_spatial = (self.JSD_spatial + self.JSD64(in_seq_vel, adv_seq_vel, 2).cpu().data.numpy()) / 2

            self.KSTest_error = (self.KSTest_error + self.KSTest(in_seq_vel, adv_seq_vel, 0).cpu().data.numpy()) / 2
            self.KSTest_temporal = (self.KSTest_temporal + self.KSTest(in_seq_vel, adv_seq_vel,
                                                                       1).cpu().data.numpy()) / 2
            self.KSTest_spatial = (self.KSTest_spatial + self.KSTest(in_seq_vel, adv_seq_vel, 2).cpu().data.numpy()) / 2

            self.mse_error = (self.mse_error + self.mse(in_seq_vel, adv_seq_vel).mean((1, 2, 3)).cpu().data.numpy()) / 2
            self.mse_temporal = (self.mse_temporal + self.mse(in_seq_vel, adv_seq_vel).mean(
                (0, 2, 3)).cpu().data.numpy()) / 2
            self.mse_spatial = (self.mse_spatial + self.mse(in_seq_vel, adv_seq_vel).mean(
                (0, 1, 3)).cpu().data.numpy()) / 2
        del self.full_mpjpe, self.full_n_mpjpe, self.full_pa_mpjpe,
        torch.cuda.empty_cache()
        return {"metric_type": self.typ_eval,
                "mpjpe": self.mpjpe,
                "queries": self.queries,
                "n_mpjpe": self.n_mpjpe,
                "pa_mpjpe": self.pa_mpjpe,

                "temporal_mpjpe": self.mpjpe_temporal,
                "temporal_n_mpjpe": self.n_mpjpe_temporal,
                "temporal_pa_mpjpe": self.pa_mpjpe_temporal,
                "temporal_hausdorff_mean": self.hausdorff_mean_temporal,
                "temporal_hausdorff_max": self.hausdorff_max_temporal,
                "temporal_mse": self.mse_temporal,
                "temporal_cos_simil": self.cosine_similarity_temporal,
                "temporal_KLD": self.KLD_temporal,
                "temporal_JSD": self.JSD_temporal,
                "temporal_KSTest": self.KSTest_temporal,

                "spatial_mpjpe": self.mpjpe_spatial,
                "spatial_n_mpjpe": self.n_mpjpe_spatial,
                "spatial_pa_mpjpe": self.pa_mpjpe_spatial,
                "spatial_hausdorff_mean": self.hausdorff_mean_spatial,
                "spatial_hausdorff_max": self.hausdorff_max_spatial,
                "spatial_mse": self.mse_spatial,
                "spatial_cos_simil": self.cosine_similarity_spatial,
                "spatial_KLD": self.KLD_spatial,
                "spatial_JSD": self.JSD_spatial,
                "spatial_KSTest": self.KSTest_spatial,

                "mpjpe_sample": self.mpjpe_sample,
                "n_mpjpe_sample": self.n_mpjpe_sample,
                "pa_mpjpe_sample": self.pa_mpjpe_sample,
                "hausdorff_mean_sample": self.hausdorff_mean_error,
                "hausdorff_max_sample": self.hausdorff_max_error,
                "mse_sample": self.mse_error,
                "cosine_simil_sample": self.cosine_similarity_error,
                "KLD_sample": self.KLD_error,
                "JSD_sample": self.JSD_error,
                "KSTest_sample": self.KSTest_error,
                }

    def _get_bound_per_sample(self, seq):
        assert self.typ_eval in ["max_val", "len_y", ]
        if self.typ_eval == "max":
            bound = seq.abs().max(1)[0].max(1)[0].max(1)[0]
        elif self.typ_eval == "len_y":
            bound = seq[:, :, :, 1].max(1)[0].max(1)[0] - seq[:, :, :, 1].min(1)[0].min(1)[0]
            bound = bound.abs()
        elif self.typ_eval == "std_y":
            bound = torch.std(seq, (1, 2, 3))
        elif self.typ_eval == "time_spatial_std":
            bound = torch.std(seq, 3)
        elif self.typ_eval == "time_spatial_std_dim":
            bound = torch.std(seq, (1, 2))
            bound = bound[:, ]

        if len(bound.shape) == 1:
            bound = bound[:, None, None, None]
        elif len(bound.shape) == 2:
            bound = bound[:, None, None, :]
        elif len(bound.shape) == 3:
            bound = bound[..., None]
        return bound

    def __repr__(self):
        return f'spatial_error:{self.spatial_error}\n' \
               f'temporal_error:{self.temporal_error}\n' \
               f'mpjpe:{self.mpjpe}\n'


# FGSM attack code
# Modified function to be compatible with the continuous domain.
class FGSM(ComputeAttackMetrics):
    def __init__(self, typ_eval="len_y", epsilon=0.01, joints=None, frames=None, db="h36m"):
        '''
          FGSM:
            epsilon: 0.0001
            joints: # unfortunately some names are the same in the joint nomenclature.
                - 0
            frames:
                - 0
                - 1
                - 5
                - 7
        '''
        super().__init__(typ_eval=typ_eval)
        self.epsilon = epsilon  # epsilon as percentage
        bones, joint_names = body_utils.get_reduced_skeleton(db, dim_used=None)

        if joints is None:
            self.joints = False
        else:
            self.joints = np.array(joints)
        if frames is None:
            self.frames = False
        else:
            self.frames = np.array(frames)

    def compute_gradient(self, seq):
        sign_data_grad = seq.grad.sign()  # data_grad
        epsilon = self.epsilon * self._get_bound_per_sample(seq)  # epsilon as percentage
        r_i = epsilon * sign_data_grad
        if self.joints is not False:
            mask = torch.zeros_like(r_i)
            mask[:, :, self.joints, :] = 1
        else:
            mask = torch.ones_like(r_i)
        if self.frames is not False:
            set_to_zero_mask = np.setdiff1d(np.arange(seq.shape[1]), self.frames)
            mask[:, set_to_zero_mask, :, :] = 0
        else:
            mask[:, np.arange(seq.shape[1]), :, :] = 1
        r_i = mask * r_i
        seq_adv = torch.autograd.Variable(seq.clone())
        return seq_adv + r_i

    def apply(self, seq, seq_vel=None, model=None, pred_func=None, params=None):
        model_name = model.__class__.__name__
        adversarial_inputs = {}
        seq.requires_grad = True
        if seq_vel is not None:
            seq_vel.requires_grad = True
        self._init_func(seq, seq_vel, model, pred_func, params)
        if model_name == "MlpMixer":
            seq_vel_adv = self.compute_gradient(seq_vel)
            seq_adv = seq
        else:
            seq_adv = self.compute_gradient(seq)
            seq_vel_adv = None
        adversarial_inputs['adv_inputs'] = seq_adv.cpu().data.numpy()
        if seq_vel_adv is not None:
            adversarial_inputs['adv_inputs_vel'] = seq_vel_adv.cpu().data.numpy()
        else:
            adversarial_inputs['adv_inputs_vel'] = seq_vel_adv
        return adversarial_inputs


# Iterative-FGSM attack code
# Modified function to be compatible with the continuous domain.
class IFGSM(ComputeAttackMetrics):
    def __init__(self, typ_eval="len_y", iterations=1, epsilon=0.01, joints=None, frames=None, db="h36m"):
        '''
          FGSM:
            epsilon: 0.0001
            joints: # unfortunately some names are the same in the joint nomenclature.
                - 0
            frames:
                - 0
                - 1
                - 5
                - 7
        '''
        super().__init__(typ_eval=typ_eval)
        self.epsilon = epsilon
        self.iterations = iterations
        bones, joint_names = body_utils.get_reduced_skeleton(db, dim_used=None)

        if joints is None:
            self.joints = False
        else:
            self.joints = np.array(joints)
        if frames is None:
            self.frames = False
        else:
            self.frames = np.array(frames)

    def compute_gradient(self, seq, seq_i, seq_adv, op_mask):
        sign_data_grad = seq_i.grad.sign()  # data_grad
        epsilon = self.epsilon * self._get_bound_per_sample(seq_i)
        r_i = epsilon * sign_data_grad / self.iterations
        if self.joints is not False:
            mask = torch.zeros_like(r_i)
            mask[:, :, self.joints, :] = 1
        else:
            mask = torch.ones_like(r_i)
        if self.frames is not False:
            set_to_zero_mask = np.setdiff1d(np.arange(seq_i.shape[1]), self.frames)
            mask[:, set_to_zero_mask, :, :] = 0
        else:
            mask[:, np.arange(seq_i.shape[1]), :, :] = 1
        r_i = mask * r_i
        seq_adv[op_mask] = seq_i[op_mask] + r_i[op_mask]

        dist = torch.norm(seq_adv - seq, float('inf'), dim=(1, 2, 3))
        mask_dist = dist > epsilon.flatten()
        temp_seq = seq_adv[mask_dist]
        seq_o = seq[mask_dist]
        mask_to_o = (temp_seq < (seq[mask_dist] - epsilon[mask_dist])) + \
                    (temp_seq >= (seq[mask_dist] + epsilon[mask_dist]))
        temp_seq[mask_to_o] = seq_o[mask_to_o]
        seq_adv[mask_dist] = temp_seq
        seq_i = torch.autograd.Variable(seq_adv.clone())
        seq_i.requires_grad = True
        return seq_i, seq_adv

    # Also known as I-FGSM Attack
    # def apply(self, model, loss, images, labels, scale, eps, alpha, iters=0):
    def apply(self, seq, seq_vel=None, model=None, pred_func=None, params=None):
        model_name = model.__class__.__name__
        adversarial_inputs = {}
        original_params = params.copy()
        seq_i = torch.autograd.Variable(seq.clone())
        seq_adv = torch.autograd.Variable(seq.clone())
        seq.requires_grad = True
        seq_i.requires_grad = True
        seq_vel_i = torch.autograd.Variable(seq_vel.clone())
        seq_vel_adv = torch.autograd.Variable(seq_vel.clone())
        seq_vel.requires_grad = True
        seq_vel_i.requires_grad = True
        self.queries = np.zeros(seq.shape[0])
        op_mask = np.arange(seq.shape[0])  # 256 256 256 224 145 83 5 1 0
        active_opt = np.zeros(seq.shape[0])
        loss_highest = np.zeros(seq.shape[0])
        for i in range(self.iterations):
            self.queries[op_mask] += 1  # batch iteration
            params["target"] = original_params["target"][op_mask]
            # params["data"] = original_params["data"][op_mask] # FOR OTHER MODELS
            k_i, loss_i = self._init_func(seq_i[op_mask], seq_vel_i[op_mask], model, pred_func, params,
                                          reduce_axis=[1, 2])
            if model_name == "MlpMixer":
                seq_vel_i, seq_vel_adv = self.compute_gradient(seq_vel, seq_vel_i, seq_vel_adv, op_mask)
                seq_adv = seq
            else:
                seq_i, seq_adv = self.compute_gradient(seq, seq_i, seq_adv, op_mask)
                seq_vel_adv = None

            loss_i = loss_i.cpu().detach().numpy()
            mask = loss_i > loss_highest[op_mask]
            loss_t = loss_highest[op_mask]  # this is done because of double mask used.
            loss_t[mask] = loss_i[mask]
            loss_highest[op_mask] = loss_t

            active_t = active_opt[op_mask]  # this is done because of double mask used.
            active_t[~mask] += 1
            active_opt[op_mask] = active_t
            op_mask = op_mask[active_opt[op_mask] < 5]
            # Times the algorithm tolerate a wrong optimization.
            if active_opt.mean() == 5:
                break;
            ##################################
        adversarial_inputs['adv_inputs'] = seq_adv.cpu().data.numpy()
        if seq_vel_adv is not None:
            adversarial_inputs['adv_inputs_vel'] = seq_vel_adv.cpu().data.numpy()
        else:
            adversarial_inputs['adv_inputs_vel'] = seq_vel_adv
        return adversarial_inputs


# Momentum Iterative-FGSM attack code
# Modified function to be compatible with the continuous domain.
class MIFGSM(ComputeAttackMetrics):
    def __init__(self, typ_eval="len_y", iterations=1, epsilon=0.01, mu=0.01, joints=None, frames=None, db="h36m"):
        '''
          FGSM:
            epsilon: 0.0001
            joints: # unfortunately some names are the same in the joint nomenclature.
                - 0
            frames:
                - 0
                - 1
                - 5
                - 7
        '''
        super().__init__(typ_eval=typ_eval)
        self.epsilon = epsilon
        self.mu = mu
        self.iterations = iterations
        bones, joint_names = body_utils.get_reduced_skeleton(db, dim_used=None)

        if joints is None:
            self.joints = False
        else:
            self.joints = np.array(joints)
        if frames is None:
            self.frames = False
        else:
            self.frames = np.array(frames)

    def compute_gradient(self, seq, seq_i, seq_adv, g_t, op_mask):
        g_t = self.mu * g_t + seq_i.grad / torch.norm(seq_i.grad, p=1, dim=[1, 2, 3], keepdim=True)
        epsilon = self.epsilon * self._get_bound_per_sample(seq_i)
        alpha = epsilon / self.iterations
        r_i = alpha * g_t.sign()

        if self.joints is not False:
            mask = torch.zeros_like(r_i)
            mask[:, :, self.joints, :] = 1
        else:
            mask = torch.ones_like(r_i)
        if self.frames is not False:
            set_to_zero_mask = np.setdiff1d(np.arange(seq_i.shape[1]), self.frames)
            mask[:, set_to_zero_mask, :, :] = 0
        else:
            mask[:, np.arange(seq_i.shape[1]), :, :] = 1
        r_i = mask * r_i
        seq_adv[op_mask] = seq_i[op_mask] + r_i[op_mask]

        dist = torch.norm(seq_adv - seq, float('inf'), dim=(1, 2, 3))
        mask_dist = dist > epsilon.flatten()
        temp_seq = seq_adv[mask_dist]
        seq_o = seq[mask_dist]
        mask_to_o = (temp_seq < (seq[mask_dist] - epsilon[mask_dist])) + \
                    (temp_seq >= (seq[mask_dist] + epsilon[mask_dist]))
        temp_seq[mask_to_o] = seq_o[mask_to_o]
        seq_adv[mask_dist] = temp_seq
        seq_i = torch.autograd.Variable(seq_adv.clone())
        seq_i.requires_grad = True
        return seq_i, seq_adv, g_t

    # Also known as I-FGSM Attack
    # def apply(self, model, loss, images, labels, scale, eps, alpha, iters=0):
    def apply(self, seq, seq_vel=None, model=None, pred_func=None, params=None):
        model_name = model.__class__.__name__
        adversarial_inputs = {}
        original_params = params.copy()
        seq_i = torch.autograd.Variable(seq.clone())
        seq_adv = torch.autograd.Variable(seq.clone())
        seq.requires_grad = True
        seq_i.requires_grad = True
        alpha = self.epsilon / self.iterations * 100
        g_t = 0
        seq_vel_i = torch.autograd.Variable(seq_vel.clone())
        seq_vel_adv = torch.autograd.Variable(seq_vel.clone())
        seq_vel.requires_grad = True
        seq_vel_i.requires_grad = True
        g_t_vel = 0
        self.queries = np.zeros(seq.shape[0])
        op_mask = np.arange(seq.shape[0])
        active_opt = np.zeros(seq.shape[0])
        loss_highest = np.zeros(seq.shape[0])
        for i in range(self.iterations):
            self.queries[op_mask] += 1  # batch iteration
            params["target"] = original_params["target"][op_mask]
            # params["data"] = original_params["data"][op_mask] # FOR OTHER MODELS
            k_i, loss_i = self._init_func(seq_i[op_mask], seq_vel_i[op_mask], model, pred_func, params,
                                          reduce_axis=[1, 2])
            if model_name == "MlpMixer":
                seq_vel_i, seq_vel_adv, g_t_vel = self.compute_gradient(seq_vel, seq_vel_i, seq_vel_adv, g_t_vel,
                                                                        op_mask)
                seq_adv = seq
            else:
                seq_i, seq_adv, g_t = self.compute_gradient(seq, seq_i, seq_adv, g_t, op_mask)
                seq_vel_adv = None

            loss_i = loss_i.cpu().detach().numpy()
            mask = loss_i > loss_highest[op_mask]
            loss_t = loss_highest[op_mask]  # this is done because of double mask used.
            loss_t[mask] = loss_i[mask]
            loss_highest[op_mask] = loss_t

            active_t = active_opt[op_mask]  # this is done because of double mask used.
            active_t[~mask] += 1
            active_opt[op_mask] = active_t
            op_mask = op_mask[active_opt[op_mask] < 5]
            # Times the algorithm tolerate a wrong optimization.
            if active_opt.mean() == 5:
                break;
        adversarial_inputs['adv_inputs'] = seq_adv.cpu().data.numpy()
        if seq_vel_adv is not None:
            adversarial_inputs['adv_inputs_vel'] = seq_vel_adv.cpu().data.numpy()
        else:
            adversarial_inputs['adv_inputs_vel'] = seq_vel_adv
        return adversarial_inputs


# DEEPFOOL code adapted for multiout regression - basically takes the average in the line 5 from Algorithm 1.
# Modified function to be compatible with the continuous domain.
class DEEPFOOL(ComputeAttackMetrics):
    def __init__(self, typ_eval="len_y", iterations=10, overshoot=0.02, joints=None, frames=None, db="h36m"):
        '''
          FGSM:
            epsilon: 0.0001
            joints: # unfortunately some names are the same in the joint nomenclature.
                - 0
            frames:
                - 0
                - 1
                - 5
                - 7
        '''
        super().__init__(typ_eval=typ_eval)
        self.overshoot = overshoot
        self.iterations = iterations
        bones, joint_names = body_utils.get_reduced_skeleton(db, dim_used=None)

        if joints is None:
            self.joints = False
        else:
            self.joints = np.array(joints)
        if frames is None:
            self.frames = False
        else:
            self.frames = np.array(frames)

    def compute_gradient(self, seq, seq_i, seq_adv, g_t, op_mask, pred):
        norm_grad = torch.norm(seq_i.grad, p=1, dim=[1, 2, 3], keepdim=True) + 1e-10
        r_i = - (seq_i.grad[:, None, :] * pred[:, :, None, :]).mean(1) / norm_grad

        if self.joints is not False:
            mask = torch.zeros_like(r_i)
            mask[:, :, self.joints, :] = 1
        else:
            mask = torch.ones_like(r_i)
        if self.frames is not False:
            set_to_zero_mask = np.setdiff1d(np.arange(seq_i.shape[1]), self.frames)
            mask[:, set_to_zero_mask, :, :] = 0
        else:
            mask[:, np.arange(seq_i.shape[1]), :, :] = 1
        r_i = mask * r_i
        seq_adv[op_mask] = seq_i[op_mask] + r_i[op_mask]

        seq_i = torch.autograd.Variable(seq_adv.clone())
        seq_i.requires_grad = True
        del r_i, mask, op_mask, seq
        return seq_i, seq_adv, g_t

    # Also known as I-FGSM Attack
    # def apply(self, model, loss, images, labels, scale, eps, alpha, iters=0):
    def apply(self, seq, seq_vel=None, model=None, pred_func=None, params=None):
        model_name = model.__class__.__name__
        adversarial_inputs = {}
        original_params = params.copy()
        seq_i = torch.autograd.Variable(seq.clone())
        seq_adv = torch.autograd.Variable(seq.clone())
        seq.requires_grad = True
        seq_i.requires_grad = True
        g_t = 0
        seq_vel_i = torch.autograd.Variable(seq_vel.clone())
        seq_vel_adv = torch.autograd.Variable(seq_vel.clone())
        seq_vel.requires_grad = True
        seq_vel_i.requires_grad = True
        g_t_vel = 0
        self.queries = np.zeros(seq.shape[0])
        op_mask = np.arange(seq.shape[0])
        active_opt = np.zeros(seq.shape[0])
        loss_highest = np.zeros(seq.shape[0])
        k_i, loss_i = self._init_func(seq_i[op_mask], seq_vel_i[op_mask], model, pred_func, params,
                                      reduce_axis=[1, 2])
        for i in range(self.iterations):
            self.queries[op_mask] += 1  # batch iteration
            params["target"] = original_params["target"][op_mask]
            # params["data"] = original_params["data"][op_mask] # FOR OTHER MODELS
            k_i[op_mask], loss_i = self._init_func(seq_i[op_mask], seq_vel_i[op_mask], model,
                                                   pred_func, params, reduce_axis=[1, 2])
            if model_name == "MlpMixer":
                seq_vel_i, seq_vel_adv, g_t_vel = self.compute_gradient(seq_vel, seq_vel_i, seq_vel_adv, g_t_vel,
                                                                        op_mask, k_i)
                seq_adv = seq
            else:
                seq_i, seq_adv, g_t = self.compute_gradient(seq, seq_i, seq_adv, g_t, op_mask, k_i)
                seq_vel_adv = None

            loss_i = loss_i.cpu().detach().numpy()
            mask = loss_i > loss_highest[op_mask]
            loss_t = loss_highest[op_mask]  # this is done because of double mask used.
            loss_t[mask] = loss_i[mask]
            loss_highest[op_mask] = loss_t

            active_t = active_opt[op_mask]  # this is done because of double mask used.
            active_t[~mask] += 1
            active_opt[op_mask] = active_t
            op_mask = op_mask[active_opt[op_mask] < 5]
            # Times the algorithm tolerate a wrong optimization.
            if active_opt.mean() == 5:
                break;
        
        adversarial_inputs['adv_inputs'] = seq_adv.cpu().data.numpy()
        if seq_vel_adv is not None:
            adversarial_inputs['adv_inputs_vel'] = seq_vel_adv.cpu().data.numpy()
        else:
            adversarial_inputs['adv_inputs_vel'] = seq_vel_adv
        del seq_i, seq_adv, seq_vel_i, seq_vel_adv
        torch.cuda.empty_cache()
        return adversarial_inputs


# GetActualGradient mode
# Collect gradient without any modification.
class NOATTACK(ComputeAttackMetrics):
    """
    Collect gradient without any modification.
    """

    def __init__(self, typ_eval="len_y", db="h36m", **kwargs):
        self.db = db
        super().__init__(typ_eval=typ_eval)

    def apply(self, seq, seq_vel=None, model=None, pred_func=None, params=None):
        adversarial_inputs = {}
        seq_i = torch.autograd.Variable(seq.clone())
        seq_i.requires_grad = True
        seq_vel_i = torch.autograd.Variable(seq_vel.clone())
        seq_vel_i.requires_grad = True
        if seq_vel is not None:
            seq_vel.requires_grad = True
        k_i, loss_i = self._init_func(seq_i, seq_vel_i, model, pred_func, params,
                                      reduce_axis=[1, 2])
        adversarial_inputs['adv_inputs'] = seq_i.cpu().data.numpy()
        if seq_vel_i is not None:
            adversarial_inputs['adv_inputs_vel'] = seq_vel_i.cpu().data.numpy()
        else:
            adversarial_inputs['adv_inputs_vel'] = seq_vel_i
        return adversarial_inputs
