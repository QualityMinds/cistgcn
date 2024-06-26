import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..layers import SE

torch.manual_seed(0)


# This is the simple CNN layer,that performs a 2-D convolution while maintaining the dimensions of the input(except for the features dimension)
class CNN_layer(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 dropout,
                 bias=True):
        super(CNN_layer, self).__init__()
        self.kernel_size = kernel_size
        padding = (
            (kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)  # padding so that both dimensions are maintained
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1

        self.block1 = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=(1, 1)),
                       nn.BatchNorm2d(out_ch),
                       nn.Dropout(dropout, inplace=True),
                       ]

        self.block1 = nn.Sequential(*self.block1)

    def forward(self, x):
        output = self.block1(x)
        return output


class FPN(nn.Module):
    def __init__(self, in_ch,
                 out_ch,
                 kernel,  # (3,1)
                 dropout,
                 reduction,
                 ):
        super(FPN, self).__init__()
        kernel_size = kernel if isinstance(kernel, (tuple, list)) else (kernel, kernel)
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        pad1 = (padding[0], padding[1])
        pad2 = (padding[0] + pad1[0], padding[1] + pad1[1])
        pad3 = (padding[0] + pad2[0], padding[1] + pad2[1])
        dil1 = (1, 1)
        dil2 = (1 + pad1[0], 1 + pad1[1])
        dil3 = (1 + pad2[0], 1 + pad2[1])
        self.block1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad1, dilation=dil1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.Dropout(dropout, inplace=True),
                                    nn.PReLU(),
                                    )
        self.block2 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad2, dilation=dil2),
                                    nn.BatchNorm2d(out_ch),
                                    nn.Dropout(dropout, inplace=True),
                                    nn.PReLU(),
                                    )
        self.block3 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad3, dilation=dil3),
                                    nn.BatchNorm2d(out_ch),
                                    nn.Dropout(dropout, inplace=True),
                                    nn.PReLU(),
                                    )
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))  # Action Context.
        self.compress = nn.Conv2d(out_ch * 3 + in_ch,
                                  out_ch,
                                  kernel_size=(1, 1))  # PRELU is outside the loop, check at the end of the code.

    def forward(self, x):
        b, dim, joints, seq = x.shape
        global_action = F.interpolate(self.pooling(x), (joints, seq))
        out = torch.cat((self.block1(x), self.block2(x), self.block3(x), global_action), dim=1)
        out = self.compress(out)
        return out


def mish(x):
    return (x * torch.tanh(F.softplus(x)))


class ConvTemporalGraphical(nn.Module):
    # Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
    Shape:
        - Input: Input graph sequence in :math:`(N, in_ch, T_{in}, V)` format
        - Output: Outpu graph sequence in :math:`(N, out_ch, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self, time_dim, joints_dim, domain, interpratable):
        super(ConvTemporalGraphical, self).__init__()

        if domain == "time":
            # learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
            size = joints_dim
            if not interpratable:
                self.A = nn.Parameter(torch.FloatTensor(time_dim, size, size))
                self.domain = 'nctv,tvw->nctw'
            else:
                self.domain = 'nctv,ntvw->nctw'
        elif domain == "space":
            size = time_dim
            if not interpratable:
                self.A = nn.Parameter(torch.FloatTensor(joints_dim, size, size))
                self.domain = 'nctv,vtq->ncqv'
            else:
                self.domain = 'nctv,nvtq->ncqv'
        if not interpratable:
            stdv = 1. / math.sqrt(self.A.size(1))
            self.A.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = torch.einsum(self.domain, (x, self.A))
        return x.contiguous()


class Map2Adj(nn.Module):
    def __init__(self,
                 in_ch,
                 time_dim,
                 joints_dim,
                 domain,
                 dropout,
                 ):
        super(Map2Adj, self).__init__()
        self.domain = domain
        inter_ch = in_ch // 2
        self.time_compress = nn.Sequential(nn.Conv2d(in_ch, inter_ch, kernel_size=1, bias=False),
                                           nn.BatchNorm2d(inter_ch),
                                           nn.PReLU(),
                                           nn.Conv2d(inter_ch, inter_ch, kernel_size=(time_dim, 1), bias=False),
                                           nn.BatchNorm2d(inter_ch),
                                           nn.Dropout(dropout, inplace=True),
                                           nn.Conv2d(inter_ch, time_dim, kernel_size=1, bias=False),
                                           )
        self.joint_compress = nn.Sequential(nn.Conv2d(in_ch, inter_ch, kernel_size=1, bias=False),
                                            nn.BatchNorm2d(inter_ch),
                                            nn.PReLU(),
                                            nn.Conv2d(inter_ch, inter_ch, kernel_size=(1, joints_dim), bias=False),
                                            nn.BatchNorm2d(inter_ch),
                                            nn.Dropout(dropout, inplace=True),
                                            nn.Conv2d(inter_ch, joints_dim, kernel_size=1, bias=False),
                                            )

        if self.domain == "space":
            ch = joints_dim
            self.perm1 = (0, 1, 2, 3)
            self.perm2 = (0, 3, 2, 1)
        if self.domain == "time":
            ch = time_dim
            self.perm1 = (0, 2, 1, 3)
            self.perm2 = (0, 1, 2, 3)

        inter_ch = ch  # // 2
        self.expansor = nn.Sequential(nn.Conv2d(ch, inter_ch, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(inter_ch),
                                      nn.Dropout(dropout, inplace=True),
                                      nn.PReLU(),
                                      nn.Conv2d(inter_ch, ch, kernel_size=1, bias=False),
                                      )
        self.time_compress.apply(self._init_weights)
        self.joint_compress.apply(self._init_weights)
        self.expansor.apply(self._init_weights)

    def _init_weights(self, m, gain=0.05):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if isinstance(m, (nn.Conv2d, nn.Conv1d)):
            torch.nn.init.xavier_normal_(m.weight, gain=gain)
        if isinstance(m, nn.PReLU):
            torch.nn.init.constant_(m.weight, 0.25)

    def forward(self, x):
        b, dims, seq, joints = x.shape
        dim_seq = self.time_compress(x)
        dim_space = self.joint_compress(x)
        o = torch.matmul(dim_space.permute(self.perm1), dim_seq.permute(self.perm2))
        Adj = self.expansor(o)
        return Adj


class Domain_GCNN_layer(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_ch, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_ch, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
            :in_ch= dimension of coordinates
            : out_ch=dimension of coordinates
            +
    """

    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 stride,
                 time_dim,
                 joints_dim,
                 domain,
                 interpratable,
                 dropout,
                 bias=True):

        super(Domain_GCNN_layer, self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)
        self.interpratable = interpratable
        self.domain = domain

        self.gcn = ConvTemporalGraphical(time_dim, joints_dim, domain, interpratable)
        self.tcn = nn.Sequential(nn.Conv2d(in_ch,
                                           out_ch,
                                           (self.kernel_size[0], self.kernel_size[1]),
                                           (stride, stride),
                                           padding,
                                           ),
                                 nn.BatchNorm2d(out_ch),
                                 nn.Dropout(dropout, inplace=True),
                                 )

        if stride != 1 or in_ch != out_ch:
            self.residual = nn.Sequential(nn.Conv2d(in_ch,
                                                    out_ch,
                                                    kernel_size=1,
                                                    stride=(1, 1)),
                                          nn.BatchNorm2d(out_ch),
                                          )
        else:
            self.residual = nn.Identity()
        if self.interpratable:
            self.map_to_adj = Map2Adj(in_ch,
                                      time_dim,
                                      joints_dim,
                                      domain,
                                      dropout,
                                      )
        else:
            self.map_to_adj = nn.Identity()
        self.prelu = nn.PReLU()

    def forward(self, x):
        #   assert A.shape[0] == self.kernel_size[1], print(A.shape[0],self.kernel_size)
        res = self.residual(x)
        self.Adj = self.map_to_adj(x)
        if self.interpratable:
            self.gcn.A = self.Adj
        x1 = self.gcn(x)
        x2 = self.tcn(x1)
        x3 = x2 + res
        x4 = self.prelu(x3)
        return x4


# Dynamic SpatioTemporal Decompose Graph Convolutions (DSTD-GC)
class DSTD_GC(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_ch, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_ch, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
            : in_ch= dimension of coordinates
            : out_ch=dimension of coordinates
            +
    """

    def __init__(self,
                 in_ch,
                 out_ch,
                 interpratable,
                 kernel_size,
                 stride,
                 time_dim,
                 joints_dim,
                 reduction,
                 dropout):
        super(DSTD_GC, self).__init__()
        self.dsgn = Domain_GCNN_layer(in_ch, out_ch, kernel_size, stride,
                                      time_dim, joints_dim, "space", interpratable, dropout)
        self.tsgn = Domain_GCNN_layer(in_ch, out_ch, kernel_size, stride,
                                      time_dim, joints_dim, "time", interpratable, dropout)

        self.compressor = nn.Sequential(nn.Conv2d(out_ch * 2, out_ch, 1, bias=False),
                                        nn.BatchNorm2d(out_ch),
                                        nn.PReLU(),
                                        SE.SELayer2d(out_ch, reduction=reduction),
                                        )
        if stride != 1 or in_ch != out_ch:
            self.residual = nn.Sequential(nn.Conv2d(in_ch,
                                                    out_ch,
                                                    kernel_size=1,
                                                    stride=(1, 1)),
                                          nn.BatchNorm2d(out_ch),
                                          )
        else:
            self.residual = nn.Identity()

        # Weighting features
        out_ch_c = out_ch // 2 if out_ch // 2 > 1 else 1
        self.global_norm = nn.BatchNorm2d(in_ch)
        self.conv_s = nn.Sequential(nn.Conv2d(in_ch, out_ch_c, (time_dim, 1), bias=False),
                                    nn.BatchNorm2d(out_ch_c),
                                    nn.Dropout(dropout, inplace=True),
                                    nn.PReLU(),
                                    nn.Conv2d(out_ch_c, out_ch, (1, joints_dim), bias=False),
                                    nn.BatchNorm2d(out_ch),
                                    nn.Dropout(dropout, inplace=True),
                                    nn.PReLU(),
                                    )
        self.conv_t = nn.Sequential(nn.Conv2d(in_ch, out_ch_c, (time_dim, 1), bias=False),
                                    nn.BatchNorm2d(out_ch_c),
                                    nn.Dropout(dropout, inplace=True),
                                    nn.PReLU(),
                                    nn.Conv2d(out_ch_c, out_ch, (1, joints_dim), bias=False),
                                    nn.BatchNorm2d(out_ch),
                                    nn.Dropout(dropout, inplace=True),
                                    nn.PReLU(),
                                    )
        self.map_s = nn.Sequential(nn.Linear(out_ch + 2 + time_dim * 2, out_ch, bias=False),
                                   nn.BatchNorm1d(out_ch),
                                   nn.Dropout(dropout, inplace=True),
                                   nn.PReLU(),
                                   nn.Linear(out_ch, out_ch, bias=False),
                                   )
        self.map_t = nn.Sequential(nn.Linear(out_ch + 2 + time_dim * 2, out_ch, bias=False),
                                   nn.BatchNorm1d(out_ch),
                                   nn.Dropout(dropout, inplace=True),
                                   nn.PReLU(),
                                   nn.Linear(out_ch, out_ch, bias=False),
                                   )
        self.prelu1 = nn.Sequential(nn.BatchNorm2d(out_ch),
                                    nn.PReLU(),
                                    )
        self.prelu2 = nn.Sequential(nn.BatchNorm2d(out_ch),
                                    nn.PReLU(),
                                    )

    def _get_stats_(self, x):
        global_avg_pool = x.mean((3, 2)).mean(1, keepdims=True)
        global_avg_pool_features = x.mean(3).mean(1)
        global_std_pool = x.std((3, 2)).std(1, keepdims=True)
        global_std_pool_features = x.std(3).std(1)
        return torch.cat((
            global_avg_pool,
            global_avg_pool_features,
            global_std_pool,
            global_std_pool_features,
        ),
            dim=1)

    def forward(self, x):
        b, dim, seq, joints = x.shape  # 64, 3, 10, 22
        xn = self.global_norm(x)

        stats = self._get_stats_(xn)
        w1 = torch.cat((self.conv_s(xn).view(b, -1), stats), dim=1)
        stats = self._get_stats_(xn)
        w2 = torch.cat((self.conv_t(xn).view(b, -1), stats), dim=1)
        self.w1 = self.map_s(w1)
        self.w2 = self.map_t(w2)
        w1 = self.w1[..., None, None]
        w2 = self.w2[..., None, None]

        x1 = self.dsgn(xn)
        x2 = self.tsgn(xn)
        out = torch.cat((self.prelu1(w1 * x1), self.prelu2(w2 * x2)), dim=1)
        out = self.compressor(out)
        return out + self.residual(xn)


class ContextLayer(nn.Module):
    def __init__(self,
                 in_ch,
                 hidden_ch,
                 output_seq,
                 input_seq,
                 joints,
                 dims=3,
                 reduction=8,
                 dropout=0.1,
                 ):
        super(ContextLayer, self).__init__()
        self.n_output = output_seq
        self.n_joints = joints
        self.n_input = input_seq
        self.context_conv1 = nn.Sequential(nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
                                           nn.BatchNorm2d(hidden_ch),
                                           nn.PReLU(),
                                           )

        self.context_conv2 = nn.Sequential(nn.Conv2d(in_ch, hidden_ch, (input_seq, 1), bias=False),
                                           nn.BatchNorm2d(hidden_ch),
                                           nn.PReLU(),
                                           )
        self.context_conv3 = nn.Sequential(nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
                                           nn.BatchNorm2d(hidden_ch),
                                           nn.PReLU(),
                                           )
        self.map1 = nn.Sequential(nn.Linear(hidden_ch, self.n_output, bias=False),
                                  nn.Dropout(dropout, inplace=True),
                                  nn.PReLU(),
                                  )
        self.map2 = nn.Sequential(nn.Linear(hidden_ch, self.n_output, bias=False),
                                  nn.Dropout(dropout, inplace=True),
                                  nn.PReLU(),
                                  )
        self.map3 = nn.Sequential(nn.Linear(hidden_ch, self.n_output, bias=False),
                                  nn.Dropout(dropout, inplace=True),
                                  nn.PReLU(),
                                  )

        self.fmap_s = nn.Sequential(nn.Linear(self.n_output * 3, self.n_joints, bias=False),
                                    nn.BatchNorm1d(self.n_joints),
                                    nn.Dropout(dropout, inplace=True), )

        self.fmap_t = nn.Sequential(nn.Linear(self.n_output * 3, self.n_output, bias=False),
                                    nn.BatchNorm1d(self.n_output),
                                    nn.Dropout(dropout, inplace=True), )

        # inter_ch = self.n_joints  # // 2
        self.norm_map = nn.Sequential(nn.Conv1d(self.n_output, self.n_output, 1, bias=False),
                                      nn.BatchNorm1d(self.n_output),
                                      nn.Dropout(dropout, inplace=True),
                                      nn.PReLU(),
                                      SE.SELayer1d(self.n_output, reduction=reduction),
                                      nn.Conv1d(self.n_output, self.n_output, 1, bias=False),
                                      nn.BatchNorm1d(self.n_output),
                                      nn.Dropout(dropout, inplace=True),
                                      nn.PReLU(),
                                      )

        self.fconv = nn.Sequential(nn.Conv2d(1, dims, 1, bias=False),
                                   nn.BatchNorm2d(dims),
                                   nn.PReLU(),
                                   nn.Conv2d(dims, dims, 1, bias=False),
                                   nn.BatchNorm2d(dims),
                                   nn.PReLU(),
                                   )
        self.SE = SE.SELayer2d(self.n_output, reduction=reduction)

    def forward(self, x):
        b, _, seq, joint_dim = x.shape
        y1 = self.context_conv1(x).max(-1)[0].max(-1)[0]
        y2 = self.context_conv2(x).view(b, -1, joint_dim).max(-1)[0]
        ym = self.context_conv3(x).mean((2, 3))
        y = torch.cat((self.map1(y1), self.map2(y2), self.map3(ym)), dim=1)
        self.joints = self.fmap_s(y)
        self.displacements = self.fmap_t(y)  # .cumsum(1)
        self.seq_joints = torch.bmm(self.displacements.unsqueeze(2), self.joints.unsqueeze(1))
        self.seq_joints_n = self.norm_map(self.seq_joints)
        self.seq_joints_dims = self.fconv(self.seq_joints_n.view(b, 1, self.n_output, self.n_joints))
        o = self.SE(self.seq_joints_dims.permute(0, 2, 3, 1))
        return o


class CISTGCN(nn.Module):
    """
    Shape:
        - Input[0]: Input sequence in :math:`(N, in_ch,T_in, V)` format
        - Output[0]: Output sequence in :math:`(N,T_out,in_ch, V)` format
        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
            :in_ch=number of channels for the coordiantes(default=3)
            +
    """

    def __init__(self, arch, learn):
        super(CISTGCN, self).__init__()
        self.clipping = arch.model_params.clipping

        self.n_input = arch.model_params.input_n
        self.n_output = arch.model_params.output_n
        self.n_joints = arch.model_params.joints
        self.n_txcnn_layers = arch.model_params.n_txcnn_layers
        self.txc_kernel_size = [arch.model_params.txc_kernel_size] * 2
        self.input_gcn = arch.model_params.input_gcn
        self.output_gcn = arch.model_params.output_gcn
        self.reduction = arch.model_params.reduction
        self.hidden_dim = arch.model_params.hidden_dim

        self.st_gcnns = nn.ModuleList()
        self.txcnns = nn.ModuleList()
        self.se = nn.ModuleList()

        self.in_conv = nn.ModuleList()
        self.context_layer = nn.ModuleList()
        self.trans = nn.ModuleList()
        self.in_ch = 10
        self.model_tx = self.input_gcn.model_complexity.copy()
        self.model_tx.insert(0, 1)  # add 1 in the position 0.

        self.input_gcn.model_complexity.insert(0, self.in_ch)
        self.input_gcn.model_complexity.append(self.in_ch)
        # self.input_gcn.interpretable.insert(0, True)
        # self.input_gcn.interpretable.append(False)
        for i in range(len(self.input_gcn.model_complexity) - 1):
            self.st_gcnns.append(DSTD_GC(self.input_gcn.model_complexity[i],
                                         self.input_gcn.model_complexity[i + 1],
                                         self.input_gcn.interpretable[i],
                                         [1, 1], 1, self.n_input, self.n_joints, self.reduction, learn.dropout))

        self.context_layer = ContextLayer(1, self.hidden_dim,
                                          self.n_output, self.n_output, self.n_joints,
                                          3, self.reduction, learn.dropout
                                          )

        # at this point, we must permute the dimensions of the gcn network, from (N,C,T,V) into (N,T,C,V)
        # with kernel_size[3,3] the dimensions of C,V will be maintained
        self.txcnns.append(FPN(self.n_input, self.n_output, self.txc_kernel_size, 0., self.reduction))
        for i in range(1, self.n_txcnn_layers):
            self.txcnns.append(FPN(self.n_output, self.n_output, self.txc_kernel_size, 0., self.reduction))

        self.prelus = nn.ModuleList()
        for j in range(self.n_txcnn_layers):
            self.prelus.append(nn.PReLU())

        self.dim_conversor = nn.Sequential(nn.Conv2d(self.in_ch, 3, 1, bias=False),
                                           nn.BatchNorm2d(3),
                                           nn.PReLU(),
                                           nn.Conv2d(3, 3, 1, bias=False),
                                           nn.PReLU(3), )

        self.st_gcnns_o = nn.ModuleList()
        self.output_gcn.model_complexity.insert(0, 3)
        for i in range(len(self.output_gcn.model_complexity) - 1):
            self.st_gcnns_o.append(DSTD_GC(self.output_gcn.model_complexity[i],
                                           self.output_gcn.model_complexity[i + 1],
                                           self.output_gcn.interpretable[i],
                                           [1, 1], 1, self.n_joints, self.n_output, self.reduction, learn.dropout))

        self.st_gcnns_o.apply(self._init_weights)
        self.st_gcnns.apply(self._init_weights)
        self.txcnns.apply(self._init_weights)

    def _init_weights(self, m, gain=0.1):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        # if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        #     torch.nn.init.xavier_normal_(m.weight, gain=gain)
        if isinstance(m, nn.PReLU):
            torch.nn.init.constant_(m.weight, 0.25)

    def forward(self, x):
        b, seq, joints, dim = x.shape
        vel = torch.zeros_like(x)
        vel[:, :-1] = torch.diff(x, dim=1)
        vel[:, -1] = x[:, -1]
        acc = torch.zeros_like(x)
        acc[:, :-1] = torch.diff(vel, dim=1)
        acc[:, -1] = vel[:, -1]
        x1 = torch.cat((x, acc, vel, torch.norm(vel, dim=-1, keepdim=True)), dim=-1)
        x2 = x1.permute((0, 3, 1, 2))  # (torch.Size([64, 10, 22, 7])
        x3 = x2

        for i in range(len(self.st_gcnns)):
            x3 = self.st_gcnns[i](x3)

        x5 = x3.permute(0, 2, 1, 3)  # prepare the input for the Time-Extrapolator-CNN (NCTV->NTCV)

        x6 = self.prelus[0](self.txcnns[0](x5))
        for i in range(1, self.n_txcnn_layers):
            x6 = self.prelus[i](self.txcnns[i](x6)) + x6  # residual connection

        x6 = self.dim_conversor(x6.permute(0, 2, 1, 3)).permute(0, 2, 3, 1)
        x7 = x6.cumsum(1)

        act = self.context_layer(x7.reshape(b, 1, self.n_output, joints * x7.shape[-1]))
        x8 = x7.permute(0, 3, 2, 1)
        for i in range(len(self.st_gcnns_o)):
            x8 = self.st_gcnns_o[i](x8)
        x9 = x8.permute(0, 3, 2, 1) + act

        return x[:, -1:] + x9,
