import argparse
import os
from pprint import pprint


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--data_dir', type=str, default='/mnt/DataDrive164/zhouhonghong/h36m',
                                 help='path to H36M dataset')
        self.parser.add_argument('--data_dir_3dpw', type=str, default='/mnt/DataDrive164/zhouhonghong/cmu_mocap',
                                 help='path to 3DPW dataset')
        self.parser.add_argument('--data_dir_cmu', type=str, default='/mnt/DataDrive164/zhouhonghong/cmu_mocap',
                                 help='path to CMU dataset')
        self.parser.add_argument('--exp', type=str, default='train', help='ID of experiment')
        self.parser.add_argument('--ckpt', type=str, default='checkpoint/', help='path to save checkpoint')

        # ===============================================================
        #                     Model options
        # ===============================================================
        self.parser.add_argument('--max_norm', dest='max_norm', action='store_true',
                                 help='maxnorm constraint to weights')
        self.parser.add_argument('--linear_size', type=int, default=256, help='size of each model layer')
        self.parser.add_argument('--num_stage', type=int, default=6, help='# layers in linear model')
        self.parser.add_argument('--num_modules', type=int, default=3, help='layers for Multiscale encoder module')
        self.parser.add_argument('--data_format', type=str, default='h36m_3d', help='[h36m, h36m_3d, cmu, cmu_3d]')
        self.parser.add_argument('--decoder', type=str, default='sequential')

        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--test_mode', type=bool, default=False,
                                 help='whether to sample part of the dataset for test model')
        self.parser.add_argument('--lr', type=float, default=5.0e-4)
        self.parser.add_argument('--lr_decay', type=int, default=2, help='every lr_decay epoch do lr decay')
        self.parser.add_argument('--lr_gamma', type=float, default=0.96)
        self.parser.add_argument('--input_n', type=int, default=10, help='observed seq length')
        self.parser.add_argument('--output_n', type=int, default=25, help='future seq length')
        self.parser.add_argument('--dct_n', type=int, default=35, help='number of DCT coeff. preserved for 3D')
        self.parser.add_argument('--actions', type=str, default='all', help='path to save checkpoint')
        self.parser.add_argument('--epochs', type=int, default=100)
        self.parser.add_argument('--dropout', type=float, default=0.5,
                                 help='dropout probability, 1.0 to make no dropout')
        self.parser.add_argument('--train_batch', type=int, default=16)
        self.parser.add_argument('--test_batch', type=int, default=64)
        self.parser.add_argument('--job', type=int, default=10, help='subprocesses to use for data loading')
        self.parser.add_argument('--is_load', dest='is_load', action='store_true', help='wether to load existing model')
        self.parser.add_argument('--sample_rate', type=int, default=2, help='frame sampling rate')
        self.parser.add_argument('--is_norm_dct', dest='is_norm_dct', action='store_true',
                                 help='whether to normalize the dct coeff')
        self.parser.add_argument('--is_norm', dest='is_norm', action='store_true',
                                 help='whether to normalize the angles/3d coordinates')

        self.parser.set_defaults(max_norm=True)
        self.parser.set_defaults(is_load=False)
        # self.parser.set_defaults(is_norm_dct=True)
        # self.parser.set_defaults(is_norm=True)

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        # make ckpt dir
        ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
        if not os.path.isdir(ckpt):
            os.makedirs(ckpt)
        self.opt.ckpt = ckpt
        # adjust scale splitting way
        data_formats = ['h36m', 'h36m_3d', 'cmu', 'cmu_3d']
        # default is h36m
        if self.opt.data_format == data_formats[0]:
            self.opt.node_s1 = 48
            self.opt.node_s2 = 30
            self.opt.node_s3 = 15
            self.opt.s2_split = [[0, 1, 2, 3],
                                 [4, 5, 6, 7],
                                 [8, 9, 10, 11],
                                 [12, 13, 14, 15],
                                 [16, 17, 18, 19, 20, 21],
                                 [22, 23, 24, 25, 26, 27],
                                 [28, 29, 30, 31, 32, 33],
                                 [34, 35, 36, 37],
                                 [38, 39, 40, 41, 42, 43],
                                 [44, 45, 46, 47]]
            self.opt.s3_split = [[0, 1, 2, 3, 4, 5, 6, 7],
                                 [8, 9, 10, 11, 12, 13, 14, 15],
                                 [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
                                 [28, 29, 30, 31, 32, 33, 34, 35, 36, 37],
                                 [38, 39, 40, 41, 42, 43, 44, 45, 46, 47]]
        # h36m_3d
        elif self.opt.data_format == data_formats[1]:
            self.opt.node_s1 = 66
            self.opt.node_s2 = 30
            self.opt.node_s3 = 15
            self.opt.s2_split = [range(0, 6),
                                 range(6, 12),
                                 range(12, 18),
                                 range(18, 24),
                                 range(24, 30),
                                 range(30, 36),
                                 range(36, 42),
                                 range(42, 51),
                                 range(51, 57),
                                 range(57, 66)]
            self.opt.s3_split = [range(0, 12),
                                 range(12, 24),
                                 range(24, 36),
                                 range(36, 51),
                                 range(51, 66)]
        # cmu
        elif self.opt.data_format == data_formats[2]:
            self.opt.node_s1 = 64
            self.opt.node_s2 = 30
            self.opt.node_s3 = 15
            self.opt.s2_split = [range(0, 5),
                                 range(5, 10),
                                 range(10, 15),
                                 range(15, 20),
                                 range(20, 26),
                                 range(26, 38),
                                 range(38, 43),
                                 range(43, 51),
                                 range(51, 56),
                                 range(56, 64)]
            self.opt.s3_split = [range(0, 10),
                                 range(10, 20),
                                 range(20, 38),
                                 range(38, 51),
                                 range(51, 64)]
        # cmu_3d
        elif self.opt.data_format == data_formats[3]:
            self.opt.node_s1 = 75
            self.opt.node_s2 = 30
            self.opt.node_s3 = 15
            self.opt.s2_split = [range(0, 6),
                                 range(6, 12),
                                 range(12, 18),
                                 range(18, 24),
                                 range(24, 30),
                                 range(30, 39),
                                 range(39, 48),
                                 range(48, 57),
                                 range(57, 66),
                                 range(66, 75)]
            self.opt.s3_split = [range(0, 12),
                                 range(12, 24),
                                 range(24, 39),
                                 range(39, 57),
                                 range(57, 75)]

        self._print()
        return self.opt
