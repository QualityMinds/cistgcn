# import matplotlib
# matplotlib.use('Qt5Agg') # have issues on cluster
import matplotlib.pyplot as plt
import numpy as np

from . import visualization as viz
from ..utils import body_utils


class Sequence:
    """
    Master class to structure the input data, basically it contains the sequence and dimensions
    """

    def __init__(self, data, dim_used=None):
        if not isinstance(data, np.ndarray):
            self.data = data.dataset.target
        else:
            if len(data.shape) < 3 and len(data.shape) > 4:
                raise ValueError(f'Invalid Input size:', data.shape, 'it must contain 3-4 dimensions')
            elif len(data.shape) == 3:
                self.data = data[None, ...]
            else:
                self.data = data
        if dim_used is not None:
            self.data = self.data[:, :, dim_used]
        self.n_samples, self.n_frames, self.n_joints, self.n_dims = self.data.shape


class Features:
    """
    Class to compute some stats. as simple as it sounds. could be contralled by index or dimensions
    (dimensions are not validated yet)
    """

    @staticmethod
    def angle_between(v1, v2, dim):
        def unit_vector(vector, dim=-1):
            """ Returns the unit vector of the vector.  """
            return vector / np.linalg.norm(vector, axis=dim, keepdims=True)

        v1_u = unit_vector(v1, dim=dim)
        v2_u = unit_vector(v2, dim=dim)
        return np.arccos(np.clip(np.einsum('bijk,bijk->bij', v1_u, v2_u), -1.0, 1.0))
        # return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def compute_mean(self, dims=(1, 2)):
        self.db.means = self.db.data.mean(dims)

    def compute_std(self, dim=(1, 2), idx=None):
        if idx is None:
            self.db.stds = self.db.data.std(dim)
        else:
            self.db.stds = self.db.data[idx].std((dim[0], dim[1]))

    def compute_angles(self, domain, dim=-1, idx=None, mode=None):
        if domain == "temporal":
            if idx is None:
                if "rel" in mode:
                    v2 = np.tile(self.db.data[:, 0:1], [self.db.data.shape[1], 1, 1])
                else:
                    v2 = self.db.data
                v1 = self.db.data
            else:
                if "rel" in mode:
                    v2 = np.tile(self.db.data[idx, 0:1], [self.db.data.shape[1], 1, 1])
                else:
                    v2 = self.db.data[idx]
                v1 = self.db.data[idx]
        else:
            AssertionError(f'please, choose domain equal to: spatial or temporal')
        if idx is None:
            self.db.angles = self.angle_between(v1, v2, dim=dim)
        else:
            self.db.angles = self.angle_between(v1[None, :], v2[None, :], dim=dim)[0]

    def compute_velocities(self, dim=1, idx=None, mode=None):
        if idx is None:
            if "rel" in mode:
                self.db.velocities = np.expand_dims(np.take(self.db.data, 0, axis=dim), dim) - self.db.data[:, 1:]
            else:
                self.db.velocities = np.diff(self.db.data, axis=dim - 1)
        else:
            if "rel" in mode:
                self.db.velocities = np.expand_dims(np.take(self.db.data[idx], 0, axis=dim - 1),
                                                    dim - 1) - self.db.data[idx, 1:]
            else:
                self.db.velocities = np.diff(self.db.data[idx], axis=dim - 1)

    def compute_accelerations(self, dim=1, idx=None, mode=None):
        if idx is None:
            if "rel" in mode:
                acc = np.expand_dims(np.take(self.db.data, 0, axis=dim), dim) - self.db.data[:, 1:]
                self.db.accelerations = np.expand_dims(np.take(acc, 0, axis=dim), dim) - acc[:, 1:]
            else:
                self.db.accelerations = np.diff(np.diff(self.db.data, axis=dim), axis=dim)
        else:
            if "rel" in mode:
                acc = np.expand_dims(np.take(self.db.data[idx], 0, axis=dim - 1), dim - 1) - self.db.data[idx, 1:]
                self.db.accelerations = np.expand_dims(np.take(acc, 0, axis=dim - 1), dim - 1) - acc[1:]
            else:
                self.db.accelerations = np.diff(np.diff(self.db.data[idx], axis=dim - 1), axis=dim - 1)


class SequenceAnalytics(Features):
    def __init__(self, data, db="cmu", dim_used=None, remove_temporal_data=False):
        """
        Args:
            data: must be numpy array or Pytorch loader
            db: dataset format
            dim_used: dimension used strongly dependent from db. [deprecated for future releases]
            remove_temporal_data: remove temporal data from dataset.
        """
        # super(SequenceAnalytics, self).__init__()
        self.remove_temporal_data = remove_temporal_data
        self.db = Sequence(data, dim_used=dim_used)
        self.conns, self.names = body_utils.get_reduced_skeleton(db, dim_used=dim_used)
        print("If you want to remove temporal intermediate data, \nPlease set: remove_temporal_data=True")

    def show(self, name=None, show=False):  # plot or save the figure.
        if name is not None:
            plt.savefig(name, bbox_inches='tight')
        if show or name is None:
            print("to save the figure, \nPlease set: name=xxx.png")
            plt.show()

    def init_figure(self, size=(12, 12)):
        self.fig = plt.figure(figsize=size)

    def plotGIF_sequence(self, name,
                         idx=None,
                         data=None,
                         fig_args=None):  # TODO: Make it more configurable such as the original file.
        # Plot 3D mesh. # scaling previous version.
        if data is None:
            data = [self.db.data[idx]]
        assert isinstance(data, (list, tuple))
        assert data[0].shape[2] == 3
        viz.create_animation(name,
                             data,
                             **fig_args)
        if self.remove_temporal_data:
            del data  # always delete data to save memory.

    def plot3D_means(self, under_sampling=1):
        self.colors = viz._get_cmap(1)
        self.ax = self.fig.add_subplot(projection='3d')
        if not hasattr(self, "means"): self.compute_mean()
        self.ax.scatter(self.db.means[::under_sampling, 0],
                        self.db.means[::under_sampling, 1],
                        self.db.means[::under_sampling, 2],
                        color=self.colors[0], marker='.')
        if self.remove_temporal_data:
            del self.db.means  # always delete data to save memory.

    def _get_index_joints_given_names_list(self, joints):
        assert isinstance(joints, (np.ndarray, list, tuple))
        filter_joints = []
        for j in joints:
            if len(np.where(np.array(self.names) == j)[0]) > 0:
                filter_joints.append(np.where(np.array(self.names) == j)[0][0])
            else:
                raise ValueError("joint names are not valid at all. Look at the dataset options:", self.names)
        filter_joints = np.array(filter_joints)
        return filter_joints

    @staticmethod
    def _compute_pseudo_norm(data, mode):
        if mode == "norm":
            mag = np.linalg.norm(data, ord=2, axis=-1)
        elif mode == "raw_sum":
            mag = data.sum(-1)
        else:
            AssertionError("This mode ivelocitiess not a valid one. Please choose: 'norm' or 'raw_sum'")
        return mag

    def Plot2D_joint_angle_displacement(self, idx, joints=None, ylim=None, module="norm", mode="absolute",
                                        linestyle='-', input_n=None, subplot=111):
        self.colors = viz._get_cmap(self.db.n_joints)
        self.compute_angles("temporal", idx=idx, mode=mode)
        angles = self.db.angles
        names = self.names
        if joints is not None:
            idxs = self._get_index_joints_given_names_list(joints)
            angles = angles[:, idxs]
            self.colors = viz._get_cmap(len(idxs))
            names = np.array(names)[idxs].tolist()

        if linestyle == "-":
            self.ang_ax = self.fig.add_subplot(subplot)
            self.ang_ax.grid()
        self.ang_ax.set_title(f'Angle displacement in time - sample id:{idx}')
        self.ang_ax.set_xlabel("frames")
        self.ang_ax.set_ylabel("Angle [0, pi]")

        for j, speed_line in enumerate(angles.T):
            self.ang_ax.plot(speed_line, color=np.array(self.colors)[j], linestyle=linestyle, label=names[j])
        if linestyle == "-":
            self.ang_ax.legend()
        if ylim is not None:
            self.ang_ax.set_ylim(ylim)
        if input_n is not None:
            self.ang_ax.axvline(x=input_n, color='r')
        if self.remove_temporal_data:
            del self.db.angles  # always delete data to save memory.

    def Plot2D_joint_velocities(self, idx, joints=None, ylim=None, module="norm", mode="absolute",
                                linestyle='-', input_n=None, subplot=111):
        self.colors = viz._get_cmap(self.db.n_joints)
        self.compute_velocities(idx=idx, mode=mode)
        velocities = self.db.velocities
        names = self.names
        if joints is not None:
            idxs = self._get_index_joints_given_names_list(joints)
            velocities = velocities[:, idxs]
            self.colors = viz._get_cmap(len(idxs))
            names = np.array(names)[idxs].tolist()

        if linestyle == "-":
            self.vel_ax = self.fig.add_subplot(subplot)
            self.vel_ax.grid()
        self.vel_ax.set_title(f'Velocity XYZ-vector module - sample id:{idx}')
        self.vel_ax.set_xlabel("frames")
        self.vel_ax.set_ylabel("velocity magnitude")

        mag = self._compute_pseudo_norm(velocities, module)
        for j, angle_line in enumerate(mag.T):
            self.vel_ax.plot(angle_line, color=np.array(self.colors)[j], linestyle=linestyle, label=names[j])
        if linestyle == "-":
            self.vel_ax.legend()
        if ylim is not None:
            self.vel_ax.set_ylim(ylim)
        if input_n is not None:
            self.vel_ax.axvline(x=input_n, color='r')
        if self.remove_temporal_data:
            del self.db.velocities  # always delete data to save memory.

    def Plot2D_joint_accelerations(self, idx, joints=None, ylim=None, module="norm", mode="absolute",
                                   linestyle='-', input_n=None, subplot=111):
        self.colors = viz._get_cmap(self.db.n_joints)
        self.compute_accelerations(idx=idx, mode=mode)
        accelerations = self.db.accelerations
        names = self.names
        if joints is not None:
            idxs = self._get_index_joints_given_names_list(joints)
            accelerations = accelerations[:, idxs]
            self.colors = viz._get_cmap(len(idxs))
            names = np.array(names)[idxs].tolist()

        if linestyle == "-":
            self.acc_ax = self.fig.add_subplot(subplot)
            self.acc_ax.grid()
        self.acc_ax.set_title(f'Acceleration XYZ-vector module - sample id:{idx}')
        self.acc_ax.set_xlabel("frames")
        self.acc_ax.set_ylabel("acceleration magnitude")

        mag = self._compute_pseudo_norm(accelerations, module)
        for j, acc_line in enumerate(mag.T):
            self.acc_ax.plot(acc_line, color=np.array(self.colors)[j], linestyle=linestyle, label=names[j])
        if linestyle == "-":
            self.acc_ax.legend()
        if ylim is not None:
            self.acc_ax.set_ylim(ylim)
        if input_n is not None:
            self.acc_ax.axvline(x=input_n, color='r')
        if self.remove_temporal_data:
            del self.db.accelerations  # always delete data to save memory.

    def Plot2D_joint_positions(self, idx, joints=None, ylim=None, module="norm", mode="absolute",
                               linestyle='-', input_n=None, subplot=111):
        self.colors = viz._get_cmap(self.db.n_joints)
        position = self.db.data[idx]
        names = self.names
        if joints is not None:
            idxs = self._get_index_joints_given_names_list(joints)
            position = position[:, idxs]
            self.colors = viz._get_cmap(len(idxs))
            names = np.array(names)[idxs].tolist()

        if linestyle == "-":
            self.pos_ax = self.fig.add_subplot(subplot)
            self.pos_ax.grid()
        self.pos_ax.set_title(f'Position XYZ-vector module - sample id:{idx}')
        self.pos_ax.set_xlabel("frames")
        self.pos_ax.set_ylabel("position magnitude")

        mag = self._compute_pseudo_norm(position, module)
        for j, acc_line in enumerate(mag.T):
            self.pos_ax.plot(acc_line, color=np.array(self.colors)[j], linestyle=linestyle, label=names[j])
        if linestyle == "-":
            self.pos_ax.legend()
        if input_n is not None:
            self.pos_ax.axvline(x=input_n, color='r')
        if ylim is not None:
            self.pos_ax.set_ylim(ylim)

    def _get_function(self, name):
        funcs = {"position": self.Plot2D_joint_positions,
                 "velocity": self.Plot2D_joint_velocities,
                 "acceleration": self.Plot2D_joint_accelerations,
                 "angle": self.Plot2D_joint_angle_displacement}
        derivative = {"position": 0,
                      "velocity": 1,
                      "acceleration": 2,
                      "angle": 1}
        idx = np.argmax([name[:3] in n for n in funcs])
        keys = list(funcs.keys())
        return funcs[keys[idx]], derivative[keys[idx]]

    def Plot2D_joint_physics(self, eval_physical_config, idx=0,
                             global_config=None,
                             joints=None,
                             mode="absolute",
                             prediction=False,
                             input_n=None):
        """
        plot the position, velocity and acceleration graphic for a sequence sample.
        args:
            joints: joints used to plot. just a list of names.
            ylim: can be a list of lists of just a list. Ex: [[-10,10],None,[0,10]]
            mode: way to compute the module of a vector, L2 or pseudo_raw_sum: options are: 'norm' or 'raw_sum'
        """

        funcs = {}
        config = eval_physical_config if isinstance(eval_physical_config, dict) else eval_physical_config.__dict__
        for i, name in enumerate(config):
            funcs[name] = {}
            funcs[name]["func"], derivative = self._get_function(name)
            funcs[name]["args"] = {}
            if global_config is not None:
                if not isinstance(global_config, dict): g_conf = global_config.__dict__
                funcs[name]["args"].update(global_config if isinstance(global_config, dict) else global_config.__dict__)
            if hasattr(eval_physical_config, name):
                if eval_physical_config.__getattribute__(name) is not None:
                    physic_conf = eval_physical_config.__getattribute__(name)
                    physic_conf = physic_conf if isinstance(physic_conf, dict) else physic_conf.__dict__
                    funcs[name]["args"].update(physic_conf)
            if prediction:
                funcs[name]["args"].update({"linestyle": '--'})
                if input_n is not None:
                    funcs[name]["args"].update({"input_n": input_n - derivative})

        for i, name in enumerate(funcs):
            print(f'Plotting {name}')
            funcs[name]["func"](idx, joints=joints, mode=mode,
                                subplot=100 * len(funcs) + 11 + i,
                                **funcs[name]["args"])
