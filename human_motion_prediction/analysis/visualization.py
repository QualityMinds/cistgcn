from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import animation
from scipy.spatial.transform import Rotation as R
from ..utils import body_utils


def plot_interpretations(info_layer, output_path, title="empty_info", db="h36m", dim_used=None):
    conn_idx, joint_names = body_utils.get_reduced_skeleton(db, dim_used)
    if len(info_layer.shape) == 1:
        if len(info_layer) < 50:
            xticks_names = joint_names if len(info_layer) == len(joint_names) else None
            plot_vector(info_layer,
                        title=title,
                        output_path=output_path,
                        xticks_names=xticks_names)
        else:
            if len(info_layer) % len(joint_names) == 0:
                xticks_names = joint_names if info_layer.reshape(-1, len(joint_names)).shape[1] == len(
                    joint_names) else None
                plot_correlation(info_layer.reshape(-1, len(joint_names)),
                                 title=title,
                                 output_path=output_path,
                                 xticks_names=xticks_names)
            else:
                print(f'{title} has an unrecognized format, accepted formats are: Matrix, vector and float')
    elif len(info_layer.shape) == 2:
        plot_correlation(info_layer,
                         title=title,
                         output_path=output_path)
    elif len(info_layer.shape) == 3:
        sz = np.array(info_layer.shape)
        if sz[1] == sz[2]:
            xticks_names = None
            if len(info_layer[0]) % len(joint_names) == 0:
                xticks_names = joint_names if info_layer[0].reshape(-1, len(joint_names)).shape[1] == len(
                    joint_names) else None
            for i, corr_m in enumerate(info_layer):
                plot_correlation(corr_m,
                                 title=f'{title}-{i}',
                                 output_path=output_path.parent.joinpath(f'{output_path.stem}-{i}{output_path.suffix}'),
                                 xticks_names=xticks_names)
            corr_m = info_layer.mean(0)
            plot_correlation(corr_m,
                             title=f'{title}-mean'.replace(".Adj-mean", ""),
                             output_path=output_path.parent.joinpath(f'{output_path.stem}-mean{output_path.suffix}'),
                             xticks_names=xticks_names)
        elif 3 in sz:
            if (sz[1:] != (len(joint_names), 3)).all():
                idx3 = np.where(3 == sz)[0][0]
                idx2 = np.where(len(joint_names) == sz)[0][0]
                idx1 = np.setdiff1d((0, 1, 2), [idx2, idx3])[0]
                info_layer = np.transpose(info_layer, (idx1, idx2, idx3))
            create_animation(Path(output_path.parent).joinpath(f'{output_path.stem.split("__")[0]}_{title}_3D.gif'),
                             [info_layer], mode="test",
                             plot_joints=True, db=db, repeat=2, online_plot=False, dim_used=dim_used)

        else:
            print(f'{title} has an unrecognized format, accepted formats are: Matrix, vector and float')
    elif isinstance(info_layer, (int, float, str, np.ndarray)):
        plot_number(info_layer,
                    title=title,
                    output_path=str(output_path).replace(title, "").replace("png", "txt"))
    else:
        print(f'{title} has an unrecognized format, accepted formats are: Matrix, vector and float')


def plot_vector(vector, title, output_path, xticks_names=None):
    numbers = np.arange(len(vector))
    plt.figure(figsize=(16, 10))
    plt.stem(vector)
    plt.xticks(numbers, numbers)
    for x, y in zip(numbers, vector):
        plt.annotate('{:.2f}'.format(y), xy=(x, y), xytext=(0, 5), textcoords='offset points', ha='center')
    if xticks_names is not None:
        plt.xticks(numbers, xticks_names, rotation=90)
    plt.title(title, fontsize=16)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close("all")


def plot_number(number, title, output_path):
    with open(output_path, 'a') as f:
        f.write(f'{title}: {number:.4f}\n')


def plot_correlation(matrix, title, output_path, xticks_names=None):
    minM = matrix.min()
    maxM = matrix.max()
    matrix = (matrix - minM) / (maxM - minM)

    # Weird behaviour, must be updated in the future to look like a correlation matrix.
    # if matrix.shape[1] == matrix.shape[0]:
    #     matrix = matrix / (np.diagonal(matrix) + 1e-8)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.matshow(matrix)
    plt.title(title, fontsize=32)
    # plt.suptitle(f'min:{minM:.2f}, max:{maxM:.2f}', x=0.5, y=1.15, fontsize=12)

    # if xticks_names is not None:
    #     plt.xticks(np.arange(matrix.shape[1]), xticks_names, rotation=90)
    #     if matrix.shape[1] == matrix.shape[0]:
    #         plt.yticks(np.arange(matrix.shape[1]), xticks_names)
    plt.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.8, 0.1, 0.1, 0.8])
    cb = plt.colorbar(im, cax=cbar_ax)
    cb.ax.tick_params(labelsize=32)
    plt.savefig(output_path, format='eps', bbox_inches='tight')
    plt.close("all")


def _define_rot(angles=np.pi / 2 * np.array([-1, 0, 0])):
    return R.from_rotvec(angles)


# REPLACE this [1, 0, 0] to [0, 0, 0] see actual body rotation
rot = _define_rot(angles=np.pi / 2 * np.array([1, 0, 0]))


def _get_cmap(n, name='jet'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    cmap = plt.cm.get_cmap(name, n)
    cmap = [cmap(i)[:3] for i in range(cmap.N)]
    return cmap


def extract_images_from_gif(gif_path, req_num_frames=None, return_images=False):
    gif_path = Path(gif_path)
    if not return_images:
        root_folder = Path(gif_path.stem)
        root_folder.mkdir(parents=True, mode=0o770, exist_ok=True)
    images = []
    with Image.open(gif_path) as im:
        if req_num_frames is None:
            num_key_frames = im.n_frames
        else:
            num_key_frames = float(req_num_frames)
        for i in range(num_key_frames):
            im.seek(im.n_frames // num_key_frames * i)
            if not return_images:
                im.save(root_folder.joinpath(f'{i}.png'))
            else:
                images.append(np.array(im.convert('RGB')))
    if len(images) == 0:
        images = None
    return images


def init_figure(seq, data, mode, plot_joints, db, size=75, color=["g", "r"],
                truncate_view=None,
                dim_used=None,
                view=None,
                hide_grid=False):
    fig = plt.figure(figsize=(10, 10), frameon=False)  # DO NOT CHANGE THIS SIZE, CHANGE dip parameter INSTEAD.
    fig.subplots_adjust(left=0, right=1., wspace=0., top=1.0, bottom=0., hspace=0.)  # -0.65
    axs_list_info = []
    axs_list = []
    xmin = data[0].min()
    xmax = data[0].max()
    ref_batch = data[0].shape[0]
    if truncate_view is not None:
        xmin = truncate_view[0]
        xmax = truncate_view[1]
    plot_number = 220
    last_plot = 5
    if mode == "train" or mode == "single" or mode == "one":
        last_plot = 2
        plot_number = 110
    for i in range(1, last_plot):
        axs_list_info.append([])
        axs = fig.add_subplot(plot_number + i, projection='3d')
        if hide_grid:
            axs.grid(False)
            axs.set_xticks([])
            axs.set_yticks([])
            axs.set_zticks([])
            plt.axis('off')
        axs.dist = 0
        axs_list.append(axs)
        axs.set_xlabel('x axis')
        axs.yaxis.set_label_text('y axis')
        axs.zaxis.set_label_text('z axis')
        axs.set_xlim3d(xmin, xmax)
        axs.set_ylim3d(xmin, xmax)
        axs.set_zlim3d(xmin, xmax)
        if view is None:
            axs.view_init(elev=20, azim=0)
        else:
            axs.view_init(elev=view[0], azim=view[1])
        for idx, pcl in enumerate(data):
            pcl = pcl[seq % ref_batch]
            if db != "expi":
                pcl = rot.apply(pcl)
                axs_list_info[-1].append(axs.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], color=color[idx], s=size))
            else:
                axs_list_info[-1].append(
                    axs.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2],
                                c=pcl.shape[0] // 2 * [color[idx], color[idx + 1]], s=size))
            conns, joint_names = body_utils.get_reduced_skeleton(db, dim_used)
            for _, conn_idx in enumerate(conns):
                if db != "expi":
                    axs_list_info[-1].append(
                        axs.plot(pcl[conn_idx, 0], pcl[conn_idx, 1], pcl[conn_idx, 2], linewidth=2, color=color[idx]))
                else:
                    axs_list_info[-1].append(
                        axs.plot(pcl[conn_idx, 0], pcl[conn_idx, 1], pcl[conn_idx, 2], linewidth=2,
                                 color=color[2 * idx + conn_idx[0] // 17]))

            if plot_joints:
                for txt in range(pcl.shape[0]):
                    axs_list_info[-1].append(
                        axs.text(pcl[txt, 0], pcl[txt, 1], pcl[txt, 2], str(txt), size=10, zorder=1, color='k'))
    return fig, axs_list, axs_list_info


def update(seq, data, axs_list, axs_list_info, db, plot_joints=False, dim_used=None, view=None):
    ref_batch = data[0].shape[0]
    # print(seq, seq % ref_batch, f'{40 * (seq % ref_batch + 1)} ms')
    for figure, axs in enumerate(axs_list):
        axs.set_title(f'{40 * (seq % ref_batch + 1)} ms', y=0.95)
        if view is None:
            if figure == 0:
                axs.view_init(elev=20, azim=-90 + 2 * seq)
            elif figure == 1:
                axs.view_init(elev=20, azim=-90)
            elif figure == 2:
                axs.view_init(elev=20, azim=0)
            else:
                axs.view_init(elev=80, azim=-90)
        else:
            axs.view_init(elev=view[0], azim=view[1])
        # TODO: hard coding, must be removed in future releases "Path3DCollection" and "Line3D"
        scatters = [g for g in axs_list_info[figure] if "Path3DCollection" in str(g)]
        skeletons = [g for g in axs_list_info[figure] if "Line3D" in str(g)]
        if plot_joints:
            texts = [g for g in axs_list_info[figure] if "Text" in str(g)]
        for i, (pcl, scatter) in enumerate(zip(data, scatters)):
            pcl = pcl[seq % ref_batch]
            if db != "expi": pcl = rot.apply(pcl)
            scatter._offsets3d = pcl.transpose((1, 0))
            conns, joint_names = body_utils.get_reduced_skeleton(db, dim_used)  # Connections - edges - bones
            for idx, conn_idx in enumerate(conns):
                skeletons[len(conns) * i + idx][0].set_data_3d((pcl[conn_idx, 0], pcl[conn_idx, 1], pcl[conn_idx, 2]))
            if plot_joints:
                for txt in range(pcl.shape[0]):
                    texts[pcl.shape[0] * i + txt].set_position((pcl[txt, 0], pcl[txt, 1]))
        plt.draw()


# online_plot is not working properly jet.
def create_animation(output_path, data, mode=None, db="cmu",
                     plot_joints=False,
                     truncate_view=None,
                     online_plot=False,
                     repeat=2,
                     interval=250,
                     dim_used=None,
                     view=None,
                     hide_grid=False,
                     multi_pose_color_patch=0.2,
                     colors=None):
    if colors is None:
        if len(data) < 3 and db != "expi":
            cmap = ["g", "r", "b"]
        elif db == "expi":  # use a similar color for the second pose
            cmap = _get_cmap(len(data))
            pose_2 = (np.array(_get_cmap(len(data))) + multi_pose_color_patch).tolist()
            for c, p in enumerate(pose_2): cmap.insert(2 * c + 1, p)
        else:
            cmap = _get_cmap(len(data))
        print(cmap)
    else:
        cmap = colors
    if mode is None:
        raise ValueError("Visualization mode was not correctly defined.")
    fig, axs_list, axs_list_info = init_figure(0, data, mode, plot_joints,
                                               db=db,
                                               color=cmap,
                                               truncate_view=truncate_view,
                                               dim_used=dim_used,
                                               hide_grid=hide_grid,
                                               view=view)
    N = data[0].shape[0] * repeat
    ani = animation.FuncAnimation(fig, update, N,
                                  fargs=(data, axs_list, axs_list_info, db, plot_joints, dim_used, view),
                                  interval=interval, blit=False, repeat=True)
    ani.save(output_path, dpi=100)  # change dip if bigger image is needed.
    if online_plot:
        plt.show()
    plt.close("all")
