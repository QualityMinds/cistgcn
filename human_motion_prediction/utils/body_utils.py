import matplotlib.pyplot as plt
import numpy as np


def get_reduced_skeleton(skeleton_type="cmu", dim_used=None, inverse=False):
    conns_reduced = None
    if skeleton_type == "expi":
        joint_names = ["L-fhead",  # (0)
                       "L-lhead",  # (1)
                       "L-rhead",  # (2)
                       "L-back",  # (3)
                       "L-lshoulder",  # (4)
                       "L-rshoulder",  # (5)
                       "L-lelbow",  # (6)
                       "L-relbow",  # (7)
                       "L-lwrist",  # (8)
                       "L-rwrist",  # (9)
                       "L-lhip",  # (10)
                       "L-rhip",  # (11)
                       "L-lknee",  # (12)
                       "L-rknee",  # (13)
                       "L-lheel",  # (14)
                       "L-rheel",  # (15)
                       "L-ltoes",  # (16)
                       "L-rtoes",  # (17)
                       "F-fhead",  # (18)
                       "F-lhead",  # (19)
                       "F-rhead",  # (20)
                       "F-back",  # (21)
                       "F-lshoulder",  # (22)
                       "F-rshoulder",  # (23)
                       "F-lelbow",  # (24)
                       "F-relbow",  # (25)
                       "F-lwrist",  # (26)
                       "F-rwrist",  # (27)
                       "F-lhip",  # (28)
                       "F-rhip",  # (29)
                       "F-lknee",  # (30)
                       "F-rknee",  # (31)
                       "F-lheel",  # (32)
                       "F-rheel",  # (33)
                       "F-ltoes",  # (34)
                       "F-rtoes",  # (35)
                       ]
        conns_reduced = [(0, 1), (0, 2), (0, 3),
                         (3, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9),
                         (3, 10), (3, 11),
                         (10, 12), (12, 14), (14, 16),
                         (11, 13), (13, 15), (15, 17),
                         ]
        conns_reduced = np.concatenate((np.array(conns_reduced), np.array(conns_reduced) + 18)).tolist()
    elif skeleton_type == "amass" or skeleton_type == "3dpw" or skeleton_type == "pw3d":
        conns_reduced = [(0, 1), (0, 2), (0, 3),
                         (1, 4), (5, 2), (3, 6),
                         (7, 4), (8, 5), (6, 9),
                         (7, 10), (8, 11), (9, 12),
                         (12, 13), (12, 14),
                         (12, 15),
                         # (13, 16), (14, 17),
                         (12, 16), (12, 17),
                         (12, 16), (12, 17),
                         (16, 18), (19, 17), (20, 18), (21, 19),
                         # (22, 20), #(23, 21),#wrists
                         # (1, 16), (2, 17) # hips to shoulders
                         ]
        inverse_mapping = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14),
                           (16, 17), (18, 19), (20, 21), (22, 23)
                           ]
        joint_names = ["Pelvis",  # 0
                       "LeftUpLeg",  # 1
                       "RightUpLeg",  # 2
                       "Spine1",  # 3, 1
                       "LeftKnee",  # 4, 2
                       "RightKnee",  # 5, 3
                       "Spine2",  # 6
                       "LeftAnkle",  # 7, 4
                       "RightAnkle",  # 8, 5
                       "Spine3",  # 9, 6
                       "LeftFoot",  # 10, 7
                       "RightFoot",  # 11
                       "Neck",  # 12, 8
                       "LeftCollar",  # 13, 9
                       "RightCollar",  # 14, 10
                       "Head",  # 15, 11
                       "LeftShoulder",  # 16
                       "RightShoulder",  # 17, 12
                       "LeftElbow",  # 18, 13
                       "RightElbow",  # 19, 14
                       "L_Wrist_End",  # 20
                       "R_Wrist_End",  # 21, 15
                       "LeftHand",  # 22
                       "RightHand",  # 23
                       ]
        if dim_used is not None:  # to be compatible with previous versions and for interpretation purposes.
            conns_reduced = [(0, 1), (0, 2), (0, 3),
                             (1, 4), (5, 2), (3, 6),
                             (7, 4), (8, 5), (6, 9),
                             (7, 10), (8, 11), (9, 12),
                             (12, 13), (12, 14),
                             (12, 15),
                             # (13, 16), (14, 17),
                             (12, 16), (12, 17),
                             (12, 16), (12, 17),
                             # (16, 18), (19, 17), (20, 18), (21, 19),
                             # (22, 20), #(23, 21),#wrists
                             # (1, 16), (2, 17) # hips to shoulders
                             ]
            inverse_mapping = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14),
                               (16, 17), (18, 19), (20, 21), (22, 23)
                               ]
            joint_names = np.array(joint_names)[dim_used].tolist()
    elif skeleton_type == "cmu":
        conns_reduced = [(0, 1), (0, 2),  # (0, 3),
                         (1, 4), (5, 2),  # (3, 6),
                         (7, 4), (8, 5),  # (6, 9),
                         (7, 10), (8, 11),  # (9, 12),
                         #  (12, 13), (12, 14),
                         (12, 15),
                         # (13, 16), (12, 16), (14, 17), (12, 17),
                         (12, 16), (12, 17),
                         (16, 18), (19, 17), (20, 18), (21, 19),
                         #  (22, 20), (23, 21),# wrists
                         (1, 16), (2, 17)
                         ]
        inverse_mapping = [(0, 4), (1, 5), (2, 6), (3, 7), (13, 19),
                           (14, 20), (15, 21), (16, 22), (17, 23),
                           (18, 24)
                           ]
        joint_names = ["L-Knee",  # 0
                       "L-Ankle",  # 1
                       "L-Heel",  # 2
                       "L-foot-index",  # 3
                       "R-Knee",  # 4
                       "R-Ankle",  # 5
                       "R-Heel",  # 6
                       "R-foot-index",  # 7
                       "Hip",  # 8
                       "Spine",  # 9
                       "Shoulder",  # 10
                       "Neck",  # 11
                       "Head",  # 12
                       "L-Shoulder",  # 13
                       "L-Elbow",  # 14
                       "L-Wrist",  # 15
                       "L-Index",  # 16
                       "L-Pinky",  # 17
                       "L-Thumb",  # 18
                       "R-Shoulder",  # 19
                       "R-Elbow",  # 20
                       "R-Wrist",  # 21
                       "R-Index",  # 22
                       "R-Pinky",  # 23
                       "R-Thumb"]  # 24
    elif skeleton_type == "h36m":
        # 32 human3.6 joint name:
        conns_reduced = [(1, 2), (2, 3), (3, 4), (4, 5),
                         (6, 7), (7, 8), (8, 9), (9, 10),
                         (0, 1), (0, 6),
                         (0, 11), (11, 13),  # added to cleaner representation view
                         # (6, 17),  # too cluttered
                         (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),
                         # (1, 25),  # too cluttered
                         (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
                         (24, 25), (24, 17),
                         (24, 14), (14, 15)
                         ]
        inverse_mapping = [(6, 1), (7, 2), (8, 3), (9, 4), (10, 5),
                           (16, 24), (17, 25), (18, 26), (19, 27),
                           (20, 28), (22, 30), (21, 29), (23, 31)
                           ]
        joint_names = ["Hips",  # 0
                       "RightUpLeg",  # 1
                       "RightLeg",  # 2, 0
                       "RightFoot",  # 3, 1
                       "RightToeBase",  # 4, 2
                       "Site",  # 5, 3
                       "LeftUpLeg",  # 6
                       "LeftLeg",  # 7, 4
                       "LeftFoot",  # 8, 5
                       "LeftToeBase",  # 9, 6
                       "Site",  # 10, 7
                       "Spine",  # 11
                       "Spine1",  # 12, 8
                       "Neck",  # 13, 9
                       "Head",  # 14, 10
                       "Site",  # 15, 11
                       "LeftShoulder",  # 16
                       "LeftArm",  # 17, 12
                       "LeftForeArm",  # 18, 13
                       "LeftHand",  # 19, 14
                       "LeftHandThumb",  # 20
                       "Site",  # 21, 15
                       "L_Wrist_End",  # 22, 16
                       "Site",  # 23
                       "RightShoulder",  # 24
                       "RightArm",  # 25, 17
                       "RightForeArm",  # 26, 18
                       "RightHand",  # 27, 19
                       "RightHandThumb",  # 28
                       "Site",  # 29, 20
                       "R_Wrist_End",  # 30, 21
                       "Site",  # 31
                       ]
        if dim_used is not None:  # to be compatible with previous versions and for interpretation purposes.
            conns_reduced = [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7],
                             [6, 7], [8, 9], [4, 8], [0, 8], [9, 10],
                             [10, 11], [18, 19], [19, 20], [19, 21],
                             [13, 14], [14, 15], [14, 16], [9, 12], [12, 13],
                             [9, 17], [17, 18],
                             ]
            inverse_mapping = [(6, 1), (7, 2), (8, 3), (9, 4), (10, 5),
                               (16, 24), (17, 25), (18, 26), (19, 27),
                               (20, 28), (22, 30), (21, 29), (23, 31)
                               ]
            joint_names = np.array(joint_names)[dim_used].tolist()
    if inverse == True:
        connections = inverse_mapping
    else:
        connections = conns_reduced
    return connections, joint_names


def plot_3d(pose3d):  # metrics["target"][5, -1]
    conns, joint_names = get_reduced_skeleton()
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pose3d[:, 0], pose3d[:, 1], pose3d[:, 2])
    ax.set_xlim3d([pose3d.min(), pose3d.max()])
    ax.set_ylim3d([pose3d.min(), pose3d.max()])
    ax.set_zlim3d([pose3d.min(), pose3d.max()])
    for conn_idx in conns:
        ax.plot(pose3d[conn_idx, 0], pose3d[conn_idx, 1], pose3d[conn_idx, 2], linewidth=5, color='black')
    plt.show()


def convert_points_to_plot(target, pred, get_color=True):
    offset = target[:, 0:1].min((2, 3), keepdims=True)
    target = target - offset
    pred = pred - offset
    pcl = np.dstack((target, pred))  # point_cloud_merge
    temporal_displacement = np.zeros_like(pcl)
    base = np.arange(0, pcl.shape[1])
    temporal_displacement[:, :, :, 0] = np.swapaxes(np.tile(base, (pcl.shape[0], pcl.shape[2], 1)), 1, 2)
    pcl = pcl + temporal_displacement * (1 + pred[0, :, :, 0].max() - pred[0, :, :, 0].min())
    colors = None
    if get_color:  # Would be nice if these lines are after pcl reshape. However, lin algebra does not allow it :(
        colors = np.zeros_like(pcl)
        colors[:, :, :pred.shape[2], :] = np.array([[[0, 255, 0]]])
        colors[:, :, pred.shape[2]:, :] = np.array([[[255, 0, 0]]])
    pcl = pcl.reshape(pcl.shape[0], -1, 3) / 5
    if get_color:
        colors = colors.reshape(pcl.shape[0], -1, 3)
    return {"pcl": pcl,
            "colors": colors, }


def create_symmetic_3d_edges(data, steps=10, db="cmu", dim_used=None):
    edges = data.copy()
    conn_idx, joint_names = get_reduced_skeleton(skeleton_type=db, dim_used=dim_used)
    for conn, jn in zip(conn_idx, joint_names):
        # largest_dist_instance = np.linalg.norm(data[:, :, conn_idx], 2, -1)
        new_edge = np.linspace(data[:, :, conn[0]], data[:, :, conn[1]], steps, axis=2)
        edges = np.dstack((edges, new_edge))
    return edges


if __name__ == "__main__":
    # Stop a debug breakpoint in return {"sample":
    self = {}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    idx = 2
    ax.scatter(self.proc_data[idx, :, 0], self.proc_data[idx, :, 1], self.proc_data[idx, :, 2])
    plt.show()
    ##############
    import analysis
    from pathlib import Path

    root_folder = Path("./")
    # Plot 3D mesh. # scaling previous version.
    analysis.create_animation(root_folder.joinpath(f'samples.gif'),
                              [self.proc_data, self.proc_data],
                              mode="train",  # also it is possible to change to train to get only 1 view.
                              plot_joints=True,
                              # [-2, 2],
                              online_plot=True,
                              db="h36m",
                              times=2)
