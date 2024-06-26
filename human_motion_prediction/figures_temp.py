from pathlib import Path

import matplotlib

matplotlib.use('Qt5Agg')  # have issues on cluster
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd


def _get_cmap(n, name='jet'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    cmap = plt.cm.get_cmap(name, n)
    cmap = [cmap(i)[:3] for i in range(cmap.N)]
    return cmap


STSGCN = Path("/home/eme/Projects/human-motion-prediction/logdir/STSGCN").rglob("*.xlsx")
MotionMixer = Path("/home/eme/Projects/human-motion-prediction/logdir/MotionMixer").rglob("*.xlsx")

STSGCN, MotionMixer = list(STSGCN), list(MotionMixer)
STSGCN.sort()
MotionMixer.sort()

db = "3dpw"
colors = _get_cmap(10)
fig, ax = plt.subplots(figsize=(8, 4))
cls = 0
for ma in MotionMixer:
    if db in str(ma):
        if ("adv_difference" in str(ma)) and (
            not "100" in str(ma)) and (
            not "50" in str(ma)) and (
            not "NOATTACK" in str(ma)
        ):
            data = pd.read_excel(ma, sheet_name="temporal_mpjpe")
            name = ma.stem.replace("_0.02", "")
            name = name[name.find(db) + len(db) + 1:name.find("__adv_difference")] + "iters"
            ax.plot(np.array(data["Unnamed: 0"][:10]), np.array(data["mean"][:10]), label=name, c=colors[cls])
            ax.plot(np.array(data["Unnamed: 0"][:10]), np.array(data["mean"][:10]), "*", c=colors[cls])
            cls += 1
plt.grid()
plt.ylabel("Δs", fontsize=15)
plt.xlabel("time", fontsize=15)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

###############################

colors = _get_cmap(15)
data = np.load("/home/eme/Projects/20221111_1223-id0734_best.npy", allow_pickle=True).all()
motion_classes = data.keys()

keys = [k for k in data["walking"]['interpretation'].keys() if
        "w1" in k or "w2" in k or "context_layer.joints" in k or "context_layer.displacements" in k]
keys = [k for k in data["walking"]['interpretation'].keys() if
        ("w1" in k or "w2" in k) and "_o." not in k]

samples, items, clss, mpjpe = [], [], [], []
for cls, m in enumerate(motion_classes):
    joints = np.array(data[m]['interpretation']['context_layer.joints'])
    joints = joints.reshape(np.int64(np.prod(joints.shape[:-1])), -1)
    dim = joints.shape[0]
    info_vecs = np.array([])
    for k in keys:
        inter_layer = np.array(data[m]['interpretation'][k])
        inter_layer = inter_layer.reshape(np.int64(np.prod(joints.shape[:-1])), -1)
        # inter_layer = (inter_layer - inter_layer.mean(1)[:, None]) / (inter_layer.std(1)[:, None] + 1e-8)
        if info_vecs.shape == (0,):
            info_vecs = inter_layer
        else:
            info_vecs = np.concatenate([info_vecs, inter_layer], 1)
    print(m, info_vecs.shape, info_vecs.mean(), info_vecs.std(), info_vecs.min(), info_vecs.max())
    samples.extend(info_vecs)
    items.append(np.array(data[m]['items']))
    clss.append(np.array([cls] * len(np.array(data[m]['items']))))
    metrics = np.array(data[m]['mpjpe_seq']).mean((1, 2))
    mpjpe.append(metrics)

samples = np.array(samples)
items = np.array(items)
clss = np.array(clss)
mpjpe = np.array(mpjpe).flatten()
samples = np.float32(samples)
print(samples.shape, samples.mean(), samples.std(), samples.min(), samples.max())

# pca = PCA(n_components=100)
# in_pca = pca.fit_transform(samples)
model = TSNE(n_components=2, n_iter=5000, init="random", learning_rate='auto', perplexity=50, verbose=2)
embedded = model.fit_transform(samples)

fig, ax = plt.subplots(figsize=(20, 20))
for cls, m in enumerate(motion_classes):
    ax.scatter(embedded[cls * dim:(cls + 1) * dim, 0], embedded[cls * dim:(cls + 1) * dim, 1],
               label=m, color=colors[cls], s=mpjpe[cls * dim:(cls + 1) * dim] * 2, marker="o")
plt.legend()
plt.tight_layout()
plt.savefig("tsne_wo_norm.png", dpi=400)

#############################################################
import cv2

path_images = "/home/eme/Projects/human-motion-prediction/logdir/h36m/32/20221111_1223-id0734_best/predict"
for cls_name in ["walking", "eating", "smoking", "discussion", "directions", "greeting", "phoning", "posing",
                 "purchases", "sitting", "sittingdown", "takingphoto", "waiting", "walkingdog", "walkingtogether"]:
    out_images = Path(path_images).glob('*.png')
    full_images = [str(i) for i in out_images if cls_name + "_" in str(i) and "average" in str(i)]
    full_images.sort()
    images = []
    for path in full_images:
        images.append(cv2.imread(path))
    if len(images) > 0:
        out_img = images[0]
        for img in images[1:]:
            out_img = np.vstack((out_img, img))
        cv2.imwrite(f'{cls_name}.png', out_img)
#############################################################
import cv2


def find_max_squared_area(target_area):
    side_length = np.sqrt(target_area)
    width = int(np.ceil(side_length))
    height = int(np.floor(side_length))
    actual_area = width * height
    while actual_area < target_area:
        width += 1
        actual_area = width * height
    return width, height, actual_area


path_images = "/home/eme/Projects/human-motion-prediction/logdir/h36m/32/20221111_1223-id0734_best/predict_full"
for cls_name in ["walking", "eating", "smoking", "discussion", "directions", "greeting", "phoning", "posing",
                 "purchases", "sitting", "sittingdown", "takingphoto", "waiting", "walkingdog", "walkingtogether"]:
    out_images = Path(path_images).glob('*.png')
    full_images = [str(i) for i in out_images if cls_name + "_" in str(i) and "average" in str(i)]
    full_images.sort()
    images = []
    for path in full_images:
        images.append(cv2.imread(path))
    if len(images) > 0:
        width, height, actual_area = find_max_squared_area(len(images))
        w, h, c = images[0].shape
        out_img = np.zeros([width * w, height * h, 3], dtype=np.uint8)
        for i, img in enumerate(images):
            out_img[i // width * w:(i // width + 1) * w, i % width * h:(i % width + 1) * h] = img
        cv2.imwrite(f'{cls_name}.png', out_img)
