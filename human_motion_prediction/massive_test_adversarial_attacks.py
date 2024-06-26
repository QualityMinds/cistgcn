#!/usr/bin/env python
import argparse
import subprocess

import numpy as np

from .utils import yaml_utils

parser = argparse.ArgumentParser()
parser.add_argument('evaluation_config', type=str, help='path to config via YAML file')
args = parser.parse_args()
opt = yaml_utils.load_yaml(args.evaluation_config, class_mode=True)

extension_path_o = opt.evaluation_config.sets[0].original_test.extension_path

opt.evaluation_config.sets[0].original_test.adversarial_attack.FGSM.joints = np.arange(32).tolist()
opt.evaluation_config.sets[0].original_test.adversarial_attack.FGSM.frames = np.arange(10).tolist()
opt.evaluation_config.sets[0].original_test.adversarial_attack.FGSM.epsilon = float(0)
extension_path = extension_path_o.replace("M_", f'M_original_____')
opt.evaluation_config.sets[0].original_test.extension_path = extension_path
yaml_utils.write_yaml(opt, 'temporal_test1.yaml', remote_struct=True)
subprocess.call(["python3", "evaluate.py", "temporal_test1.yaml"])

for epsilon in np.arange(0.005, 0.1001, 0.005):
    for f in np.arange(0, 8):
        epsilon = np.round(epsilon, 5)
        f = np.round(f, 5)
        opt.evaluation_config.sets[0].original_test.adversarial_attack.FGSM.joints = np.arange(32).tolist()
        opt.evaluation_config.sets[0].original_test.adversarial_attack.FGSM.frames = np.sort(
            np.arange(8, f, -1)).tolist()
        opt.evaluation_config.sets[0].original_test.adversarial_attack.FGSM.epsilon = float(epsilon)
        extension_path = extension_path_o.replace("M_", f'M_f_81_{f}j_eps_{epsilon}')
        opt.evaluation_config.sets[0].original_test.extension_path = extension_path

        yaml_utils.write_yaml(opt, 'temporal_test1.yaml', remote_struct=True)
        subprocess.call(["python3", "evaluate.py", "temporal_test1.yaml"])

for epsilon in np.arange(0.005, 0.1001, 0.005):
    for f in np.arange(2, 10):
        epsilon = np.round(epsilon, 5)
        f = np.round(f, 5)
        opt.evaluation_config.sets[0].original_test.adversarial_attack.FGSM.joints = np.arange(32).tolist()
        opt.evaluation_config.sets[0].original_test.adversarial_attack.FGSM.frames = np.arange(1, f).tolist()
        opt.evaluation_config.sets[0].original_test.adversarial_attack.FGSM.epsilon = float(epsilon)
        extension_path = extension_path_o.replace("M_", f'M_f_18_{f}j_eps_{epsilon}')
        opt.evaluation_config.sets[0].original_test.extension_path = extension_path

        yaml_utils.write_yaml(opt, 'temporal_test1.yaml', remote_struct=True)
        subprocess.call(["python3", "evaluate.py", "temporal_test1.yaml"])
#
for epsilon in np.arange(0.005, 0.1001, 0.005):
    for f in np.arange(1, 11):
        epsilon = np.round(epsilon, 5)
        f = np.round(f, 5)
        opt.evaluation_config.sets[0].original_test.adversarial_attack.FGSM.joints = np.arange(32).tolist()
        opt.evaluation_config.sets[0].original_test.adversarial_attack.FGSM.frames = np.arange(f).tolist()
        opt.evaluation_config.sets[0].original_test.adversarial_attack.FGSM.epsilon = float(epsilon)
        extension_path = extension_path_o.replace("M_", f'M_f{f}j_eps_{epsilon}')
        opt.evaluation_config.sets[0].original_test.extension_path = extension_path

        yaml_utils.write_yaml(opt, 'temporal_test.yaml', remote_struct=True)
        subprocess.call(["python3", "evaluate.py", "temporal_test.yaml"])

for epsilon in np.arange(0.005, 0.1001, 0.005):
    for j in np.arange(32):
        epsilon = np.round(epsilon, 5)
        j = np.round(j, 5)
        opt.evaluation_config.sets[0].original_test.adversarial_attack.FGSM.joints = [j]
        opt.evaluation_config.sets[0].original_test.adversarial_attack.FGSM.frames = np.arange(10).tolist()
        opt.evaluation_config.sets[0].original_test.adversarial_attack.FGSM.epsilon = float(epsilon)
        extension_path = extension_path_o.replace("M_", f'M_j{j}f_eps_{epsilon}')
        opt.evaluation_config.sets[0].original_test.extension_path = extension_path

        yaml_utils.write_yaml(opt, 'temporal_test.yaml', remote_struct=True)
        subprocess.call(["python3", "evaluate.py", "temporal_test.yaml"])
