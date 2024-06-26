#!/usr/bin/env python
import argparse
import copy
import subprocess

import numpy as np

from .utils import yaml_utils

parser = argparse.ArgumentParser()
parser.add_argument('evaluation_config', type=str, help='path to config via YAML file')
args = parser.parse_args()
opt = yaml_utils.load_yaml(args.evaluation_config, class_mode=True)

model_name = opt.general_config.model_name
model_file_path = opt.general_config.model_file_path
evaluation_path = opt.general_config.evaluation_path
template_config = opt.template_config
config_path = opt.general_config.robustness_test_config_path
opt_robustness_test = yaml_utils.load_yaml(config_path, class_mode=False)
opt_robustness_test['general_config']['load_model_path'] = model_file_path

for var in opt.evaluation_config:
    var_name = var.name
    print("VARNAME: ", var_name)
    filename_ori = f'{model_name}_{var_name}'
    if hasattr(var, "x") or hasattr(var, "noise"):
        x = np.linspace(var.x[0], var.x[1], var.x[2]).round(2)
    if hasattr(var, "y"):
        y = np.linspace(var.y[0], var.y[1], var.y[2]).round(2)
    if hasattr(var, "z"):
        z = np.linspace(var.z[0], var.z[1], var.z[2]).round(2)
    template_config.continuous = var.continuous if hasattr(var, "continuous") else False
    template_config.keep = var.keep if hasattr(var, "keep") else False
    template_config.seq_idx = var.seq_idx if hasattr(var, "seq_idx") else ""
    if var_name == 'rotation' or var_name == 'scale' or var_name == 'translation' or var_name == 'noise':
        for x_i in x:
            # write the yaml file every single loop
            template = copy.copy(template_config)
            if var_name == 'noise':
                template.noise = float(x_i)
                filename = filename_ori + f'_{x_i:.2f}'
            else:
                template.x = float(x_i)
                filename = filename_ori + f'x_{x_i:.2f}'
            filename = filename + '_cont' if template.continuous else filename
            filename = filename + '_keep' if template.keep else filename
            filename = filename + f'_seq_{template.seq_idx[0]}_{template.seq_idx[1]}' if template.seq_idx else filename
            opt_robustness_test['evaluation_config']['sets'][0]['original_test']['robustness_test'] = {
                var_name: template.__dict__}
            opt_robustness_test['evaluation_config']['sets'][0]['original_test']['extension_path'] = filename
            yaml_utils.write_yaml(opt_robustness_test, config_path, remote_struct=False)
            subprocess.call([evaluation_path, config_path, "--robustness_test"])
        if var_name == 'noise':
            continue
        for y_i in y:
            template = copy.copy(template_config)
            template.y = float(y_i)
            filename = filename_ori + f'y_{y_i:.2f}'
            filename = filename + '_cont' if template.continuous else filename
            filename = filename + '_keep' if template.keep else filename
            filename = filename + f'_seq_{template.seq_idx[0]}_{template.seq_idx[1]}' if template.seq_idx else filename
            opt_robustness_test['evaluation_config']['sets'][0]['original_test']['robustness_test'] = {
                var_name: template.__dict__}
            opt_robustness_test['evaluation_config']['sets'][0]['original_test']['extension_path'] = filename
            yaml_utils.write_yaml(opt_robustness_test, config_path, remote_struct=False)
            subprocess.call([evaluation_path, config_path, "--robustness_test"])
        for z_i in z:
            template = copy.copy(template_config)
            template.z = float(z_i)
            filename = filename_ori + f'z_{z_i:.2f}'
            filename = filename + '_cont' if template.continuous else filename
            filename = filename + '_keep' if template.keep else filename
            filename = filename + f'_seq_{template.seq_idx[0]}_{template.seq_idx[1]}' if template.seq_idx else filename
            opt_robustness_test['evaluation_config']['sets'][0]['original_test']['robustness_test'] = {
                var_name: template.__dict__}
            opt_robustness_test['evaluation_config']['sets'][0]['original_test']['extension_path'] = filename
            yaml_utils.write_yaml(opt_robustness_test, config_path, remote_struct=False)
            subprocess.call([evaluation_path, config_path, "--robustness_test"])
    elif var_name == 'flip':
        template = template_config
        template.x = var.cond_x
        template.y = False
        template.z = False
        filename = filename_ori + 'x'
        filename = filename + '_keep' if template.keep else filename
        filename = filename + f'_seq_{template.seq_idx[0]}_{template.seq_idx[1]}' if template.seq_idx else filename
        opt_robustness_test['evaluation_config']['sets'][0]['original_test']['robustness_test'] = {
            var_name: template.__dict__}
        opt_robustness_test['evaluation_config']['sets'][0]['original_test']['extension_path'] = filename
        yaml_utils.write_yaml(opt_robustness_test, config_path, remote_struct=False)
        subprocess.call([evaluation_path, config_path, "--robustness_test"])
        template.x = False
        template.y = var.cond_y
        template.z = False
        filename = filename_ori + 'y'
        filename = filename + '_keep' if template.keep else filename
        filename = filename + f'_seq_{template.seq_idx[0]}_{template.seq_idx[1]}' if template.seq_idx else filename
        opt_robustness_test['evaluation_config']['sets'][0]['original_test']['robustness_test'] = {
            var_name: template.__dict__}
        opt_robustness_test['evaluation_config']['sets'][0]['original_test']['extension_path'] = filename
        yaml_utils.write_yaml(opt_robustness_test, config_path, remote_struct=False)
        subprocess.call([evaluation_path, config_path, "--robustness_test"])
        template.x = False
        template.y = False
        template.z = var.cond_z
        filename = filename_ori + 'z'
        filename = filename + '_keep' if template.keep else filename
        filename = filename + f'_seq_{template.seq_idx[0]}_{template.seq_idx[1]}' if template.seq_idx else filename
        opt_robustness_test['evaluation_config']['sets'][0]['original_test']['robustness_test'] = {
            var_name: template.__dict__}
        opt_robustness_test['evaluation_config']['sets'][0]['original_test']['extension_path'] = filename
        yaml_utils.write_yaml(opt_robustness_test, config_path, remote_struct=False)
        subprocess.call([evaluation_path, config_path, "--robustness_test"])
    elif var_name == 'posinvers':
        template = template_config
        filename = filename_ori + '_keep' if template.keep else filename
        filename = filename + f'_seq_{template.seq_idx[0]}_{template.seq_idx[1]}' if template.seq_idx else filename
        opt_robustness_test['evaluation_config']['sets'][0]['original_test']['robustness_test'] = {
            var_name: template.__dict__}
        opt_robustness_test['evaluation_config']['sets'][0]['original_test']['extension_path'] = filename
        yaml_utils.write_yaml(opt_robustness_test, config_path, remote_struct=False)
        subprocess.call([evaluation_path, config_path, "--robustness_test"])
    # clean up config file
    opt_robustness_test['evaluation_config']['sets'][0]['original_test']['robustness_test'] = ""
    opt_robustness_test['evaluation_config']['sets'][0]['original_test']['extension_path'] = ""
    yaml_utils.write_yaml(opt_robustness_test, config_path, remote_struct=False)
