from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import yaml_utils


def visualize_robustness_test_result(var_name, var_value, model_name_list, folder_path_list, conditions):
    if var_name == 'flipx' or var_name == 'flipy' or var_name == 'flipz' or var_name == 'posinvers':
        var_sweep = [0, 1]
    else:
        var_sweep = np.linspace(var_value[0], var_value[1], var_value[2]).round(2)
    result_dict = {}
    max_result_list = []
    min_result_list = []
    for model_name, folder_path in zip(model_name_list, folder_path_list):
        print(model_name)
        i = 0
        result_array = np.full(len(var_sweep), -1.0)
        folder_name = Path(folder_path)
        if var_name == 'flipx' or var_name == 'flipy' or var_name == 'flipz' or var_name == 'posinvers':
            file_name_ori = Path(f'metrics_original_test_{model_name}_ori.xlsx')
            file_name = Path(f'metrics_original_test_{model_name}_{var_name}{conditions}.xlsx')
            if folder_name.joinpath(file_name).exists() and folder_name.joinpath(file_name_ori).exists():
                data_ori = pd.read_excel(folder_name.joinpath(file_name_ori), 1)
                data = pd.read_excel(folder_name.joinpath(file_name), 1)
            else:
                print(file_name)
                continue
            if model_name == 'short-CISTGCN-16' or model_name == 'short-CISTGCN-32':
                result_array[0] = float(data_ori[['mean']].iloc[10])
                result_array[1] = float(data[['mean']].iloc[10])
            else:
                result_array[0] = float(data_ori[['mean']].iloc[25])
                result_array[1] = float(data[['mean']].iloc[25])
            result_dict[model_name] = result_array
            i += 1
        else:
            for var in var_sweep:
                if var_name == 'translationx' or var_name == 'translationy' or var_name == 'translationz' or var_name == 'translationall':
                    file_name = Path(f'metrics_original_test_{model_name}_{var_name}_{var:.2f}{conditions}.xlsx')
                elif var_name == 'rotationx' or var_name == 'rotationy' or var_name == 'rotationz':
                    file_name = Path(f'metrics_original_test_{model_name}_{var_name}_{round(var):.2f}{conditions}.xlsx')
                else:
                    file_name = Path(f'metrics_original_test_{model_name}_{var_name}_{var:.2f}{conditions}.xlsx')
                if folder_name.joinpath(file_name).exists():
                    data = pd.read_excel(folder_name.joinpath(file_name), 1)
                else:
                    print("Missing file: ", file_name)
                    continue
                if model_name == 'short-CISTGCN-16' or model_name == 'short-CISTGCN-32':
                    result_array[i] = float(data[['mean']].iloc[10])
                else:
                    result_array[i] = float(data[['mean']].iloc[25])
                result_dict[model_name] = result_array
                i += 1
        result_array[result_array > np.median(result_array) * 100] = -1.0
        result_dict[model_name] = result_array
        if np.all((result_array < 0)):
            max_value = 0
            min_value = float('inf')
        else:
            max_value = np.max(result_array)
            min_value = np.min(result_array[result_array >= 0])
        max_result_list.append(max_value)
        min_result_list.append(min_value)
    if conditions:
        var_name = f'{var_name}_{conditions}'
    plot_graph(var_name, var_sweep, result_dict, min_result_list, max_result_list, average_mpjpe=True)


def visualize_sequence_error(var_name, var_value, model_name_list, folder_path_list, conditions):
    var_sweep = np.linspace(40, 1000, 25)

    result_dict = {}
    max_result_list = []
    min_result_list = []

    for model_name, folder_path in zip(model_name_list, folder_path_list):
        print(model_name)
        result_array = np.full(len(var_sweep), -1.0)
        folder_name = Path(folder_path)
        file_name = Path(f'metrics_original_test_{model_name}_{var_name}_{var_value:.2f}{conditions}.xlsx')
        if folder_name.joinpath(file_name).exists():
            data = pd.read_excel(folder_name.joinpath(file_name), 1)
        else:
            print("Missing file: ", file_name)
            continue
        result_array[:] = data[['mean']].iloc[:25].to_numpy()[:, 0]
        result_array[result_array > np.median(result_array) * 100] = -1.0
        result_dict[model_name] = result_array
        if np.all((result_array < 0)):
            max_value = 0
            min_value = float('inf')
        else:
            max_value = np.max(result_array)
            min_value = np.min(result_array[result_array >= 0])
        max_result_list.append(max_value)
        min_result_list.append(min_value)
    var_name = f'{var_name}_{var_value:.2f}{conditions}'
    plot_graph(var_name, var_sweep, result_dict, min_result_list, max_result_list)


def plot_graph(var_name, var_sweep, result_dict, min_result_list, max_result_list, average_mpjpe=False):
    fig, ax = plt.subplots()
    line_config_list = ['b-o', 'r--x', 'y:+', 'c--*', 'g:v', 'k--x', 'w--*']
    ax.set_title(f'{var_name} Robustness Test', fontsize=14)
    ax.set_xlabel(f'{var_name}', fontsize=12)
    ax.set_ylabel('average mpjpe', fontsize=12)
    l_list = []
    idx = 0
    for key in result_dict.keys():
        l_list.append(ax.plot(var_sweep, result_dict[key], line_config_list[idx])[0])
        idx += 1
    plt.xlim([min(var_sweep), max(var_sweep)])
    max_result_list.sort()
    if len(max_result_list) >= 2:
        if np.max(max_result_list) < 2 * max_result_list[-2]:
            plt.ylim([np.floor(np.min(min_result_list) / 10) * 10, np.max(max_result_list)])
        else:
            plt.ylim([np.floor(np.min(min_result_list) / 10) * 10, max_result_list[-2] + max_result_list[-2] * 0.05])
    fig.legend(tuple(l_list), tuple(result_dict.keys()), loc='upper right')
    plt.tight_layout()
    fig_path = Path('logdir').joinpath(f'{var_name}.png')
    print("Figure path: ", fig_path)
    fig.savefig(fig_path)


def main(opt):
    if opt.visualization_config.mode == 'average_error':
        for robustness_test, robustness_test_value in zip(opt.visualization_config.robustness_test,
                                                          opt.visualization_config.robustness_test_value):
            visualize_robustness_test_result(robustness_test, robustness_test_value,
                                             opt.visualization_config.model_names,
                                             opt.visualization_config.robustness_test_folder_path,
                                             opt.visualization_config.conditions)
    else:
        for robustness_test, robustness_test_value in zip(opt.visualization_config.robustness_test,
                                                          opt.visualization_config.robustness_test_value):
            visualize_sequence_error(robustness_test, robustness_test_value, opt.visualization_config.model_names,
                                     opt.visualization_config.robustness_test_folder_path,
                                     opt.visualization_config.conditions)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('visualization_config', type=str, help='path to config via YAML file')
    args = parser.parse_args()
    opt = yaml_utils.load_yaml(args.visualization_config, class_mode=True)
    main(opt)
