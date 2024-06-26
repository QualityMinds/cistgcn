import numpy as np
import pandas as pd

from ..utils import body_utils

GROUP_LENGTH = 1000000  # set nr of rows to slice df


def adding_stats(df_new, select_keys, actions_keys):
    total_indexes = len(df_new.index)
    df_new["mean"] = df_new[select_keys].mean(1).tolist()
    df_new["std"] = df_new[select_keys].std(1).tolist()
    df_new["min"] = df_new[select_keys][df_new[select_keys] != 0].min(1).tolist()
    df_new["max"] = df_new[select_keys].max(1).tolist()
    df_new["quantile .50"] = df_new[select_keys].quantile(q=0.50, axis=1).tolist()
    df_new["quantile .75"] = df_new[select_keys].quantile(q=0.75, axis=1).tolist()

    if total_indexes > 2:
        df_new.loc["mean"] = df_new[select_keys].iloc[:total_indexes].mean(0)
        df_new.loc["std"] = df_new[select_keys].iloc[:total_indexes].std(0)
        df_new.loc["min"] = df_new[select_keys][df_new[select_keys] != 0].iloc[:total_indexes].min(0)
        df_new.loc["max"] = df_new[select_keys].iloc[:total_indexes].max(0)
        df_new.loc["quantile .50"] = df_new[select_keys].iloc[:total_indexes].quantile(q=0.50, axis=0)
        df_new.loc["quantile .75"] = df_new[select_keys].iloc[:total_indexes].quantile(q=0.75, axis=0)
        df_new.iloc[total_indexes, len(actions_keys)] = df_new[select_keys].mean().mean()
    return df_new


# convert this to modular way in future releases.
def record_sheet(metrics, file_name, compute="metrics", apply_sort=True, skeleton_type="cmu"):
    assert compute == "metrics" or compute == "samples"

    bones, joint_names = body_utils.get_reduced_skeleton(skeleton_type)

    df = pd.DataFrame(metrics)
    actions_keys = list(metrics.keys())
    valid_keys = list(metrics[actions_keys[0]])
    if "recall" in valid_keys: valid_keys.remove("recall")
    if "f1score" in valid_keys: valid_keys.remove("f1score")
    if "pred" in valid_keys: valid_keys.remove("pred")
    if "target" in valid_keys: valid_keys.remove("target")
    if "igrads" in valid_keys: valid_keys.remove("igrads")
    if "inputs" in valid_keys: valid_keys.remove("inputs")
    if "adversarial_metrics" in valid_keys: valid_keys.remove("adversarial_metrics")
    if "items" in valid_keys: valid_keys.remove("items")
    process_keys = [v for v in valid_keys if "seq" not in v]
    df_new = df.loc[process_keys]
    process_keys.append("samples")

    new_metric = df.loc[[v for v in valid_keys if "seq" in v]]
    analysis_keys = [s for s in new_metric[actions_keys[0]].keys()]

    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    if compute == "metrics":
        df_new.loc["samples"] = [metrics[a]['mpjpe_seq'].shape[0] for a in actions_keys]
        df_new["mean"] = [df_new.iloc[v][:len(actions_keys)].mean() for v in range(len(process_keys))]
        df_new["std"] = [df_new.iloc[v][:len(actions_keys)].std() for v in range(len(process_keys))]
        df_new["min"] = [df_new.iloc[v][:len(actions_keys)].min() for v in range(len(process_keys))]
        df_new["max"] = [df_new.iloc[v][:len(actions_keys)].max() for v in range(len(process_keys))]
        df_new["quantile .50"] = [df_new.iloc[v][:len(actions_keys)].quantile(q=0.50) for v in range(len(process_keys))]
        df_new["quantile .75"] = [df_new.iloc[v][:len(actions_keys)].quantile(q=0.75) for v in range(len(process_keys))]
        df_new.to_excel(writer, sheet_name='Global-Actions')

        for name, reduce_dims in zip(["Sequence-Action", "Joint-Action"], [(0, 2), (0, 1)]):
            for key in analysis_keys:
                df_new = {}
                for act in actions_keys:
                    df_new[act] = new_metric[act].loc[key].mean(reduce_dims).tolist()
                df_new = pd.DataFrame(df_new)
                if "Sequence" in name:
                    df_new.index = [str(i) + " ms" for i in (40 * np.arange(1, len(df_new.index) + 1)).tolist()]
                elif "Joint" in name:
                    df_new.index = [str(i) + "_" + joint_names[i] for i in range(len(df_new.index))]
                adding_stats(df_new, actions_keys, actions_keys)
                df_new.to_excel(writer, sheet_name=f'{name}-{key.replace("_length", "_l").replace("_seq", "")}'[:31])

        # Metric x Action x Samples x Sequence (N or N-1) x Joint
        name = "Joint-Sequence"
        for key in analysis_keys:
            df_new = {}
            values = []
            for act in actions_keys:
                values.append(new_metric[act].loc[key].mean(0).tolist())  # mean over Samples.
            joint_sequence = np.mean(values, axis=0)
            for row in range(joint_sequence.shape[0]):
                df_new[row] = joint_sequence[row].tolist()
            df_new = pd.DataFrame(df_new)
            df_new.columns = [str(i) + " ms" for i in (40 * np.arange(1, len(df_new.columns) + 1)).tolist()]
            df_new.index = [str(i) + "_" + joint_names[i] for i in range(len(df_new.index))]
            adding_stats(df_new, list(df_new.columns), list(df_new.columns))
            df_new.to_excel(writer, sheet_name=f'{name}-{key.replace("_length", "_l").replace("_seq", "")}'[:31])

        name = "J-S"  # More raw data, very flat output => Joint-Sequence-Action-metric
        for key in analysis_keys:
            for act in actions_keys:
                df_new = {}
                joint_sequence = new_metric[act].loc[key].mean(0)
                for row in range(joint_sequence.shape[0]):
                    df_new[row] = joint_sequence[row].tolist()
                df_new = pd.DataFrame(df_new)
                df_new.columns = [str(i) + " ms" for i in (40 * np.arange(1, len(df_new.columns) + 1)).tolist()]
                df_new.index = [str(i) + "_" + joint_names[i] for i in range(len(df_new.index))]
                adding_stats(df_new, list(df_new.columns), list(df_new.columns))
                sheet_name = f'{name}-{act.replace("_signal", "_s").replace("/", ".")[:16]}-{key.replace("_length", "_l").replace("_seq", "")}'
                df_new.to_excel(writer, sheet_name=sheet_name[:31])
    elif compute == "samples":
        # More raw data, very flat output => Joint-Sequence-Action-metric
        for key in analysis_keys:
            for act in actions_keys:
                for name, reduce_dims in zip(["S-A", "J-A"], [(2), (1)]):
                    print(name, key, act)
                    df_new = {}
                    joint_sequence = new_metric[act].loc[key].mean(reduce_dims)
                    for row in range(joint_sequence.shape[0]):
                        df_new[row] = joint_sequence[row].tolist()
                    df_new = pd.DataFrame(df_new)
                    df_new.columns = [str(i) for i in range(len(df_new.columns))]  # samples
                    if "S" in name:
                        df_new.index = [str(i) + " ms" for i in (40 * np.arange(1, len(df_new.index) + 1)).tolist()]
                    elif "J" in name:
                        df_new.index = [str(i) + "_" + joint_names[i] for i in range(len(df_new.index))]
                    # Sort samples
                    if apply_sort:
                        idx = np.argsort(df_new.mean(0)).values[::-1]
                        df_new = df_new.iloc[:, idx]
                    adding_stats(df_new, list(df_new.columns), actions_keys)
                    key1 = key.replace("_length", "_l").replace("_seq", "")
                    sheet_name = f'{name}-{act.replace("_signal", "_s").replace("/", ".")[:16]}-{key1}'
                    df_new = df_new.transpose()
                    if df_new.shape[0] >= GROUP_LENGTH:
                        for i, rows in enumerate(range(0, df_new.shape[1], GROUP_LENGTH)):
                            print()
                            print(df_new.shape, df_new.iloc[rows: rows + GROUP_LENGTH].shape)
                            df_new.iloc[rows: rows + GROUP_LENGTH].to_excel(writer, sheet_name=sheet_name + f'_{i}'[:31])
                    else:
                        df_new.to_excel(writer, sheet_name=sheet_name[:31])
    writer.close()

    # Added specially for adversarial attacks if present.
    if "adversarial_metrics" in metrics[actions_keys[0]].keys():
        writer = pd.ExcelWriter(file_name.replace('.xlsx', '_adv_difference.xlsx'), engine='xlsxwriter')
        df = pd.DataFrame(metrics)
        actions_keys = list(metrics.keys())
        valid_keys = list(metrics[actions_keys[0]]["adversarial_metrics"].keys())
        for key_name in valid_keys:
            if '_sample' in key_name:
                for k in metrics.keys():
                    val_to_process = df[k]["adversarial_metrics"][key_name]
                    args = np.argsort(val_to_process)[::-1]
                    df_new = {}
                    val_to_process = val_to_process[args]
                    df_new[k] = val_to_process.tolist()
                    df_new = pd.DataFrame(df_new)
                    df_new.index = args
                    df_new.to_excel(writer, sheet_name=f'{key_name.replace("_sample", "")}-{k}'[:31])
            else:
                df_new = {}
                for k in metrics.keys():
                    val_to_process = df[k]["adversarial_metrics"][key_name]
                    if isinstance(val_to_process, np.ndarray):
                        if val_to_process.shape:
                            df_new[k] = val_to_process.tolist()
                        else:
                            df_new[k] = [val_to_process]
                    elif isinstance(val_to_process, list):
                        df_new[k] = val_to_process
                    else:
                        df_new[k] = [val_to_process]
                df_new = pd.DataFrame.from_dict(df_new, orient='index').transpose()
                if 'mpjpe' == key_name:
                    df_new["mean"] = [df_new.iloc[0][:len(actions_keys)].mean()]
                    df_new["std"] = [df_new.iloc[0][:len(actions_keys)].std()]
                    df_new["min"] = [df_new.iloc[0][:len(actions_keys)].min()]
                    df_new["max"] = [df_new.iloc[0][:len(actions_keys)].max()]
                    df_new["quantile .50"] = [df_new.iloc[0][:len(actions_keys)].quantile(q=0.50)]
                    df_new["quantile .75"] = [df_new.iloc[0][:len(actions_keys)].quantile(q=0.75)]
                    df_new.index = ["mpjpe"]
                    support = [metrics[a]['mpjpe_seq'].shape[0] for a in actions_keys]
                    support.extend([0, 0, 0, 0, 0, 0])
                    df_new.loc["samples"] = support
                elif "temporal" in key_name:
                    df_new.index = [str(i) + " ms" for i in (40 * np.arange(1, len(df_new.index) + 1)).tolist()]
                elif "spatial" in key_name:
                    df_new.index = [str(i) + "_" + joint_names[i] for i in range(len(df_new.index))]
                if not "metric_type" in key_name or "sample" in key_name:
                    adding_stats(df_new, list(df_new.columns), list(df_new.columns))
                df_new.to_excel(writer, sheet_name=key_name[:31])
        writer.close()


if __name__ == "__main__":
    metrics = np.load("/home/eme/Projects/human-motion-prediction/test.npy", allow_pickle=True).all()
    metrics
