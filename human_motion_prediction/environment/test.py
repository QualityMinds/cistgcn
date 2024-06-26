import copy

import numpy as np
import torch
from tqdm.auto import tqdm

from . import adversarial_attacks as adv
from .. import losses


class Metrics:
    def __init__(self, w, reduce_axis, db):
        self.w = w
        self.r_ax = reduce_axis
        self.db = db
        self.out_losses = {}
        self.mpjpe_list = losses.LossOperator()
        self.pa_mpjpe_list = losses.LossOperator()
        self.n_mpjpe_list = losses.LossOperator()
        self.mve_list = losses.LossOperator()
        self.mae_list = losses.LossOperator()  # mean_angle_error
        self.w_mpjpe_list = losses.LossOperator()
        self.bone_length_list = losses.LossOperator()
        self.w_bone_length_list = losses.LossOperator()
        self.w_joints_list = losses.LossOperator()
        self.w_joints_temp_list = losses.LossOperator()

    # TODO: Recall and f1score is not implemented
    def get_values(self, key_to_process):
        if key_to_process == True:
            self.mpjpe_seq = self.mpjpe_list.get_all()
            self.pa_mpjpe_seq = self.pa_mpjpe_list.get_all()
            self.n_mpjpe_seq = self.n_mpjpe_list.get_all()
            self.mae_seq = self.mae_list.get_all()
            self.mve_seq = self.mve_list.get_all()
            self.w_mpjpe_seq = self.w_mpjpe_list.get_all()
            self.bone_length_seq = self.bone_length_list.get_all()
            self.w_bone_length_seq = self.w_bone_length_list.get_all()
            self.w_joints_seq = self.w_joints_list.get_all()
            self.w_joints_temp_seq = self.w_joints_temp_list.get_all()
        else:
            self.mpjpe_seq = self.mpjpe_list.mean(0)
            self.pa_mpjpe_seq = self.pa_mpjpe_list.mean(0)
            self.n_mpjpe_seq = self.n_mpjpe_list.mean(0)
            self.mae_seq = self.mae_list.mean(0)
            self.mve_seq = self.mve_list.mean(0)
            self.w_mpjpe_seq = self.w_mpjpe_list.mean(0)
            self.bone_length_seq = self.bone_length_list.mean(0)
            self.w_bone_length_seq = self.w_bone_length_list.mean(0)
            self.w_joints_seq = self.w_joints_list.mean(0)
            self.w_joints_temp_seq = self.w_joints_temp_list.mean(0)

    def get_average(self, seq_len):
        self.mpjpe_list.average(seq_len)
        self.pa_mpjpe_list.average(seq_len)
        self.n_mpjpe_list.average(seq_len)
        self.mae_list.average(seq_len)
        self.mve_list.average(seq_len)
        self.w_mpjpe_list.average(seq_len)
        self.bone_length_list.average(seq_len)
        self.w_bone_length_list.average(seq_len)
        self.w_joints_list.average(seq_len)
        self.w_joints_temp_list.average(seq_len)

    def compute(self, outputs, target, speeds=None):
        with torch.no_grad():
            speeds /= (speeds.max(2, keepdims=True)[0] + 1e-6)
            temporal_w = self.w.unsqueeze(0).unsqueeze(2).tile(outputs.shape[0], 1, outputs.shape[2])
            speed_w = speeds + temporal_w
            speed_temporal_w = speed_w / speed_w.max(0)[0]
            out_result = losses.mpjpe(outputs.detach(), target, reduce_axis=self.r_ax).cpu().data.numpy()
            self.mpjpe_list.append(out_result)
            out_result = losses.pa_mpjpe(outputs, target, reduce_axis=self.r_ax).cpu().data.numpy()
            self.pa_mpjpe_list.append(out_result)
            out_result = losses.n_mpjpe(outputs, target, reduce_axis=self.r_ax).cpu().data.numpy()
            self.n_mpjpe_list.append(out_result)
            out_result = losses.mean_angles_error(outputs, target, reduce_axis=self.r_ax).cpu().data.numpy()
            self.mae_list.append(out_result)
            out_result = losses.mean_velocity_error(outputs, target, reduce_axis=self.r_ax).cpu().data.numpy()
            self.mve_list.append(out_result)
            out_result = losses.weighted_mpjpe(outputs, target, w=temporal_w, reduce_axis=self.r_ax).cpu().data.numpy()
            self.w_mpjpe_list.append(out_result)
            out_result = losses.bone_length_error(outputs, target, skeleton_type=self.db,
                                                  reduce_axis=self.r_ax).cpu().data.numpy()
            self.bone_length_list.append(out_result)
            out_result = losses.weighted_bone_length_error(outputs, target, w=self.w, skeleton_type=self.db,
                                                           reduce_axis=self.r_ax).cpu().data.numpy()
            self.w_bone_length_list.append(out_result)
            out_result = losses.weighted_mpjpe(outputs, target, w=speeds, reduce_axis=self.r_ax).cpu().data.numpy()
            self.w_joints_list.append(out_result)
            out_result = losses.weighted_mpjpe(outputs, target, w=speed_temporal_w,
                                               reduce_axis=self.r_ax).cpu().data.numpy()
            self.w_joints_temp_list.append(out_result)
        torch.cuda.empty_cache()


def _predict(model, loader, inputs, target, inputs_vel=None, test_mode=True):
    name = model.__class__.__name__
    if name == 'CISTGCN' or name == 'STSGCN' or name == 'PGBIG' or name == 'siMLPe' or name == 'HRI' or name == 'MMA':
        if test_mode:
            outputs = model(inputs[:, :, loader.dataset.dim_used])
        else:
            outputs = model(inputs)
    elif name == 'MlpMixer':
        if test_mode:
            outputs = model(inputs_vel[:, :, loader.dataset.dim_used], inputs[:, -1, loader.dataset.dim_used])
        else:
            outputs = model(inputs_vel, inputs[:, -1])
    else:
        NotImplementedError

    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]
    if name == 'PGBIG':
        outputs_list = []
        for output in outputs:
            mygt = target.clone()
            mygt[:, :, loader.dataset.dim_used, :] = output.clone()
            if hasattr(loader.dataset, "dim_repeat_32"):
                mygt[:, :, loader.dataset.dim_repeat_32, :] = output[:, :, loader.dataset.dim_repeat_22, :].clone()
            # output = mygt
            outputs_list.append(mygt)
        outputs = outputs_list
    else:
        mygt = target.clone()
        mygt[:, :, loader.dataset.dim_used, :] = outputs.clone()
        if hasattr(loader.dataset, "dim_repeat_32"):
            mygt[:, :, loader.dataset.dim_repeat_32, :] = outputs[:, :, loader.dataset.dim_repeat_22, :].clone()
        outputs = mygt
    del mygt
    torch.cuda.empty_cache()
    return outputs


def _compute_metrics(outputs, inputs, target, speeds, model, Evaluator, interpretation_keywords=None, unnormalize=None):
    global data_std, data_mean, model_interpretation_outputs
    if not isinstance(outputs, list):
        outputs = [outputs]
    for output in outputs:
        if unnormalize is not None:
            output = output * data_std + data_mean
            target = target * data_std + data_mean
            inputs = inputs * data_std + data_mean

        Evaluator.compute(output, target, speeds)
        if interpretation_keywords:
            for key in interpretation_keywords:
                try:
                    main_struct = model
                    for k in key.split("."):
                        main_struct = getattr(main_struct, k)
                    if not key in model_interpretation_outputs:
                        model_interpretation_outputs[key] = [main_struct.squeeze().cpu().numpy()]
                    else:
                        model_interpretation_outputs[key].append(main_struct.squeeze().cpu().numpy())
                except:
                    print(f'{key} is not available on model')
    if model.__class__.__name__ == 'PGBIG':
        Evaluator.get_average(seq_len=4)
    return Evaluator, output, target, inputs


def _run_test(model, loader, loader_to_process, Evaluator, interpretation_keywords, db, kwargs):
    global data_std, data_mean, model_interpretation_outputs
    name = model.__class__.__name__
    model_interpretation_outputs = {}
    adv_metrics = {}
    original_inputs = {}
    if kwargs.get("idx_for_valid_seq") is not None:
        idx_for_valid_seq = int(kwargs.get("idx_for_valid_seq"))
    else:
        idx_for_valid_seq = 0
    get_all_samples = False
    if kwargs.get("get_all_samples") is not None:
        get_all_samples = kwargs.get("get_all_samples")
    if kwargs.get("unnormalize") is not None:
        unnormalize = kwargs.get("unnormalize")
        data_std = torch.from_numpy(unnormalize["data_std"]).cuda()
        data_mean = torch.from_numpy(unnormalize["data_mean"]).cuda()

    igrads_list, inputs_list, output_list, target_list, item_list, = [], [], [], [], []
    for i, data in tqdm(enumerate(loader_to_process), total=loader_to_process.__len__()):
        if name == "PGBIG":
            inputs = data["processed"].cuda()
            inputs_vel = torch.empty(inputs.shape).cuda()
        else:
            inputs = data["sample"].cuda()
            inputs_vel = data["sample_vel"].cuda() if name == "MlpMixer" else torch.empty(inputs.shape).cuda()
        original_inputs['inputs'] = inputs
        original_inputs['inputs_vel'] = inputs_vel
        target = data["target"].cuda()
        speeds = data["target_gvel"].cuda()[:, :, :, 0]
        # any string means that we just want to obtain gradients
        ############ Adversarial Attacks Modules ############
        if kwargs.get("adversarial_attacks") is not None:
            attack_conf = kwargs.get("adversarial_attacks").__dict__
            attack_name = list(attack_conf.keys())[0]
            attack_conf[attack_name].db = db
            Attacker = getattr(adv, attack_name)(**attack_conf[attack_name].__dict__)
            params = {"loader": loader, "target": target}
            adversarial_inputs = Attacker.apply(inputs, inputs_vel, model, _predict, params)  # Call FGSM Attack
            original_inputs['adv_inputs'] = torch.from_numpy(adversarial_inputs['adv_inputs']).cuda()
            if original_inputs['adv_inputs_vel'] is not None:
                original_inputs['adv_inputs_vel'] = torch.from_numpy(adversarial_inputs['adv_inputs_vel']).cuda()
            adv_metrics = Attacker._get_metrics(original_inputs['adv_inputs'], original_inputs['inputs'],
                                                original_inputs['adv_inputs_vel'], original_inputs['inputs_vel'],
                                                )
            del adversarial_inputs
            vars_to_analyze = "adv_"
        else:
            vars_to_analyze = ""
        ############ Adversarial Attacks Inference ############
        outputs = _predict(model, loader,
                           original_inputs[vars_to_analyze + 'inputs'],
                           target,
                           original_inputs[vars_to_analyze + 'inputs_vel'])
        Evaluator, output, target, original_inputs[vars_to_analyze + 'inputs'] = _compute_metrics(outputs,
                                                                                                  original_inputs[
                                                                                                      vars_to_analyze + 'inputs'],
                                                                                                  target, speeds,
                                                                                                  model,
                                                                                                  Evaluator,
                                                                                                  interpretation_keywords,
                                                                                                  kwargs.get(
                                                                                                      "unnormalize"))

        # TODO: include metrics such as... Recall f1score.
        if kwargs.get("unnormalize") is not None:
            output = output - data_mean / data_std
            target = target - data_mean / data_std
        if kwargs.get("adversarial_attacks") is None or not hasattr(kwargs.get("adversarial_attacks"), "NoAttack"):
            igrads = torch.zeros(original_inputs[vars_to_analyze + 'inputs'].shape[0]).cuda()
        else:
            igrads = original_inputs[vars_to_analyze + 'inputs'].grad  # check if works for MotionMixer

        igrads = igrads.detach().cpu().numpy()
        inputs = original_inputs[vars_to_analyze + 'inputs'].detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        items = data["item"].detach().cpu().numpy()
        if get_all_samples:  # This key return the last batch value or full data.
            igrads_list.extend(igrads)
            inputs_list.extend(original_inputs[vars_to_analyze + 'inputs'])
            output_list.extend(output)
            target_list.extend(target)
            item_list.extend(items)
        else:
            igrads_list = igrads
            inputs_list = inputs
            output_list = output
            target_list = target
            item_list = items
        if model.__class__.__name__ == 'PGBIG':
            Evaluator.get_average(seq_len=4)
        # if i == 0: break;  # Remove when debug finishes
    # Get values or Average.
    Evaluator.get_values(kwargs.get("compute_joint_error"))  # AFAIK: True or False
    igrads = np.array(igrads_list)
    inputs = np.array(inputs_list)
    output = np.array(output_list)
    target = np.array(target_list)
    items = np.array(item_list)

    # Recall f1score.
    if kwargs.get("unnormalize") is not None:
        output = output - data_mean / data_std
        target = target - data_mean / data_std
    return {"Evaluator": Evaluator,
            "inputs": inputs,
            "igrads": igrads,
            "outputs": output,
            "target": target,
            "items": items,
            "model_interpretation_outputs": model_interpretation_outputs,
            "adversarial_metrics": adv_metrics,
            }


def test(loader, model, idx=None, **kwargs):
    # Values (0,2) are specifc for sequences
    # Also we could get the error per joint if we use (0,1)
    if idx is not None:
        loader_to_process = copy.deepcopy(loader)
        loader_to_process.dataset.target = loader.dataset.target[idx]
        if loader_to_process.dataset.class_seq is not None:
            loader_to_process.dataset.class_seq = loader.dataset.class_seq[idx]
    else:
        loader_to_process = loader  # TODO: illegal memory access and bad coding. SOLVE IT SOON
    reduce_axis = (0, 2)
    interpretation_keywords = False
    if kwargs.get("get_interpretation") is not None:
        interpretation_keywords = kwargs.get("get_interpretation")
    if kwargs.get("output_n"):
        output_n = kwargs.get("output_n")
    else:
        output_n = 10
    db = "cmu"
    if kwargs.get("db"):
        db = kwargs.get("db")

    w = torch.arange(1, output_n + 1).cuda()
    w = w / w.max()
    if kwargs.get("compute_joint_error") == True:
        reduce_axis = None

    Evaluator = Metrics(w, reduce_axis, db)  # Metric object to do a better modularization
    model.eval()
    if kwargs.get("adversarial_attacks") is None:
        with torch.no_grad():
            results = _run_test(model, loader, loader_to_process, Evaluator, interpretation_keywords, db, kwargs)
    else:
        results = _run_test(model, loader, loader_to_process, Evaluator, interpretation_keywords, db, kwargs)
    torch.cuda.empty_cache()

    metrics = {"mpjpe": results["Evaluator"].mpjpe_list.mean(),
               "mpjpe_seq": results["Evaluator"].mpjpe_seq,
               "pa_mpjpe": results["Evaluator"].pa_mpjpe_list.mean(),
               "pa_mpjpe_seq": results["Evaluator"].pa_mpjpe_seq,
               "n_mpjpe": results["Evaluator"].n_mpjpe_list.mean(),
               "n_mpjpe_seq": results["Evaluator"].n_mpjpe_seq,
               "mae": results["Evaluator"].mae_list.mean(),
               "mae_seq": results["Evaluator"].mae_seq,
               "mve": results["Evaluator"].mve_list.mean(),
               "mve_seq": results["Evaluator"].mve_seq,
               "w_mpjpe": results["Evaluator"].w_mpjpe_seq.mean(),
               "w_mpjpe_seq": results["Evaluator"].w_mpjpe_seq,
               "bone_l": results["Evaluator"].bone_length_seq.mean(),
               "bone_l_seq": results["Evaluator"].bone_length_seq,
               "w_bone_l": results["Evaluator"].w_bone_length_seq.mean(),
               "w_bone_l_seq": results["Evaluator"].w_bone_length_seq,
               "w_joints": results["Evaluator"].w_joints_seq.mean(),
               "w_joints_seq": results["Evaluator"].w_joints_seq,
               "w_joints_t": results["Evaluator"].w_joints_temp_seq.mean(),
               "w_joints_t_seq": results["Evaluator"].w_joints_temp_seq,

               "inputs": results["inputs"],
               "igrads": results["igrads"],
               "pred": results["outputs"],
               "target": results["target"],
               "items": results["items"],

               "recall": None,
               "f1score": None,
               }

    if interpretation_keywords:
        metrics["interpretation"] = results["model_interpretation_outputs"]
    if kwargs.get("adversarial_attacks") is not None:
        metrics["adversarial_metrics"] = results["adversarial_metrics"]
    return metrics
