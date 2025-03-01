import numpy as np
import torch
import time
import torch.utils
import torch.utils.data

from typing import List, Tuple, Dict, Union
from scipy.stats import norm
from sklearn.metrics import auc, roc_curve

from info import METHODS_ALIASES
from dataset import get_batch
from extra_audit import tabular_quantile_gdbt
from signals import get_signal_name, get_signals
from utils import select_alg, generate_mask_matrix, save_audit_results

def metric_based_auditing(
    alg: str,
    membership: np.array,
    target_signal: np.array,
    in_signals: np.array=None,
    out_signals: np.array=None,
    population_signals: np.array=None,
):
    """Metric based data auditing algorithm.

    Args:
        alg (str): Auditing algorithm name.
        membership (np.array[np.int]): An array with dimension [N], where M represents the number of audit data/samples/users for membership.
        target_signal (np.array[np.float]): An array with dimension [M], which contains the audit data's membership signal in the target model.
        in_signals (np.array[np.float]): An array with dimension [M,N], which contains the audit data's membership signal in its member reference model. 
        out_signals (np.array[np.float]): An array with dimension [M,N], which contains the audit data's membership signal in its non-member reference model. 
        population_signals (np.array[np.float]): An array with dimension [M], which contains the population data's membership signal in the target model.
    
    Returns:
        fpr_list (np.array[np.float]): False positive rate of auditing results based on different decision thresholds.
        tpr_list (np.array[np.float]): True positive rate of auditing results based on different decision thresholds. 
        roc_auc (float): Area under TPR-FPR Curve.
    """

    start_time = time.time()
    alg_prefix = alg.split('-')[0]
    if alg_prefix == "reference_in_out":
        assert in_signals is not None and out_signals is not None, "in_signals and out_signals are necessary for reference_in_outt~ audit."
    elif alg_prefix == "reference_out":
        assert out_signals is not None, "out_signals are necessary for reference_out~ audit."
    elif alg_prefix == "population":
        assert population_signals is not None, "population_signals are necessary for population~ audit."
    elif alg_prefix == "target":
        pass
    else:
        raise NotImplementedError(f"{alg_prefix}~ audit is not implemented")
        
    if alg == "reference_in_out-pdf":
        mean_in = np.median(in_signals, 1)
        mean_out = np.median(out_signals, 1)
        std_in = np.std(in_signals, 1)
        std_out = np.std(out_signals, 1)
        pr_in = norm.logpdf(target_signal, mean_in, std_in + 1e-30)
        pr_out = norm.logpdf(target_signal, mean_out, std_out + 1e-30)
        score = pr_out - pr_in
    elif alg == "reference_out-pdf":
        mean_out = np.median(out_signals, 1)
        std_out = np.std(out_signals, 1)
        pr_out = norm.logpdf(target_signal, mean_out, std_out + 1e-30)
        score = pr_out
    elif alg == "reference_in_out-pdf_fix_var":
        mean_in = np.median(in_signals, 1)
        mean_out = np.median(out_signals, 1)
        std_in = np.std(in_signals)
        std_out = np.std(out_signals)
        pr_in = norm.logpdf(target_signal, mean_in, std_in + 1e-30)
        pr_out = norm.logpdf(target_signal, mean_out, std_out + 1e-30)
        score = pr_out - pr_in
    elif alg == "reference_out-pdf_fix_var":
        mean_out = np.median(out_signals, 1)
        std_out = np.std(out_signals)
        pr_out = norm.logpdf(target_signal, mean_out, std_out + 1e-30)
        score = pr_out
    elif alg == "reference_out-percentile": # Attack R of "Enhanced Membership Inference Attacks against Machine Learning Models"
        mean_out = np.median(out_signals, 1)
        std_out = np.std(out_signals, 1)
        pr_out = norm.cdf(-target_signal, -mean_out, std_out + 1e-30) # we use negative rescale logit as signal
        score = -pr_out
    elif alg == "population-percentile": # Attack P of "Enhanced Membership Inference Attacks against Machine Learning Models"
        mean_out = np.median(population_signals)
        std_out = np.std(population_signals)
        pr_out = norm.cdf(-target_signal, -mean_out, std_out + 1e-30)
        score = -pr_out
    elif alg == "target":
        score = target_signal
    elif alg == "reference_in_out-non":
        mean_in = np.median(in_signals, 1)
        mean_out = np.median(out_signals, 1)
        score = target_signal - (mean_in+mean_out)/2
    elif alg == "reference_out-non":
        mean_out = np.median(out_signals, 1)
        score = target_signal - mean_out
    else:
        raise ValueError("Unknown algorithm")
    
    prediction, answers = score, membership
    fpr_list, tpr_list, _ = roc_curve(answers, -prediction)
    acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)
    roc_auc = auc(fpr_list, tpr_list)
    tpr_at_low_fpr = tpr_list[np.where(fpr_list < 0.001)[0][-1]]

    print(
        f"{alg} AUC: %.4f, Accuracy: %.4f, TPR@0.1%%FPR: %.4f, Time cost: %.4f"
        % (roc_auc, acc, tpr_at_low_fpr, time.time()-start_time)
    )
    return fpr_list, tpr_list, roc_auc

def rmia(
    alg: str,
    membership: np.array,
    target_signal: np.array,
    population_signals: np.array,
    in_signals: np.array,
    out_signals: np.array,
    population_in_signals: np.array,
    population_out_signals: np.array,
    scale=None,
    gamma: int=2,
):
    assert alg in ["reference_in_out-population_rmia", "reference_out-population_rmia"], "Rmia does not support algorithm {alg}."
    population_mean_signals = (np.mean(population_in_signals, 1) + np.mean(population_out_signals, 1))/2
    print(population_mean_signals.shape)
    print(population_signals.shape)
    population_ratio = population_signals / population_mean_signals
    target_mean_out_signals = np.mean(out_signals, 1)
    if alg == "reference_in_out-population_rmia":
        scales = [-1]
        target_mean_in_signals = np.mean(in_signals, 1)
        target_mean_signals = [(target_mean_out_signals + target_mean_in_signals) / 2]
    else:
        if scale is None:
            scales = np.linspace(0,1,num=11)
        else:
            scales = [scale]
        target_mean_signals = []
        for s in scales:
            target_mean_signals.append(
                ( (1+s) * target_mean_out_signals + 1 - s) / 2
            )

    results = []
    roc_aucs = []
    for target_mean_signal in target_mean_signals:
        target_ratio = target_signal / target_mean_signal
        print(target_ratio.shape, population_ratio.shape)
        rmia_scores = np.zeros_like(target_ratio)
        for i in range(len(target_ratio)):
            rmia_scores[i] = np.mean((target_ratio[i] / population_ratio) > gamma)
        
        fpr_list, tpr_list, roc_auc = metric_based_auditing(
            alg="target", 
            membership=membership, 
            target_signal=-rmia_scores, 
            in_signals=None, 
            out_signals=None,
            population_signals=None,
        )
        results.append((fpr_list, tpr_list, roc_auc))
        roc_aucs.append(roc_auc)
    
    best_index = np.argmax(np.array(roc_aucs))
    print(f"Best index: {best_index}, best scale: {scales[best_index]}")
    print(f"Best AUC: {roc_aucs[best_index]}")
    fpr_list, tpr_list, roc_auc = results[best_index]
    best_scale = scales[best_index]
    return fpr_list, tpr_list, roc_auc, best_scale
            
def prepare_signals(
    target_model: torch.nn.Module,
    reference_model_list: List[torch.nn.Module],
    signal_name: str,
    configs: Dict,
    target_dataset: torch.utils.data.Dataset,
    population_dataset: torch.utils.data.Dataset,
    target_data_index: np.array,
    reference_train_data_indexes: List[np.array],
    get_population_signal: bool=False, 
):
    """Prepare signals for metric based auditing benchmark.

    Args:
        target_model (torch.nn.Module): The target model to be audited. We support pytorch models as input.
        reference_model_list (List[torch.nn.Module]): The reference model list to help compute in signals and out signals.
        signal_name (str): Membership signal to reflect whether the target model used the audit data.
        configs (Dict): Target model name, reference model name, device, and audit batch size.
        target_dataset (torch.utils.data.Dataset): The audit/target data. # Dict for text data and Tuple (eg. (images, labels)) for other data.
        population_dataset (torch.utils.data.Dataset): The population(shadow) data, i.e. the entire dataset except for the part of the target data.
        target_data_index (one-dimensional np.array(int)): The target data's index in the whole dataset.
        reference_train_data_indexes (two-dimensional np.array(int)): The training data indexes of reference models in the whole dataset.
        get_population_signal (bool): Calculate population data's signals or not.
    
    Returns:
        target_signals (one-dimensional np.array[np.float]): Target data's signals in the target model.
        in_signals (two-dimensional np.array[np.float]): Target data's signals in its member reference models.
        out_signals (two-dimensional np.array[np.float]): Target data's signals in its non-member reference models.
        population_signals (one-dimensional np.array[np.float]): Population data's signals in the target model.

    """

    target_model_name, reference_model_name = configs["audit"]["target_model"], configs["audit"]["reference_model"]
    device, batch_size = configs["audit"]["device"], configs["audit"]["audit_batch_size"]
    target_signals, population_signals = get_signals(
        [target_model], 
        target_model_name, 
        signal_name, 
        device, 
        batch_size, 
        target_dataset, 
        population_dataset, 
        get_population_signal,
    )
    target_signals = target_signals[0] if len(target_signals) > 0 else None
    population_signals = population_signals[0] if population_signals is not None and len(population_signals) > 0 else None
    reference_signals, _ = get_signals(
        reference_model_list, 
        reference_model_name, 
        signal_name, 
        device, 
        batch_size, 
        target_dataset, 
        population_dataset, 
        get_population_signal=False
    )

    # TODO check in_size and out_size's value
    in_signals = []
    out_signals = []
    mask_matrix = generate_mask_matrix(reference_train_data_indexes, target_data_index)
    for i in range(len(target_data_index)):
        in_signals.append(
            reference_signals[mask_matrix[:, i], i]
        )
        out_signals.append(
            reference_signals[~mask_matrix[:, i], i]
        )
    in_size = min(min(map(len, in_signals)), configs["train"]["num_in_models"])
    out_size = min(min(map(len, out_signals)), configs["train"]["num_out_models"])
    in_signals = np.array([x[:in_size] for x in in_signals]).astype("float32")
    out_signals = np.array([x[:out_size] for x in out_signals]).astype("float32")
    return target_signals, in_signals, out_signals, population_signals

def metric_ref_based_benchmark(
    alg_list: List[str],
    target_model: torch.nn.Module,
    reference_model_list: List[torch.nn.Module],
    population_model_list: List[torch.nn.Module],
    target_dataset: torch.utils.data.Dataset,
    population_dataset: torch.utils.data.Dataset,
    target_data_index: np.array,
    population_data_index: np.array,
    reference_train_data_indexes: List[np.array],
    population_train_data_indexes: List[np.array],
    membership: np.array,
    configs: Dict,
    report_dir: str,
):
    """Metric based auditing benchmark.

    Args:
        alg_list (List[str]): The data auditing algorithms list.
        target_model (torch.nn.Module): The target model to be audited. We support pytorch models as input.
        reference_model_list (List[torch.nn.Module]): The reference model list to help compute in_signals and out_signals.
        population_model_list (List[torch.nn.Module]): The population model list.
        target_dataset (torch.utils.data.Dataset): The audit/target data. # Dict for text data and Tuple (eg. (images, labels)) for other data.
        population_dataset (torch.utils.data.Dataset): The population(shadow) data, i.e. the entire dataset except for the part of the target data.
        target_data_indexes (one-dimensional np.array(int)): The target data's index in the whole dataset.
        population_data_indexes (one-dimensional np.array(int)): The population data's index in the whole dataset.
        reference_train_data_indexes (two-dimensional np.array(int)): The training data indexes of reference models in the whole dataset.
        population_train_data_indexes (two-dimensional np.array(int)): The training data indexes of population models in the whole dataset.
        membership (np.array[np.int]): An array with dimension [N], where M represents the number of audit data/samples/users for membership.
        configs (Dict): Target model name, reference model name, device, and audit batch size.
        report_dir (str): The path to save audit results including tpr, fpr, and other metrics.

    Returns:
        all_fpr_list (List[np.array[np.float]]): False positive rate based on different decision thresholds of different auditing algorithms.
        all_tpr_list (List[np.array[np.float]]): True positive rate based on different decision thresholds of different auditing algorithms.
        all_auc_list (List[float]): Area under TPR-FPR Curve of different auditing algorithms.
        membership_list (List[np.array[np.int]]): Auditing data's membership for different auditing algorithms.
    """
    start_time = time.time()
    print(f"Metric based auditing benchmark starts.")
    print(100 * "#")
    all_fpr_list = []
    all_tpr_list = []
    all_auc_list = []
    membership_list = []
    all_alg_names = []

    # get signal names for metric-based audit (e.g. loss, entropy, ...)
    signal_names = get_signal_name(alg_list)

    for signal_name in signal_names:
        signal_start_time = time.time()
        print(f"Signal name: {signal_name}")

        # get algorithms which need the given signal
        algs = select_alg(alg_list, signal_name)
        print(algs)

        # if "population" in algs or "reference_in_out-population_rmia" in algs or "reference_out-population_rmia" in algs or ("reference_out-quantile" in algs and configs["data"]["dataset_type"] == "tabular"):

        print(algs)
        if len(set(["population-percentile", "reference_in_out-population_rmia", "reference_out-population_rmia", "query-quantile"]) & set(algs)) > 0:
            get_population_signal = True  
        else:
            get_population_signal = False

        # prepare signal for metric based attack
        #target_signal, in_signals, out_signals, population_signals = prepare_signals(
        tardata_tarmodel_sig, tardata_in_sig, tardata_out_sig, popdata_tarmodel_sig = prepare_signals(
            target_model=target_model,
            reference_model_list=reference_model_list,
            signal_name=signal_name,
            configs=configs,
            target_dataset=target_dataset,
            population_dataset=population_dataset,
            target_data_index=target_data_index,
            reference_train_data_indexes=reference_train_data_indexes,
            get_population_signal=get_population_signal, 
        ) 
        print(f"Prepare the signals costs {time.time() - signal_start_time:.5f} seconds")
        
        for alg in algs:
            if alg is not None:
                print(f"Runing algorithm {METHODS_ALIASES[alg+'+'+signal_name]} with signal {signal_name}.")
                if alg == "query-quantile" and configs["data"]["dataset_type"] == "tabular":
                    #assert configs["data"]["dataset_type"] == "tabular", "dataset type error!"
                    target_batch = get_batch(target_dataset, model_name="none")
                    pop_batch = get_batch(population_dataset, model_name="none")
                    target_data, _ = target_batch
                    population_data, _ = pop_batch
                    fpr_list, tpr_list, roc_auc = tabular_quantile_gdbt(
                        quantile_model_name = configs["audit"]["reference_model"], 
                        membership=membership, 
                        target_signal=tardata_tarmodel_sig,
                        target_data=target_data, 
                        population_signal=popdata_tarmodel_sig,
                        population_data=population_data, 
                        seed=configs["random_seed"],
                        device=configs["audit"]["device"],
                    )

                elif alg in ["reference_in_out-population_rmia", "reference_out-population_rmia"]:
                    _, popdata_in_sig, popdata_out_sig, _ = prepare_signals(
                        target_model=None,
                        reference_model_list=population_model_list,
                        signal_name=signal_name,
                        configs=configs,
                        target_dataset=population_dataset,
                        population_dataset=target_dataset,
                        target_data_index=population_data_index,
                        reference_train_data_indexes=population_train_data_indexes,
                        get_population_signal=False, 
                    )

                    if configs["audit"]["dataset"] in ["cifar10", "cinic10"]:
                        best_scale = 0.3
                    elif configs["audit"]["dataset"] == "cifar100":
                        best_scale = 0.6
                    elif configs["audit"]["dataset"] == "imagenet":
                        best_scale = 1
                    elif configs["audit"]["dataset"] == "purchase100":
                        best_scale = 0.2
                    else:
                        best_scale = 0.4

                    fpr_list, tpr_list, roc_auc, _ = rmia(
                        alg=alg, 
                        membership=membership, 
                        target_signal=tardata_tarmodel_sig,
                        population_signals=popdata_tarmodel_sig,
                        in_signals=tardata_in_sig,
                        out_signals=tardata_out_sig, 
                        population_in_signals=popdata_in_sig,
                        population_out_signals=popdata_out_sig,
                        scale=best_scale,
                        gamma=2,
                    )

                else:
                    fpr_list, tpr_list, roc_auc = metric_based_auditing(
                        alg=alg, 
                        membership=membership, 
                        target_signal=tardata_tarmodel_sig,
                        in_signals=tardata_in_sig, 
                        out_signals=tardata_out_sig, 
                        population_signals=popdata_tarmodel_sig,
                    )
                
                # Save results
                all_fpr_list.append(fpr_list)
                all_tpr_list.append(tpr_list)
                all_auc_list.append(roc_auc)
                membership_list.append(membership)
                all_alg_names.append(f"{alg}+{signal_name}")
                print(100 * "#")
                
    print(f"Metric based auditing benchmark cost {time.time() - start_time:.5f} seconds.")
    print(100 * "#")

    save_audit_results(membership_list, all_fpr_list, all_tpr_list, all_alg_names, report_dir)
    return all_fpr_list, all_tpr_list, all_auc_list, membership_list