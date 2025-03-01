import pickle
import os
import numpy as np

import json
from typing import Dict, List, Union
from pathlib import Path
from sklearn.metrics import auc
from info import *

def save_data_split_info(
    log_dir: str, 
    target_dataset_splits: Dict,
    target_dataset_splits_ref: Dict,
    shadow_dataset_splits: Dict, 
    subtar_dataset_splits: Dict, 
    subtar_shadow_dataset_splits: Dict, 
    target_data_index: np.array, 
    subtar_data_index: np.array, 
    subtar_ratio: float,
):
    """
    Save data split information and indexes of target dataset and subset of the target dataset in the whole dataset.
    """
    data_split_info_dir = f"{log_dir}/data_split_info"
    Path(data_split_info_dir).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(f"{data_split_info_dir}/target_dataset_splits"):
        with open(f"{data_split_info_dir}/target_dataset_splits","wb") as f:
            pickle.dump(target_dataset_splits, f)
        f.close()
        np.save(f"{data_split_info_dir}/target_data_index", target_data_index)
        print("Save target_dataset_splits and target_data_index.")
    else:
        print("target_dataset_splits and target_data_index already exist.")

    if not os.path.exists(f"{data_split_info_dir}/target_dataset_splits_ref"):
        with open(f"{data_split_info_dir}/target_dataset_splits_ref","wb") as f:
            pickle.dump(target_dataset_splits_ref, f)
        f.close()
        print("Save target_dataset_splits_ref.")
    else:
        print("target_dataset_splits_ref already exist.")

    if not os.path.exists(f"{data_split_info_dir}/shadow_dataset_splits"):
        with open(f"{data_split_info_dir}/shadow_dataset_splits","wb") as f:
            pickle.dump(shadow_dataset_splits, f)
        f.close()
        print("Save shadow_dataset_splits.")
    else:
        print("shadow_dataset_splits already exist.")

    if not os.path.exists(f"{data_split_info_dir}/subtar({subtar_ratio})_dataset_splits"):
        with open(f"{data_split_info_dir}/subtar({subtar_ratio})_dataset_splits","wb") as f:
            pickle.dump(subtar_dataset_splits, f)
        f.close()
        np.save(f"{data_split_info_dir}/subtar({subtar_ratio})_data_index", subtar_data_index)
        print(f"Save subtar({subtar_ratio})_dataset_splits and subtar({subtar_ratio})_data_index.")
    else:
        print(f"subtar({subtar_ratio})_dataset_splits and subtar({subtar_ratio})_data_index already exist.")

    if not os.path.exists(f"{data_split_info_dir}/subtar({subtar_ratio})_shadow_dataset_splits"):
        with open(f"{data_split_info_dir}/subtar({subtar_ratio})_shadow_dataset_splits","wb") as f:
            pickle.dump(subtar_shadow_dataset_splits, f)
        f.close()
        print(f"Save subtar({subtar_ratio})_shadow_dataset_splits.")
    else:
        print(f"subtar({subtar_ratio})_shadow_dataset_splits already exist.")

def read_data_split_info(log_dir: str, subtar_ratio: float):
    """
    Read data split information and indexes of target dataset and subset of the target dataset in the whole dataset.
    """
    data_split_info_dir = f"{log_dir}/data_split_info"

    with open(f"{data_split_info_dir}/target_dataset_splits","rb") as f:
        target_dataset_splits = pickle.load(f)
    f.close()

    with open(f"{data_split_info_dir}/target_dataset_splits_ref","rb") as f:
        target_dataset_splits_ref = pickle.load(f)
    f.close()

    with open(f"{data_split_info_dir}/shadow_dataset_splits","rb") as f:
        shadow_dataset_splits = pickle.load(f)
    f.close()

    with open(f"{data_split_info_dir}/subtar({subtar_ratio})_dataset_splits","rb") as f:
        subtar_dataset_splits = pickle.load(f)
    f.close()

    with open(f"{data_split_info_dir}/subtar({subtar_ratio})_shadow_dataset_splits","rb") as f:
        subtar_shadow_dataset_splits = pickle.load(f)
    f.close()

    target_data_index = np.load(f"{data_split_info_dir}/target_data_index.npy")
    subtar_data_index = np.load(f"{data_split_info_dir}/subtar({subtar_ratio})_data_index.npy")
    return target_dataset_splits, target_dataset_splits_ref, shadow_dataset_splits, subtar_dataset_splits, subtar_shadow_dataset_splits, target_data_index, subtar_data_index

def _get_alg_list(alg):
    return METHODS_ALIASES_REVERSE[alg] if alg in METHODS_ALIASES_REVERSE.keys() else alg

def get_alg_list(algs: List, ref_data_dis: str):
    """
    Preprocess the input algorithm name in the command to get the algorithms list.
    """
    alg_list_map = {
        "benchmark": ALL_ALG_LIST,
        "bench": ALL_ALG_LIST,
        "metric": METRIC_ALG_LIST,
        "learn": NN_ALG_LIST,
        "model": REFERENCE_ALG_LIST,
        "query": QUERY_ALG_LIST+EXTRA_ALG_LIST,

        "target": TARGET_ALG_LIST,
        "nn": NN_ALG_LIST,
        "reference": REFERENCE_ALG_LIST,
        "referenceoff": REFERENCE_ALG_OFFLINE_LIST,
        "base": BASE_ALG_LIST, 
        "extra": EXTRA_ALG_LIST, 
    }

    if len(algs) == 1:
        alg_list = alg_list_map.get(algs[0].lower(), [_get_alg_list(algs[0])])
    else:
        alg_list = []
        for alg in algs:
            alg_list += alg_list_map.get(alg.lower(), [_get_alg_list(alg)])

    print(f"Algorithms to run: {alg_list}")
    check_alg_name(alg_list)
    
    if ref_data_dis == "shadow":
        metric_ref_alg_list = list(set(REFERENCE_ALG_OFFLINE_LIST+METRIC_ALG_LIST) & set(alg_list))
    else:
        metric_ref_alg_list = list(set(REFERENCE_ALG_LIST+METRIC_ALG_LIST) & set(alg_list))
        
    nn_alg_list = list(set(NN_ALG_LIST) & set(alg_list))
    query_alg_list = list(set(QUERY_ALG_LIST) & set(alg_list))
    extra_alg_list = list(set(EXTRA_ALG_LIST) & set(alg_list))

    if "query+adversarial_for_query_num" in alg_list:
        query_alg_list.append("query+adversarial_for_query_num")
    return metric_ref_alg_list, nn_alg_list, query_alg_list, extra_alg_list

def check_alg_name(alg_list: List):
    """
    Check whether each algorithm in the algorithm list is supported.
    """
    for alg in alg_list:
        assert alg in ALL_ALG_LIST, f"Algorithm {alg} is not supported!"

def select_alg(algs: Union[List, str], signal_name: str):
    """
    Get algorithms that match the given signal name.
    """
    if type(algs) == list:
        alg_list = []
        for alg in algs:
            if alg.split('+')[1] == signal_name:
                alg_list.append(alg.split('+')[0])
        return alg_list
    elif type(algs) == str:
        if algs.split('+')[1] == signal_name:
            return algs.split('+')[0]
    else:
        raise TypeError("Input algorithms should be 'List' or 'String'.")
    
def get_config_file(dataset_name: str, model_name: str):
    """
    Select the appropriate config file based on the model name and dataset name.
    """
    # get domain that given dataset support
    for key, value in DOMAIN_SUPPORT_DATASET.items():
        if dataset_name in value:
            dataset_domain = key

    # get domains that given model support
    model_domains = []
    for key, value in DOMAIN_SUPPORT_MODEL.items():
        if model_name in value:
            model_domains.append(key)
    print(dataset_domain, model_domains)
    assert dataset_domain in model_domains, "dataset_domain and model_domain should match"
    
    dataset_config_file_name = f"configs/config_benchmark_{dataset_name}.yaml"
    model_config_file_name = f"configs/config_benchmark_{model_name}.yaml"

    if os.path.exists(model_config_file_name) and os.path.exists(dataset_config_file_name):
        raise ValueError(f"Please manually select which config file to use.")
    
    if os.path.exists(model_config_file_name):
        config_file_name = model_config_file_name
    elif os.path.exists(dataset_config_file_name):
        config_file_name = dataset_config_file_name
    else:
        config_file_name = f"configs/config_benchmark_{dataset_domain}.yaml"
    return config_file_name

def generate_mask_matrix(reference_indexes: List[np.array], target_index: np.array):
    """
    Generate mask matrix indicating that whether the target index appears in the reference index.

    Returns:
        mask_matrix (List[np.array]): i-th row of the mask_matrix indicates that whether the target index appears in the i-th row of the reference indexes.
    """
    mask_matrix = np.zeros((len(reference_indexes), len(target_index))).astype(bool)
    for i in range(len(reference_indexes)):
        mask_matrix[i] = np.in1d(target_index, reference_indexes[i])
    return mask_matrix

def _log_value(probs: np.array, small_value: float=1e-30):
    return -np.log(np.maximum(probs, small_value))

def _m_entr_comp(probs: np.array, true_labels: np.array):
    """
    Compute modified prediction entropy based on original open-source code

    Reference:
        Systematic Evaluation of Privacy Risks of Machine Learning Models", accepted by USENIX Security 2021
    """
    log_probs = _log_value(probs)
    reverse_probs = 1-probs
    log_reverse_probs = _log_value(reverse_probs)
    modified_probs = np.copy(probs)
    modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
    modified_log_probs = np.copy(log_reverse_probs)
    modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
    return np.sum(np.multiply(modified_probs, modified_log_probs),axis=1)

def save_audit_results(membership_list, fpr_lists, tpr_lists, alg_names, save_path):
    """
    Save audit results, including fpr, tpr, and membership to help calculate other metrics like precision, fnr, etc.
    """
    if len(alg_names) > 0:
        for fpr, tpr, membership, alg_name in zip(fpr_lists, tpr_lists, membership_list, alg_names):
            alg_name = METHODS_ALIASES[alg_name]
            Path(f"{save_path}/{alg_name}").mkdir(parents=True, exist_ok=True)
            np.save(f"{save_path}/{alg_name}/fpr",fpr)
            np.save(f"{save_path}/{alg_name}/tpr",tpr)
            np.save(f"{save_path}/{alg_name}/label",membership)
            save_more_metric(base_dir=f"{save_path}/{alg_name}", alg_name=alg_name)

def save_more_metric(base_dir, alg_name):
    """
    Save more audit metrics, like precision, fnr, etc.
    """
    fpr = np.load(f"{base_dir}/fpr.npy")
    tpr = np.load(f"{base_dir}/tpr.npy")
    if os.path.exists(f"{base_dir}/positive.npy") and os.path.exists(f"{base_dir}/negative.npy"):
        positive = np.load(f"{base_dir}/positive.npy")
        negative = np.load(f"{base_dir}/negative.npy")
        if len(positive.shape) > 1:
            positive = positive[0]
            negative = negative[0]
    elif os.path.exists(f"{base_dir}/label.npy"):
        membership = np.load(f"{base_dir}/label.npy")
        positive = np.sum(membership==1)
        negative = np.sum(membership!=1)
    else:
        print('Need to know number of negative and positive samples')
    
    auroc = auc(fpr, tpr)
    tpr_at_low_fpr_1 = tpr[np.where(fpr<.01)[0][-1]]
    tpr_at_low_fpr_2 = tpr[np.where(fpr<.1)[0][-1]]

    # Set threshold to get best accurancy, fpr and tpr is calculated based on this threshold.
    acc = np.max(1-(fpr+(1-tpr))/2)
    acc_sort = np.argmax(1-(fpr+(1-tpr))/2)
    tpr  = tpr[acc_sort]
    fpr  = fpr[acc_sort]

    # Calculate tp, fn, fp and tn
    tp = int(tpr * positive)
    fn = positive - tp
    fp = int(fpr * negative)
    fnr = np.nan_to_num(fn / (tp + fn))

    # Calculate precision, recall, and f1 score
    precision = np.nan_to_num(tp / (tp + fp)) if tp + fp != 0 else 0
    recall    = np.nan_to_num(tp / (tp + fn))
    f1 = np.nan_to_num((2 * precision * recall) / (precision + recall))

    # Save auditing results based on different evaluation metrics
    metric_list = [alg_name, precision, recall, f1, acc, fnr, fpr, tpr-fpr, tpr_at_low_fpr_1, tpr_at_low_fpr_2, auroc] #tpr, 
    np.save(f"{base_dir}/metrics.npy", np.array(metric_list))

    results = {"Algorithm": alg_name, 'Precision': precision, "Recall": recall, "F1-score": f1, "Accuracy": acc, 
        "FNR": fnr, "FPR": fpr, "Membership advantage": tpr-fpr, "TPR@1%FPR": tpr_at_low_fpr_1, 
        "TPR@10%FPR": tpr_at_low_fpr_2, "AUROC": auroc}
    with open(f"{base_dir}/results.json", "w") as f:
        json.dump(results,f)
    return metric_list





