"""This file is the main entry point for running the priavcy auditing."""
import argparse
import os
import pickle
import sys
import time
import yaml
import random
import numpy as np
import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "../basic/")
torch.backends.cudnn.benchmark = True

from core import load_existing_models
from dataset import get_dataset, get_dataset_subset, DATASET_TYPE
from models import INPUT_OUTPUT_SHAPE

from utils import get_config_file, get_alg_list, read_data_split_info
from metric_ref_based_audit import metric_ref_based_benchmark
from nn_based_audit import nn_based_benchmark
from query_based_audit import query_based_benchmark
from extra_audit import extra_audit_benchmark

    


def init_configs(args):
    """
    Initialize the configuration.
    """
    args.cf = get_config_file(args.dataset, args.target_model)
    with open(args.cf, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    configs["audit"]["dataset"] = args.dataset
    configs["audit"]["ref_data_dis"] = args.ref_dis
    configs["audit"]["target_model"] = args.target_model
    configs["audit"]["reference_model"] = args.ref_model
    configs["audit"]["subtar_ratio"] = args.subtar_ratio
    configs["data"]["dataset_type"] = DATASET_TYPE[args.dataset]
    
    if type(INPUT_OUTPUT_SHAPE[args.dataset]) == list:
        configs["train"]["num_classes"] = INPUT_OUTPUT_SHAPE[args.dataset][1]
    else:
        configs["train"]["num_classes"] = INPUT_OUTPUT_SHAPE[args.dataset]

    if args.gpu != "FromConfigFile":
        configs["audit"]["device"] = "cuda:"+str(args.gpu)
        configs["train"]["device"] = "cuda:"+str(args.gpu)

    if args.number_of_reference_models > 0:
        assert args.number_of_reference_models <= configs["train"]["num_in_models"] + configs["train"]["num_out_models"], "The number of reference models used should not be greater than the number of trained reference models."
        configs["audit"]["number_of_reference_models"] = args.number_of_reference_models
    else:
        configs["audit"]["number_of_reference_models"] = configs["train"]["num_in_models"] + configs["train"]["num_out_models"]

    assert np.max(np.array(args.target_model_ids)) <= configs["train"]["num_target_model"], "The id of target models used should not be greater than the number of trained target models."
    configs["audit"]["target_model_ids"] = args.target_model_ids

    if args.query_num > 0:
        configs["audit"]["query_num"] = args.query_num
    
    if args.boundary_eva_num > 0:
        configs["audit"]["boundary_eva_num"] = args.boundary_eva_num

    # Set the random seed to facilitate reproducibility of results
    seed = configs["random_seed"]
    configs["train"]["random_seed"] = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    return configs

def load_data(configs):
    """
    Load the dataset and data split information.
    """
    # TODO: remove the param model_name in get_dataset()
    dataset = get_dataset(
        dataset_name=configs["audit"]["dataset"], 
        data_dir=configs["data"]["data_dir"], 
    )

    data_split_dir = f'''{configs["train"]["log_dir"]}/{configs["data"]["dataset_type"]}/{configs["audit"]["dataset"]}'''
    target_dataset_splits, target_dataset_splits_ref, shadow_splits, subtar_splits, subtar_shadow_splits, target_data_index, subtar_data_index = read_data_split_info(
        data_split_dir, 
        configs["audit"]["subtar_ratio"]
    )

    if configs["audit"]["ref_data_dis"] == "target_ref":
        ref_dataset_splits = target_dataset_splits_ref
    elif configs["audit"]["ref_data_dis"] == "shadow":
        ref_dataset_splits = shadow_splits
    else:
        target_data_index = subtar_data_index
        if configs["audit"]["ref_data_dis"] == "subtar":
            ref_dataset_splits = subtar_splits
        elif configs["audit"]["ref_data_dis"] == "subtar_shadow":
            ref_dataset_splits = subtar_shadow_splits
        else:
            raise ValueError("The distribution of the reference data is set incorrectly.")
        
    return dataset, target_dataset_splits, target_data_index, ref_dataset_splits, shadow_splits

def load_model_meta(dir, log_dir='meta_log'):
    model_metadata = {"model_metadata": {}}
    files = os.listdir(f"{dir}/models_metadata")
    for file in files:
        file_path = f"{dir}/models_metadata/{file}"
        model_id = int(file.split('.')[0])
        if os.path.isfile(file_path):
            with open(file_path, "rb") as f:
                metadata = pickle.load(f)
            model_metadata["model_metadata"][model_id] = metadata
    return model_metadata

def load_population_models(configs):
    dataset_name = configs["audit"]["dataset"]
    reference_model_name = configs["audit"]["reference_model"]

    meta_log_dir = configs["train"]["log_dir"]
    pop_model_dir = f'''{meta_log_dir}/{configs["data"]["dataset_type"]}/{dataset_name}/{reference_model_name}/shadow'''
        
    # Load or initialize models based on metadata
    population_model_metadata = load_model_meta(pop_model_dir, log_dir=configs["train"]["log_dir"])

    population_model_idxes = np.arange(configs["audit"]["number_of_reference_models"])

    population_model_list = []
    print("load the saved population model.")
    for idx in population_model_idxes:
        population_model_list.append(load_existing_models(
            model_metadata_dict=population_model_metadata, 
            matched_idx=[idx], 
            model_name=reference_model_name, 
            dataset=dataset, 
            dataset_name=dataset_name
        )[0])
    print(100 * "#")
    return population_model_list, population_model_idxes

def load_target_model(configs, target_model_idx=0):
    dataset_name = configs["audit"]["dataset"]
    target_model_name = configs["audit"]["target_model"]
    meta_log_dir = configs["train"]["log_dir"]
    target_model_dir = f'''{meta_log_dir}/{configs["data"]["dataset_type"]}/{dataset_name}/{target_model_name}/target'''

    # Load or initialize models based on metadata
    target_model_metadata = load_model_meta(target_model_dir, log_dir=configs["train"]["log_dir"])

    print("load the saved target model.")
    target_model = load_existing_models(
        model_metadata_dict=target_model_metadata, 
        matched_idx=[target_model_idx], 
        model_name=target_model_name, 
        dataset=dataset, 
        dataset_name=dataset_name
    )[0]

    return target_model

def load_reference_models(configs):
    dataset_name, ref_data_dis = configs["audit"]["dataset"], configs["audit"]["ref_data_dis"]
    assert ref_data_dis != "target", "reference model path can be '/target_ref', but not '/target'."
    subtar_ratio = configs["audit"]["subtar_ratio"]
    reference_model_name = configs["audit"]["reference_model"]
    meta_log_dir = f'''{configs["train"]["log_dir"]}/{configs["data"]["dataset_type"]}'''

    if ref_data_dis == "subtar":
        ref_model_dir = f"{meta_log_dir}/{dataset_name}/{reference_model_name}/subtar({subtar_ratio})"
    elif ref_data_dis == "subtar_shadow":
        ref_model_dir = f"{meta_log_dir}/{dataset_name}/{reference_model_name}/subtar({subtar_ratio})_shadow"
    else:
        ref_model_dir = f"{meta_log_dir}/{dataset_name}/{reference_model_name}/{ref_data_dis}"
        

    ref_model_metadata = load_model_meta(ref_model_dir, log_dir=configs["train"]["log_dir"])
    reference_model_idxes = np.arange(configs["audit"]["number_of_reference_models"])
    reference_model_list = []
    print("load the saved reference model.")
    for idx in reference_model_idxes:
        reference_model_list.append(load_existing_models(
            model_metadata_dict=ref_model_metadata, 
            matched_idx=[idx], 
            model_name=reference_model_name, 
            dataset=dataset, 
            dataset_name=dataset_name
        )[0])

    return reference_model_list, reference_model_idxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="student", help="Dataset name")
    parser.add_argument("--target_model", type=str, default="mlp", help="Target model name")
    parser.add_argument("--ref_model", type=str, default="mlp", help="Reference model name")
    parser.add_argument("--gpu",  type=str,  default="FromConfigFile", help="GPU device id")
    parser.add_argument("--algs", nargs='+', default="Base", help="MIA algorithms")
    parser.add_argument("--ref_dis", type=str, default="target_ref", choices=["target_ref", "shadow", \
        "subtar", "subtar_shadow"], help="Data distribution of reference models")
    parser.add_argument("--subtar_ratio", type=float, default=0.5, help="Sampling target dataset ratio")
    parser.add_argument("--target_model_ids", type=int,  nargs='+', default=[0,1,2,3,4], help="The audited target model id")
    parser.add_argument("--number_of_reference_models", type=int, default=16, help="The number of reference models used in auditing algorithms.")
    parser.add_argument("--query_num", type=int, default=-1, help="Number of additional queries for each sample in ['query-noise', 'query-augmented'] algorthms.")
    parser.add_argument("--boundary_eva_num", type=int, default=-1, help="Number of evaluation samples in boundary-based auditing methods.")
    
    ################################################################################
    ######################### Initialize the key variable  #########################
    ################################################################################

    torch.set_num_threads(5)
    start_time = time.time()
    
    args = parser.parse_args()
    configs = init_configs(args)
    metric_ref_alg_list, nn_alg_list, query_alg_list, extra_alg_list = get_alg_list(args.algs, args.ref_dis)
    gbdt_quantile_flag = configs["audit"]["reference_model"] in ['catboost', 'lightgbm', 'lr'] and "query-quantile+rescaled_logits" in extra_alg_list

    ##################################################################################
    ###############  Load the dataset and population models ##########################
    ##################################################################################

    print(f"Loading the dataset.")
    s_time = time.time()
    dataset, target_dataset_splits, target_data_index, ref_dataset_splits, shadow_splits = load_data(configs)

    target_dataset = get_dataset_subset(dataset, target_data_index)

    print(f"Loading the population model.")
    if len(metric_ref_alg_list) > 0 or gbdt_quantile_flag:
        population_model_list, population_model_idxes = load_population_models(configs)
        population_train_data_indexes = [shadow_splits["split"][idx]["train"] for idx in population_model_idxes]
        population_data_index = np.hstack((shadow_splits["split"][0]["train"], shadow_splits["split"][0]["test"]))
        population_dataset = get_dataset_subset(dataset, population_data_index)
    
    reference_data_index = np.hstack((ref_dataset_splits["split"][0]["train"], ref_dataset_splits["split"][0]["test"]))
    reference_dataset = get_dataset_subset(dataset, reference_data_index)

    print(
        f"Prepare the datasets and population models costs {time.time() - s_time:.5f} seconds" 
    )
    print(100 * "#")

    ##################################################################################
    #####################  Load the reference model  #################################
    ##################################################################################

    if len(metric_ref_alg_list) > 0 or len(nn_alg_list) > 0 or len(query_alg_list) > 0 or gbdt_quantile_flag:
        s_time = time.time()
        reference_model_list, reference_model_idxes = load_reference_models(
            configs=configs, 
        )

        nn_reference_model_idx = 0
        reference_train_data_indexes = [ref_dataset_splits["split"][idx]["train"] for idx in reference_model_idxes]
        nn_reference_model_train_index = reference_train_data_indexes[nn_reference_model_idx]
        nn_reference_membership = np.in1d(reference_data_index, nn_reference_model_train_index).astype(int)

        print(
            f"Prepare the reference models and reference membership costs {time.time() - s_time:.5f} seconds" 
        )
        print(100 * "#")
    
    ##################################################################################
    ###########################  Auditing benchmark  #################################
    ##################################################################################

    configs["audit"]["report_dir"] = f'''{configs['audit']['report_log']}'''
    configs["quantile_root_checkpoint_path"] = f'''{configs["train"]["log_dir"]}/quantile_log/'''

    for target_model_idx in configs["audit"]["target_model_ids"]:
        ##################################################################################
        ##############################  Extra audits #####################################
        ##################################################################################

        s_time = time.time()
        print(f"Extra auditing benchmark starts.")
        # Query-ref and Query-quantile are implemented using the code from the original paper.

        for alg in extra_alg_list:
            extra_audit_benchmark(alg, configs, target_model_idx)

        if gbdt_quantile_flag:
            metric_ref_alg_list.append("query-quantile+rescaled_logits")

        print(f"Extra audits cost {time.time() - s_time:.5f} seconds.")
        print(100 * "#")

        ##################################################################################
        ###########################  Load the target model  ##############################
        ##################################################################################

        s_time = time.time()
        target_model = load_target_model(
            configs=configs, 
            target_model_idx=target_model_idx,
        )

        print(
            f"Prepare the target model costs {time.time() - s_time:.5f} seconds." 
        )
        print(100 * "#")

        ##################################################################################
        ############  Load the training indexes and membership  ##########################
        ##################################################################################

        target_model_train_index = target_dataset_splits["split"][target_model_idx]["train"]
        membership = np.in1d(target_data_index, target_model_train_index).astype(int)

        report_dir = f'''{configs["audit"]["report_dir"]}/target_model_idx_{target_model_idx}'''
        print(100 * "#")

        ##################################################################################
        ##############################  Metric based #####################################
        ##################################################################################

        if len(metric_ref_alg_list) > 0:
            metric_ref_based_benchmark(
                alg_list=metric_ref_alg_list,
                target_model=target_model,
                reference_model_list=reference_model_list,
                population_model_list=population_model_list,
                target_dataset=target_dataset,
                population_dataset=population_dataset,
                target_data_index=target_data_index,
                population_data_index=population_data_index,
                reference_train_data_indexes=reference_train_data_indexes,
                population_train_data_indexes=population_train_data_indexes,
                membership=membership,
                configs=configs,
                report_dir=report_dir,
            )

        ##################################################################################
        ###########################  Neural network based  ###############################
        ##################################################################################
        
        if len(nn_alg_list) > 0:
            nn_based_benchmark(
                alg_list=nn_alg_list,
                target_model=target_model,
                reference_model=reference_model_list[nn_reference_model_idx],
                target_dataset=target_dataset,
                reference_dataset=reference_dataset,
                membership=membership,
                reference_membership=nn_reference_membership,
                configs=configs,
                report_dir=report_dir,
            )

        ####################################################################################
        ################################  Query based  #####################################
        ####################################################################################

        if len(query_alg_list) > 0:
            query_based_benchmark(
                alg_list=query_alg_list,
                target_model=target_model,
                reference_model=reference_model_list[nn_reference_model_idx],
                target_dataset=target_dataset,
                reference_dataset=reference_dataset,
                dataset=dataset,
                target_data_index=target_data_index,
                reference_data_index=reference_data_index,
                membership=membership,
                reference_membership=nn_reference_membership,
                configs=configs,
                report_dir=report_dir,
            )


        use_time = time.time() - start_time
        print(100 * "#")
        print(f"End of auditing target model:{target_model_idx}.")
        print(f"Run the code for the all steps costs {use_time:.5f} seconds.")
        print(100 * "#")
