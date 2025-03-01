import argparse
import os
import pickle
import sys
import time
import yaml
import random
import numpy as np
import torch

sys.path.insert(0, "../basic/")
torch.backends.cudnn.benchmark = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from core import prepare_datasets_for_reference_in_attack, prepare_models
from dataset import get_dataset, DATASET_TYPE
from utils import save_data_split_info, read_data_split_info, get_config_file

##################################################################################
###############  Prepare the models based on data split info  ####################
##################################################################################
def prepare_model(
    model_ids, 
    model_checkpoint_dir, 
    dataset_name, 
    dataset, 
    data_split_info, 
    configs
):
    if len(model_ids) > 0:
        Path(f"{model_checkpoint_dir}/models_metadata").mkdir(parents=True, exist_ok=True)
        # Initialize models based on metadata
        model_metadata_list = {"model_metadata": {}}

        # Load model
        baseline_time = time.time()
        # check if the models are trained
        (model_list, model_metadata_dict, trained_model_idx_list) = prepare_models(
            model_ids,
            model_metadata_list,
            f"{model_checkpoint_dir}",
            dataset_name,
            dataset,
            data_split_info,
            configs["train"],
        )
        
        model_metadata = model_metadata_dict["model_metadata"]
        for model_id in model_ids:
            with open(f"{model_checkpoint_dir}/models_metadata/{model_id}.pkl", "wb") as f:
                pickle.dump(model_metadata[model_id], f)
        
        print(
            f"Prepare the models costs {time.time() - baseline_time:.5f} seconds."
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   type=str,  default="student", help="Dataset name")
    parser.add_argument("--model",     type=str,  default="mlp", help="Model name")
    parser.add_argument("--gpu",       type=str,  default="FromConfigFile", help="GPU device id")
    parser.add_argument("--model_ids", type=int,  nargs='+', default=[0], help="-1 means generating dataset split information. Other ids indicate training target/reference model")
    parser.add_argument("--subtar_ratio", type=float, default=0.5, help="Sampling target dataset ratio")

    parser.add_argument("--not_train_target_flag",     action="store_true", help="Train models based on target_dataset_splits or not.")
    parser.add_argument("--not_train_target_ref_flag", action="store_true", help="Train models based on target_dataset_splits_ref or not.")
    parser.add_argument("--not_train_shadow_flag",     action="store_true", help="Train models based on shadow_dataset_splits or not.")
    parser.add_argument("--train_subtar_flag",     action="store_true", help="Train models based on subtar_dataset_splits or not.")
    parser.add_argument("--train_subtar_shadow_flag", action="store_true", help="Train models based on subtar_shadow_dataset_splits or not.")

    ##################################################################################
    ###########################  Load the parameters  ################################
    ##################################################################################

    torch.set_num_threads(1)
    
    args = parser.parse_args()
    dataset_name, model_name = args.dataset, args.model
    print(f"Dataset name: {dataset_name}, Model name: {model_name}, GPU: {args.gpu}.")
    
    subtar_ratio = args.subtar_ratio
    args.cf = get_config_file(dataset_name, model_name)
    with open(args.cf, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)
    
    configs["data"]["dataset"] = dataset_name
    configs["train"]["model_name"] = model_name
    if args.gpu != "FromConfigFile":
        configs["train"]["device"] = "cuda:"+str(args.gpu)
    
    from models import INPUT_OUTPUT_SHAPE
    configs["train"]["num_classes"] = INPUT_OUTPUT_SHAPE[dataset_name]

    model_ids = args.model_ids
    model_ids.sort()

    # Set the random seed and log_dir
    seed = configs["random_seed"]
    configs["train"]["random_seed"] = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Create folders for saving the logs if they do not exist
    dataset_type = DATASET_TYPE[args.dataset]
    log_dir = f'''{configs["train"]["log_dir"]}/{dataset_type}/{dataset_name}'''
    model_checkpoint_dir = f"{log_dir}/{model_name}"

    # Set up the logger
    start_time = time.time()

    ##################################################################################
    ###########################  Load the dataset  ###################################
    ##################################################################################

    baseline_time = time.time()
    dataset = get_dataset(dataset_name, configs["data"]["data_dir"])

    # Target dataset size (The left is audit/auxiliary dataset)
    dataset_size = int(configs["data"]["target_ratio"] * len(dataset))  
    p_ratio = configs["data"]["train_ratio"]         # N_target_training / N_target
    num_target_model = configs["train"]["num_target_model"]
    num_reference_models = configs["train"]["num_in_models"] + configs["train"]["num_out_models"]

    '''
    Each original dataset is divided into two disjoint datasets: target dataset and shadow dataset.
    subtar_dataset is subset of target dataset.
    subtar_shadow_dataset = subtar_dataset + shadow dataset.
    Target model is trained and tested in target dataset. Reference model is trained and tested in the shadow/subtar/subtar_shadow dataset.
    '''
    if -1 in model_ids:
        target_dataset_splits, target_dataset_splits_ref, shadow_dataset_splits, subtar_dataset_splits, subtar_shadow_dataset_splits, target_data_index, subtar_data_index = \
            prepare_datasets_for_reference_in_attack(
                len(dataset),
                dataset_size,
                num_target_models=num_target_model,
                num_reference_models=num_reference_models,
                keep_ratio=p_ratio,
                is_uniform=True, #is_uniform=False,
                sub_target_ratio=subtar_ratio
            )
        save_data_split_info(log_dir, target_dataset_splits, target_dataset_splits_ref, shadow_dataset_splits, subtar_dataset_splits, subtar_shadow_dataset_splits, target_data_index, subtar_data_index, subtar_ratio)

    target_dataset_splits, target_dataset_splits_ref, shadow_dataset_splits, subtar_dataset_splits, subtar_shadow_dataset_splits, target_data_index, subtar_data_index = read_data_split_info(log_dir, subtar_ratio)

    ##################################################################################
    ###########################  Prepare base model  #################################
    ##################################################################################

    new_target_model_ids = [model_id for model_id in model_ids if model_id < num_target_model]
    new_ref_model_ids = [model_id for model_id in model_ids if model_id < num_reference_models]
    
    if -1 not in model_ids:
        if not args.not_train_target_flag:
            print("Train target models.")
            prepare_model(new_target_model_ids, f"{model_checkpoint_dir}/target", dataset_name, dataset, target_dataset_splits, configs)

        if not args.not_train_target_ref_flag:
            print("Train reference models with target dataset.")
            prepare_model(new_ref_model_ids, f"{model_checkpoint_dir}/target_ref", dataset_name, dataset, target_dataset_splits_ref, configs)
            
        if not args.not_train_shadow_flag:
            print("Train reference models with shadow dataset.")
            prepare_model(new_ref_model_ids, f"{model_checkpoint_dir}/shadow", dataset_name, dataset, shadow_dataset_splits, configs)
        if args.train_subtar_flag:
            print("Train reference models with portion of the target dataset.")
            prepare_model(new_ref_model_ids, f"{model_checkpoint_dir}/subtar({subtar_ratio})", dataset_name, dataset, subtar_dataset_splits, configs)
        if args.train_subtar_shadow_flag:
            print("Train reference models with portion of the target dataset + shadow dataset.")
            prepare_model(new_ref_model_ids, f"{model_checkpoint_dir}/subtar({subtar_ratio})_shadow", dataset_name, dataset, subtar_shadow_dataset_splits, configs)

    use_time = time.time() - start_time
    print(100 * "#")
    print("End of run")
    print(f"Run the code for the all steps costs {use_time:.5f} seconds")
    print(100 * "#")
