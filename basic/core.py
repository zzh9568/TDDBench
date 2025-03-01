import copy
import logging
import pickle
import time
from ast import List

import numpy as np
import torch
import torchvision
import joblib
import lightgbm as lgb
import evaluate

from torch import nn
from torchvision import transforms
from catboost import CatBoostClassifier
from dataset import get_dataloader, text_dataset_preprocess

from models import get_model
from train import train, inference, catboost_train, catboost_inference, \
    text_model_train, text_model_inference
from transformers import AutoModelForSequenceClassification

def load_existing_models(
    model_metadata_dict: dict,
    matched_idx: List(int),
    model_name: str,
    dataset: torchvision.datasets,
    dataset_name: str,
):
    """Load existing models from dicks for matched_idx.

    Args:
        model_metadata_dict (dict): Model metedata dict.
        matched_idx (List): List of model index we want to load.
        model_name (str): Model name.
        dataset_list (List): Dataset List.
        dataset (torchvision.datasets): Dataset.
        dataset_name (str): Dataset name.
    Returns:
        List (nn.Module): List of models.
    """
    model_list = []
    if len(matched_idx) > 0:
        for metadata_idx in matched_idx:
            metadata = model_metadata_dict["model_metadata"][metadata_idx]
            if dataset.dataset_type == 'text':
                from models import INPUT_OUTPUT_SHAPE
                num_labels = INPUT_OUTPUT_SHAPE[dataset_name]
                model = AutoModelForSequenceClassification.from_pretrained(metadata['model_path'], num_labels=num_labels)
            elif model_name == 'catboost':
                model = CatBoostClassifier(task_type="GPU", devices="cuda:1")
                model.load_model(f"{metadata['model_path']}")
            elif model_name in ['lightgbm', "svm", "lr"]:
                model = joblib.load(f"{metadata['model_path']}")
                model.n_jobs = 1
            else:
                model = get_model(model_name, dataset_name)
                with open(f"{metadata['model_path']}", "rb") as file:
                    model_weight = pickle.load(file)
                model.load_state_dict(model_weight)
            model_list.append(model)

            print(f"load the saved checkpoint from paths: {metadata['model_path']}")
        return model_list
    else:
        return []


def prepare_datasets_for_reference_in_attack(
    all_dataset_size: int,
    target_dataset_size: int,
    num_target_models: int,
    num_reference_models: int,
    keep_ratio: float,
    is_uniform: bool,
    sub_target_ratio: float,
):
    """Prepare the datasets for reference_in attacks. Each data point will be randomly chosen by half of the models with probability keep_ratio and the rest of the models will be trained on the rest of the dataset.
    The partioning method is from https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/train.py
    Args:
        all_dataset_size (int): Size of the whole dataset.
        target_dataset_size (int): Size of the target dataset.
        num_target_models (int): Number of target model.
        num_reference_models (int): Number of reference models.
        keep_ratio (float): Indicate the probability of keeping the target point for training the model.
        is_uniform (bool): Indicate whether to perform the splitting in a uniform way.
        sub_target_ratio (float): The ratio of subset of the target dataset to the target dataset.
    Returns:
        dict: Data split information.
        list: List of boolean indicating whether the model is trained on the target point.
        list: List of target data index on which the adversary wants to infer the membership.
    """
    assert num_target_models >= 1 and num_reference_models >= 2, "At least 1 target model and 2 reference model."
    assert num_reference_models % 2 == 0, "The reference models should be paired."

    target_index = np.random.choice(all_dataset_size, target_dataset_size, replace=False)
    shadow_index = np.setdiff1d(np.arange(all_dataset_size), target_index)

    num_sub_target = int(target_dataset_size * sub_target_ratio)
    subtar_index = np.random.choice(target_index, num_sub_target, replace=False)


    target_dataset_splits = generate_dataset_split(target_index, shadow_index, num_target_models, keep_ratio, is_uniform, for_target_model=True)
    
    target_dataset_splits_ref = generate_dataset_split(target_index, None, num_reference_models, keep_ratio, is_uniform, for_target_model=False)

    shadow_dataset_splits = generate_dataset_split(shadow_index, None, num_reference_models, keep_ratio, is_uniform, for_target_model=False)
    subtar_dataset_splits = generate_dataset_split(subtar_index, None, num_reference_models, keep_ratio, is_uniform, for_target_model=False)
    subtar_shadow_dataset_splits = generate_dataset_split(np.hstack((subtar_index, shadow_index)), None, num_reference_models, keep_ratio, is_uniform, for_target_model=False)
    return target_dataset_splits, target_dataset_splits_ref, shadow_dataset_splits, subtar_dataset_splits, subtar_shadow_dataset_splits, target_index, subtar_index

def generate_dataset_split(
    all_index: np.array,
    left_index: np.array,
    num_models: int,
    keep_ratio: float,
    is_uniform: bool,
    for_target_model: bool,
):
    if not for_target_model:
        assert num_models % 2 == 0, "The reference models should be paired."
        num_models = int(num_models / 2)

    dataset_size = len(all_index)
    index_list = []
    if is_uniform:
        keep = np.random.uniform(0, 1, size=(num_models, dataset_size)) <= keep_ratio
    else:
        selected_matrix = np.random.uniform(0, 1, size=(num_models, dataset_size))
        order = selected_matrix.argsort(0)
        keep = order < int(keep_ratio * num_models)
    
    if for_target_model:
        for i in range(num_models):
            index_list.append(
                {
                    "train": all_index[keep[i]],
                    "test": all_index[~keep[i]],
                    "audit": left_index,
                }
            )
        dataset_splits = {"split": index_list, "split_method": f"random_{keep_ratio}_for_target_model."}
    else:
        for i in range(num_models):
            index_list.append(
                {
                    "train": all_index[keep[i]],
                    "test": all_index[~keep[i]],
                    "audit": left_index,
                }
            )
            index_list.append(
                {
                    "train": all_index[~keep[i]],
                    "test": all_index[keep[i]],
                    "audit": left_index,
                }
            )
        dataset_splits = {"split": index_list, "split_method": f"random_{keep_ratio}_for_paired_reference_models."}
    return dataset_splits

def get_train_transform(dataset, dataset_name, augmentation, image_size):
    if dataset_name in ["celeba", "lfw"]:
        if augmentation == "crop":
            train_transform = transforms.Compose(
                [
                    transforms.Resize(size=(32, 32)),
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        elif augmentation == "flip":                
            train_transform = transforms.Compose(
                [
                    transforms.Resize(size=(32, 32)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        elif augmentation == "both":                
            train_transform = transforms.Compose(
                [
                    transforms.Resize(size=(32, 32)),
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            train_transform = dataset.transform
    else:
        if augmentation == "crop":
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        elif augmentation == "flip":                
            train_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        elif augmentation == "both":                
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            train_transform = dataset.transform
    return train_transform

def prepare_models(
    model_ids: List(int),
    model_metadata_dict: dict,
    log_dir: str,
    dataset_name: str,
    dataset: torchvision.datasets,
    data_split: dict,
    configs: dict,
    save_model: bool=True,
):
    """Train models based on the dataset split information.

    Args:
        model_ids (List[int]): Indicate the training model's ids (0 for target model).
        model_metadata_dict (dict): Metadata information about the existing models.
        log_dir (str): Log directory that saved all the information, including the model checkpoint.
        dataset_name (str): Name of the dataset.
        dataset (torchvision.datasets): The whole dataset.
        data_split (dict): Data split information. 'split' contains a list of dict, each of which has the train, test and audit information. 'split_method' indicates the how the dataset is generated.
        configs (dict): Indicate the traininig information.
        save_model (bool): Whether to save model checkpoint.
    Returns:
        nn.Module: List of trained models
        dict: Updated Metadata of the existing models
        List(int): Updated index list that matches the target model configurations.
    """
    # Initialize the model list
    model_list = []
    target_model_idx_list = []
    # Train the additional target models based on the dataset split
    for split in model_ids:
        meta_data = {}
        baseline_time = time.time()

        print(50 * "-")
        print(
            f"Training the {split}-th model: ",
            f"Train size {len(data_split['split'][split]['train'])}, Test size {len(data_split['split'][split]['test'])}",
        )

        if dataset.dataset_type == 'text':
            pro_dataset = text_dataset_preprocess(
                model_name=configs["model_name"],
                all_data=dataset, 
                padding=False,
            )
            trainer = text_model_train(log_dir, pro_dataset, data_split["split"][split]["train"], data_split["split"][split]["test"], configs)
            train_dataset = pro_dataset.select(indices=data_split["split"][split]["train"])
            test_dataset  = pro_dataset.select(indices=data_split["split"][split]["test"])
            train_loss, train_acc = text_model_inference(trainer, train_dataset)
            test_loss, test_acc = text_model_inference(trainer, test_dataset)
            print(f"Train accuracy {train_acc}, Train Loss {train_loss}")
            print(f"Valid(Test) accuracy {test_acc}, Valid(Test) Loss {test_loss}")
            model = trainer.model

        elif configs["model_name"] in ["catboost", "lightgbm", "svm", "lr"]:
            train_loader = get_dataloader(
                torch.utils.data.Subset(dataset, data_split["split"][split]["train"]),
                batch_size=len(data_split["split"][split]["train"]),
                shuffle=True,
            )
            for _X_train, _y_train in train_loader:
                pass
            test_loader = get_dataloader(
                torch.utils.data.Subset(dataset, data_split["split"][split]["test"]),
                batch_size=len(data_split["split"][split]["train"]),
            )
            for _X_valid, _y_valid in test_loader:
                pass

            _X_train, _y_train = _X_train.numpy(), _y_train.numpy()
            _X_valid, _y_valid = _X_valid.numpy(), _y_valid.numpy()
            # Train the target model based on the configurations.
            model, best_params = catboost_train(_X_train, _y_train, _X_valid, _y_valid, configs, dataset_name)

            # Test performance on the training dataset and test dataset
            test_acc = catboost_inference(model, _X_valid, _y_valid, configs["model_name"])
            train_acc = catboost_inference(model, _X_train, _y_train, configs["model_name"])
            print(f"Train accuracy {train_acc}")
            print(f"Valid(Test) accuracy {test_acc}")
            train_loss, test_loss = "None", "None"

        elif configs["model_name"] != "speedyresnet":
            if dataset.dataset_type == "image":
                #dataset = copy.deepcopy(dataset)
                augmentation = configs["augmentation"].lower()
                assert augmentation in ["none", "crop", "flip", "both"], f"Augmentaion {augmentation} is not supported!"
                image_size =  dataset[0][0].size()[-1]

                train_transform = get_train_transform(dataset, dataset_name, augmentation, image_size)
                dataset_copy = copy.deepcopy(dataset)
                dataset_copy.transform = train_transform
                train_dataset = torch.utils.data.Subset(dataset_copy, data_split["split"][split]["train"])
                test_dataset  = torch.utils.data.Subset(dataset, data_split["split"][split]["test"])

            else:
                train_dataset = torch.utils.data.Subset(dataset, data_split["split"][split]["train"])
                test_dataset  = torch.utils.data.Subset(dataset, data_split["split"][split]["test"])

            train_loader = get_dataloader(
                train_dataset,
                batch_size=configs["batch_size"],
                shuffle=True,
            )
            test_loader = get_dataloader(
                test_dataset,
                batch_size=configs["test_batch_size"],
            )

            # Train the target model based on the configurations.
            model = train(
                get_model(configs["model_name"], dataset_name),
                train_loader,
                configs,
                test_loader,
            )
            # Test performance on the training dataset and test dataset
            test_loss, test_acc = inference(model, test_loader, configs["device"])
            train_loss, train_acc = inference(model, train_loader, configs["device"])
            print(f"Train accuracy {train_acc}, Train Loss {train_loss}")
            print(f"Valid(Test) accuracy {test_acc}, Valid(Test) Loss {test_loss}")

        else:
            raise ValueError(
                f"The {configs['model_name']} is not supported for the {dataset_name}."
            )

        model_list.append(copy.deepcopy(model))
        logging.info(
            "Prepare %s-th target model costs %s seconds ",
            split,
            time.time() - baseline_time,
        )

        print(50 * "-")

        model_idx = split

        if save_model:
            if dataset.dataset_type == 'text':
                model.save_pretrained(f"{log_dir}/model_{model_idx}")
                trainer.tokenizer.save_pretrained(f"{log_dir}/model_{model_idx}")
                meta_data["model_path"] = f"{log_dir}/model_{model_idx}"
            elif configs["model_name"] == 'catboost':
                    meta_data["best_params"] = best_params
                    model.save_model(f"{log_dir}/model_{model_idx}.model")
                    meta_data["model_path"] = f"{log_dir}/model_{model_idx}.model"
            elif configs["model_name"] in ['lightgbm', "svm", "lr"]:
                meta_data["best_params"] = best_params
                joblib.dump(model,f"{log_dir}/model_{model_idx}.pkl")
                meta_data["model_path"] = f"{log_dir}/model_{model_idx}.pkl"
            else:
                with open(f"{log_dir}/model_{model_idx}.pkl", "wb") as f:
                    pickle.dump(model.state_dict(), f)
                meta_data["model_path"] = f"{log_dir}/model_{model_idx}.pkl"
            meta_data["train_split"] = data_split["split"][split]["train"]
            meta_data["test_split"] = data_split["split"][split]["test"]
            meta_data["audit_split"] = data_split["split"][split]["audit"]
            meta_data["num_train"] = len(data_split["split"][split]["train"])
            meta_data["optimizer"] = configs["optimizer"]
            meta_data["batch_size"] = configs["batch_size"]
            meta_data["epochs"] = configs["epochs"]
            meta_data["model_name"] = configs["model_name"]
            meta_data["split_method"] = data_split["split_method"]
            meta_data["model_idx"] = model_idx
            meta_data["learning_rate"] = configs["learning_rate"]
            meta_data["weight_decay"] = configs["weight_decay"]
            meta_data["train_acc"] = train_acc
            meta_data["test_acc"] = test_acc
            meta_data["train_loss"] = train_loss
            meta_data["test_loss"] = test_loss
            meta_data["dataset"] = dataset_name

            model_metadata_dict["model_metadata"][model_idx] = meta_data

            target_model_idx_list.append(model_idx)
    return model_list, model_metadata_dict, target_model_idx_list
