"""This file contains functions for loading the dataset"""
import os
import pickle
from ast import List
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image
from ucimlrepo import fetch_ucirepo
from tqdm import tqdm
from torch.utils.data import Dataset
#from torchvision.datasets import VisionDataset
from medmnist.dataset import (PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST,
                                BreastMNIST, BloodMNIST, TissueMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST,
                                OrganMNIST3D, NoduleMNIST3D, AdrenalMNIST3D, FractureMNIST3D, VesselMNIST3D, SynapseMNIST3D)
from medmnist.dataset import MedMNIST2D
from medmnist.info import INFO

from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
from util import get_text_model_path

DATASET_TYPE = {
    "cifar10": "image", "cifar100": "image", "lfw": "image", "celeba": "image",\
    "pathmnist": "image", "octmnist": "image", "chestmnist": "image", 'breastmnist': "image", \
    "dermamnist": "image", "retinamnist": "image", "bloodmnist": "image", "organamnist": "image", \
    "purchase100": "tabular", "texas100": "tabular", "location": "tabular", "credit": "tabular", \
    "abalone": "tabular", "student": "tabular", "iris": "tabular", "breast_cancer": "tabular", \
    "cancer": "tabular", "adult": "tabular", "diabete": "tabular", "insurance": "tabular", \
    "tweet_eval_hate": "text", "rotten_tomatoes": "text", "ag_news": "text", "imdb": "text", \
    "medical_institutions": "text", "medical_meadow": "text", "twitter_sentiment": "text", \
    "ecthr_articles": "text", "contract_types": "text",
    "scotus": "text", "cola": "text", "sst2": "text",
}

DATASET_INFO = {
    "pathmnist": "PathMNIST", "octmnist": "OCTMNIST", "chestmnist": "ChestMNIST", 'breastmnist': "BreastMNIST",
    "dermamnist": "DermaMNIST", "retinamnist": "RetinaMNIST", "bloodmnist": "BloodMNIST", "organamnist": "OrganAMNIST",
}

class MedMNIST_All(MedMNIST2D):
    """MedMNIST which contains all data including train, val and test"""

    def __init__(self, name, root, imgs, labels, transform, size=None, \
                 as_rgb=False, target_transform=None, split='all'):
        self.info = INFO[name]
        self.imgs = imgs
        self.labels = labels
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.as_rgb = as_rgb

        if (size is None) or (size == 28):
            self.size = 28
            self.size_flag = ""
        else:
            assert size in self.available_sizes
            self.size = size
            self.size_flag = f"_{size}"

    def __len__(self):
        assert self.imgs.shape[0] == self.info["n_samples"]['train'] + \
            self.info["n_samples"]['val'] + self.info["n_samples"]['test']
        return self.imgs.shape[0]

class ImageDataset(Dataset):
    """Students Performance dataset."""

    def __init__(self, X, y, transform):
        """Initializes instance of class StudentsPerformanceDataset.
        Args:
            csv_file (str): Path to the csv file with the students data.
        """
        self.data = X
        self.targets = y
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target
    
class TabularDataset(Dataset):
    """Students Performance dataset."""

    def __init__(self, X, y):
        """Initializes instance of class StudentsPerformanceDataset.
        Args:
            csv_file (str): Path to the csv file with the students data.
        """
        self.data = X
        self.targets = y

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        return [self.data[idx], self.targets[idx]]


def get_dataset(dataset_name: str, data_dir: str):
    """Load the dataset from the pickle file or download it from the internet.
    Args:
        dataset_name (str): Dataset name
        data_dir (str): Indicate the log directory for loading the dataset

    Raises:
        NotImplementedError: Check if the dataset type has been implemented.

    Returns:
        torchvision.datasets: Whole dataset.
    """
    dataset_name = dataset_name.lower()
    dataset_type = DATASET_TYPE[dataset_name]
    path = f"{data_dir}/{dataset_type}/{dataset_name}"
    if dataset_type != 'text':
        Path(path).mkdir(parents=True, exist_ok=True)
    if dataset_name in ['pathmnist', 'octmnist', 'breastmnist', 'chestmnist']:
        Path(f"{path}/{dataset_name}").mkdir(parents=True, exist_ok=True)
    print("Path to the data:", path)

    if dataset_type == 'text':
        if os.path.exists(f"{path}"):
            all_data = load_from_disk(f"{path}")
            print(f"Load data from {path}.")
        else:
            all_data = get_text_dataset(dataset_name, data_dir)
            all_data.save_to_disk(f"{path}")
    else:
        if os.path.exists(f"{path}/{dataset_name}.pkl"):
            with open(f"{path}/{dataset_name}.pkl", "rb") as file:
                all_data = pickle.load(file)
            print(f"Load data from {path}/{dataset_name}.pkl")
        else:
            if dataset_type == 'image':
                all_data = get_image_dataset(dataset_name, data_dir)
            elif dataset_type == 'tabular':
                all_data = get_tabular_dataset(dataset_name, data_dir)
            else:
                raise NotImplementedError(f"{dataset_type} is not implemented")
    
    all_data.dataset_type = dataset_type
    if dataset_name == 'breast_cancer':
        all_data.data_type = 'continuous'
    print(f"the whole dataset size: {len(all_data)}")
    return all_data

def get_image_dataset(dataset_name: str, data_dir: str):
    """Download the image dataset from the internet and data processing.
    Args:
        dataset_name (str): Dataset name
        data_dir (str): Indicate the log directory for loading the dataset

    Raises:
        NotImplementedError: Check if the dataset has been implemented.

    Returns:
        torchvision.datasets: Whole dataset.
    """
    path = f"{data_dir}/image/{dataset_name}/{dataset_name}"
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    if dataset_name == "cifar10":
        all_data = torchvision.datasets.CIFAR10(
            root=path, train=True, download=True, transform=transform
        )
        test_data = torchvision.datasets.CIFAR10(
            root=path, train=False, download=True, transform=transform
        )
        all_features = np.concatenate([all_data.data, test_data.data], axis=0)
        all_targets = np.concatenate([all_data.targets, test_data.targets], axis=0)
        all_data.data = all_features
        all_data.targets = all_targets
    elif dataset_name == "cifar10-demo":
        all_data = torchvision.datasets.CIFAR10(
            root=path, train=True, download=True, transform=transform
        )
        test_data = torchvision.datasets.CIFAR10(
            root=path, train=False, download=True, transform=transform
        )
        all_features = np.concatenate([all_data.data, test_data.data], axis=0)
        all_targets = np.concatenate([all_data.targets, test_data.targets], axis=0)
        all_data.data = all_features[:1000]
        all_data.targets = all_targets[:1000]
    elif dataset_name == "cifar100":
        all_data = torchvision.datasets.CIFAR100(
            root=path, train=True, download=True, transform=transform
        )
        test_data = torchvision.datasets.CIFAR100(
            root=path, train=False, download=True, transform=transform
        )
        all_features = np.concatenate([all_data.data, test_data.data], axis=0)
        all_targets = np.concatenate([all_data.targets, test_data.targets], axis=0)
        all_data.data = all_features
        all_data.targets = all_targets
    elif dataset_name == "lfw":
        transform = transforms.Compose(
            [
                transforms.Resize(size=(32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        all_data = torchvision.datasets.LFWPeople(
            root=path, split='train', download=True, transform=transform
        )
        test_data = torchvision.datasets.LFWPeople(
            root=path, split='test', download=True, transform=transform
        )
        all_features = np.concatenate([all_data.data, test_data.data], axis=0)
        all_targets = np.concatenate([all_data.targets, test_data.targets], axis=0)
        all_data.data = all_features
        all_data.targets = all_targets
    elif dataset_name == "celeba":
        transform = transforms.Compose(
            [
                transforms.Resize(size=(32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        ori_data = torchvision.datasets.CelebA(
            root=path, split='all', download=True, transform=None,
        )
        sex_attr_index = ori_data.attr_names.index("Male")
        sex_attr = ori_data.attr[:,sex_attr_index:sex_attr_index+1]

        all_features = []
        for i in tqdm(range(len(ori_data))):
            all_features.append(torch.tensor(np.expand_dims(ori_data[i][0], -1)))
        all_features = torch.cat(all_features, -1)
        all_features = all_features.numpy()

        all_features = all_features.transpose(3,0,1,2)

        all_targets = sex_attr.squeeze().numpy()
        print(f"fearture shape: {all_features.shape}")
        all_data = ImageDataset(all_features, all_targets, transform=transform)
    elif dataset_name in ['pathmnist', 'octmnist', 'breastmnist', 'chestmnist', "dermamnist", "retinamnist", "bloodmnist", "organamnist"]:
        train_data = eval(DATASET_INFO[dataset_name])(root=path, split='train', download=True, transform=transform, as_rgb=True)
        val_data   = eval(DATASET_INFO[dataset_name])(root=path, split='val',   download=True, transform=transform, as_rgb=True)
        test_data  = eval(DATASET_INFO[dataset_name])(root=path, split='test',  download=True, transform=transform, as_rgb=True)
        all_features = np.concatenate([train_data.imgs,   val_data.imgs,   test_data.imgs],   axis=0)
        all_targets  = np.concatenate([train_data.labels, val_data.labels, test_data.labels], axis=0).squeeze()
        all_data = MedMNIST_All(name=dataset_name, root=path, as_rgb=True, imgs=all_features, labels=all_targets, transform=transform)
    else:
        raise NotImplementedError(f"{dataset_name} is not implemented")
    with open(f"{path}.pkl", "wb") as file:
        pickle.dump(all_data, file)
    print(f"Save data to {path}.pkl")
    return all_data


def get_tabular_dataset(dataset_name: str, data_dir: str):
    """Download the tabular dataset from the internet and data processing.
    Args:
        dataset_name (str): Dataset name
        data_dir (str): Indicate the log directory for loading the dataset

    Raises:
        NotImplementedError: Check if the dataset has been implemented.

    Returns:
        torchvision.datasets: Whole dataset.
    """
    path = f"{data_dir}/tabular/{dataset_name}/{dataset_name}"
    if dataset_name == "purchase100":
        # https://www.comp.nus.edu.sg/~reza/files/datasets.html
        if os.path.exists("../data/tabular/purchase/dataset_purchase"):
            df = pd.read_csv(
                "../data/tabular/purchase/dataset_purchase", header=None, encoding="utf-8"
            ).to_numpy()
            y = df[:, 0] - 1
            X = df[:, 1:].astype(np.float32)
            all_data = TabularDataset(X, y)
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            print(f"Save data to {path}.pkl")
        else:
            raise NotImplementedError(
                f"{dataset_name} is not installed correctly in ../data/purchase"
            )
        
    elif dataset_name == "texas100":
        # https://www.comp.nus.edu.sg/~reza/files/datasets.html
        if os.path.exists("../data/tabular/texas/100/feats"):
            X = pd.read_csv("../data/tabular/texas/100/feats", header=None, encoding="utf-8")
            X = X.to_numpy().astype(np.float32)
            y = pd.read_csv("../data/tabular/texas/100/labels", header=None, encoding="utf-8")
            y = y.to_numpy().reshape(-1) - 1
            all_data = TabularDataset(X, y)
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            print(f"Save data to {path}.pkl")
        else:
            raise NotImplementedError(
                f"{dataset_name} is not installed correctly in ../data/texas"
            )
        
    elif dataset_name == "adult":
        '''
        The adult dataset preprocess is referred to Paper: Practical Blind Membership Inference Attack via Differential Comparisons
        '''
        adult = fetch_ucirepo(id=2)

        df = adult.data.features.join(adult.data.targets)
        # Handle for Null Data
        df = df.fillna(np.nan)

        # Reformat Column We Are Predicting: 0 means less than 50K. 1 means greater than 50K.
        df['income']=df['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})

        # Fill Missing Category Entries
        df["workclass"] = df["workclass"].fillna("X")
        df["occupation"] = df["occupation"].fillna("X")
        df["native-country"] = df["native-country"].fillna("United-States")

        # Convert Sex value to 0 and 1
        df["sex"] = df["sex"].map({"Male": 0, "Female":1})

        # Create Married Column - Binary Yes(1) or No(0)
        df["marital-status"] = df["marital-status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')
        df["marital-status"] = df["marital-status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')
        df["marital-status"] = df["marital-status"].map({"Married":1, "Single":0})
        df["marital-status"] = df["marital-status"].astype(int)

        # Drop the data you don't want to use
        df.drop(labels=["workclass","fnlwgt", "education","occupation","relationship","race","native-country"], axis = 1, inplace = True)

        array = df.values
        X = array[:,0:-1].astype(np.float32)
        y = array[:,-1].astype(int)
        all_data = TabularDataset(X, y)
        with open(f"{path}.pkl", "wb") as file:
            pickle.dump(all_data, file)
        print(f"Save data to {path}.pkl")

    elif dataset_name == "student":
        student = fetch_ucirepo(id=697)
        df = student.data.features.join(student.data.targets)

        # Handle for Null Data
        df = df.fillna(np.nan)
        df['Target']=df['Target'].map({'Dropout': 0, 'Enrolled': 1, 'Graduate': 2})

        drop_columns = ['Marital Status', 'Application mode', 'Course', 'Nacionality']
        # Drop the data you don't want to use
        df.drop(labels=drop_columns, axis = 1, inplace = True)

        array = df.values
        X = array[:,0:-1].astype(np.float32)
        y = array[:,-1].astype(int)
        all_data = TabularDataset(X, y)
        with open(f"{path}.pkl", "wb") as file:
            pickle.dump(all_data, file)
        print(f"Save data to {path}.pkl")

    elif dataset_name == "iris":
        from sklearn.datasets import load_iris
        iris = load_iris()
        X,y = iris.data.astype(np.float32), iris.target
        all_data = TabularDataset(X, y)
        with open(f"{path}.pkl", "wb") as file:
            pickle.dump(all_data, file)
        print(f"Save data to {path}.pkl")

    elif dataset_name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        breast_cancer = load_breast_cancer()
        X,y = breast_cancer.data.astype(np.float32), breast_cancer.target
        all_data = TabularDataset(X, y)
        with open(f"{path}.pkl", "wb") as file:
            pickle.dump(all_data, file)
        print(f"Save data to {path}.pkl")

    else:
        raise NotImplementedError(f"{dataset_name} is not implemented")
    print(f"the whole dataset size: {len(all_data)}")
    return all_data


def get_text_dataset(dataset_name: str, data_dir: str):
    # Datasets can be downloaded from Huggingface
    if dataset_name == "tweet_eval_hate":
        data_disk_path = f"{data_dir}/text/raw_data/cardiffnlp/tweet_eval/hate"
    elif dataset_name == "rotten_tomatoes":
        data_disk_path = f"{data_dir}/text/raw_data/cornell-movie-review-data/rotten_tomatoes"
    elif dataset_name == "ag_news":
        data_disk_path = f"{data_dir}/text/raw_data/fancyzhx/ag_news"
    elif dataset_name == "medical_institutions":
        data_disk_path = f"{data_dir}/text/raw_data/blinoff/medical_institutions_reviews"
    elif dataset_name == "imdb":
        data_disk_path = f"{data_dir}/text/raw_data/imdb"
    elif dataset_name == "medical_meadow":
        data_disk_path = f"{data_dir}/text/raw_data/medical_meadow_health_advice"
    elif dataset_name == "twitter_sentiment":
        data_disk_path = f"{data_dir}/text/raw_data/Twitter_Sentiment"
    elif dataset_name == "scotus":
        data_disk_path = f"{data_dir}/text/raw_data/coastalcph/lex_glue/scotus"
    elif dataset_name == "cola":
        data_disk_path = f"{data_dir}/text/raw_data/nyu-mll/glue/cola"
    elif dataset_name == "sst2":
        data_disk_path = f"{data_dir}/text/raw_data/nyu-mll/glue/sst2"
    else:
        raise NotImplementedError(f"{dataset_name} is not implemented")
    
    '''
    The unsupervised part of the imdb dataset is unlabeled(label=-1), and the test part in cola and sst2 is unlabeled(label=-1).
    '''
    dataset = load_from_disk(data_disk_path)
    dataset_keys = list(dataset.keys())
    accept_keys = ["train", "validation", "test"]

    if dataset_name != "imdb":
        assert len(set(dataset_keys) - set(accept_keys)) == 0, f"Recheck the keys of DataDict {dataset_name}."
    if dataset_name in ["cola", "sst2"]:
        use_keys = list(set(dataset_keys) & set(["train", "validation"]))
    else:
        use_keys = list(set(dataset_keys) & set(accept_keys))
    subdataset_list = []
    for split_key in use_keys:
        subdataset_list.append(dataset[split_key])
    all_data = concatenate_datasets(subdataset_list)

    if dataset_name in ["cola", "sst2"]:
        all_data = all_data.rename_column("sentence", "text")
        #all_data = all_data.remove_columns("idx")
    return all_data

import copy
#padding for main.py; no padding for train_base_model.py
def text_dataset_preprocess(model_name, all_data, padding=False):
    all_data = copy.deepcopy(all_data)
    
    model_path = get_text_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    def preprocess_function_with_padding(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    if padding:
        pro_data = all_data.map(preprocess_function_with_padding, batched=True)
        pro_data = pro_data.remove_columns(["text"])
        pro_data = pro_data.rename_column("label", "labels")
        if "idx" in pro_data.features.keys():
            pro_data = pro_data.remove_columns(["idx"])
    else:
        pro_data = all_data.map(preprocess_function, batched=True)
    pro_data.set_format("torch")
    return pro_data

def get_dataset_subset(
    dataset: torchvision.datasets, index: List(int)
):
    """Get a subset of the dataset.

    Args:
        dataset (torchvision.datasets): Whole dataset.
        index (list): List of index.
        model_name (str): name of the model.
    """
    assert max(index) < len(dataset) and min(index) >= 0, "Index out of range"
    if dataset.dataset_type == 'text':
        sub_dataset = dataset.select(indices=index)
    else:
        sub_dataset = torch.utils.data.Subset(dataset, index)
    sub_dataset.dataset_type = getattr(dataset, "dataset_type", "none")
    sub_dataset.data_type = getattr(dataset, "data_type", "none")

    return sub_dataset

def get_batch(dataset: torchvision.datasets, model_name: str):
    if dataset is None:
        return None
    elif dataset.dataset_type == 'text':
        pro_dataset = text_dataset_preprocess(
            model_name=model_name, 
            all_data=dataset, 
            padding=True,
        )
    else:
        pro_dataset = dataset
    data_loader = get_dataloader(
        pro_dataset,
        batch_size=len(dataset),
        shuffle=False,
    )
    for batch in data_loader:
        pass
    return batch

class InfiniteRepeatDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]
    

def get_dataloader(
    dataset: torchvision.datasets,
    batch_size: int,
    shuffle: bool = True,
):  
    repeated_data = InfiniteRepeatDataset(dataset)
    return torch.utils.data.DataLoader(
        repeated_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=16,
    )
