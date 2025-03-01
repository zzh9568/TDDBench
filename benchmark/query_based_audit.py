import sys
import os
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import pickle


from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Union
from torchvision import transforms

from info import METHODS_ALIASES
from datasets import Dataset
from heapq import nlargest
from models import INPUT_OUTPUT_SHAPE
from dataset import get_dataloader, get_dataset_subset, get_batch
from core import prepare_models
from privacy_meter.model import PytorchModelTensor
from info import DOMAIN_SUPPORT_MODEL
from signals import GBDTNumpy, TextModelTensor, get_signal_model_name, _get_signals, _text_get_signals, get_signals
from utils import save_audit_results
from metric_ref_based_audit import metric_based_auditing
from nn_based_audit import _prepare_dataloaders, nn_based_auditing

sys.path.insert(0, "../basic/extra_baselines")
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import PyTorchClassifier
from art.utils import compute_success

def _generate_neighbours(text, repeat_num):
    '''
    Get text's neighbours with the assistance of BERT

    Reference: Membership Inference Attacks against Language Models via Neighbourhood Comparison
    '''

    from transformers import BertForMaskedLM, BertTokenizer
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    model_path = f"../huggingface/models/google-bert/bert-base-uncased"
    search_tokenizer = AutoTokenizer.from_pretrained(model_path)
    search_model = AutoModelForMaskedLM.from_pretrained(model_path).to(torch.device("cuda"))
    token_dropout = torch.nn.Dropout(p=0.7)

    text_tokenized = search_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to(torch.device("cuda"))
    #original_text = search_tokenizer.batch_decode(text_tokenized)[0]

    candidate_scores = dict()
    replacements = dict()

    for target_token_index in list(range(len(text_tokenized[0,:])))[1:]:

        target_token = text_tokenized[0,target_token_index]
        embeds = search_model.bert.embeddings(text_tokenized)
        embeds = torch.cat((embeds[:,:target_token_index,:], token_dropout(embeds[:,target_token_index,:]).unsqueeze(dim=0), embeds[:,target_token_index+1:,:]), dim=1)
        
        token_probs = torch.softmax(search_model(inputs_embeds=embeds).logits, dim=2)

        original_prob = token_probs[0,target_token_index, target_token]

        top_probabilities, top_candidates = torch.topk(token_probs[:,target_token_index,:], 10, dim=1)

        for cand, prob in zip(top_candidates[0], top_probabilities[0]):
            if not cand == target_token:

                alt = torch.cat((text_tokenized[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to(torch.device("cuda")), text_tokenized[:,target_token_index+1:]), dim=1)
                #alt_text = search_tokenizer.batch_decode(alt)[0]
                alt_text = search_tokenizer.batch_decode(alt[:,1:-1])[0]
                candidate_scores[alt_text] = prob/(1-original_prob)
                replacements[(target_token_index, cand)] = prob/(1-original_prob)

    #highest_scored_texts = nlargest(100, candidate_scores, key = candidate_scores.get)
    highest_scored_texts = nlargest(repeat_num, candidate_scores, key = candidate_scores.get)
    return highest_scored_texts

def generate_neighbours(
    dataset: torch.utils.data.Dataset,
    repeat_num: int=10,
):
    dataset_name = dataset.dataset_name
    select_data_index = dataset.select_data_index
    save_dataset_pool_path = f"../data/text/dataset_pool/bert/{dataset_name}.pkl"
    if os.path.exists(save_dataset_pool_path):
        with open(save_dataset_pool_path, "rb") as f:
            save_dataset_pool = pickle.load(f)
        f.close()
        dataset_pool = []
        for i in range(repeat_num+1):
            new_dataset = save_dataset_pool[i].select(indices=select_data_index)
            new_dataset.dataset_type = save_dataset_pool[i].dataset_type
            dataset_pool.append(new_dataset)
    else:
        texts = dataset["text"]
        labels = dataset["label"]
        aug_texts = [[] for _ in range(repeat_num)]
        for text in tqdm(texts):
            neighbors = _generate_neighbours(text, repeat_num)
            for i in range(repeat_num):
                aug_texts[i].append(neighbors[i])
        dataset_pool = [copy.deepcopy(dataset)]
        for i in range(repeat_num):
            new_dataset = Dataset.from_dict(
                {
                    "text":aug_texts[i], 
                    "label": copy.deepcopy(labels)
                }
            )
            new_dataset.dataset_type = dataset.dataset_type
            dataset_pool.append(new_dataset)
    return dataset_pool

def prepare_signals(
    data_pool: List,
    targets: np.array,
    model_list: torch.nn.Module,
    model_name: str, 
    configs: Dict,
    signal_name: str,
):
    """ Prepare signals for each dataset in the data pool.

    Returns:
        all_signals: signals of data pool based on the model_list
    """

    assert signal_name in ["loss", "correctness"], f"We do not support signal {signal_name} in query based auditing."
    device, batch_size = configs["audit"]["device"], configs["audit"]["audit_batch_size"]
    signal_model_name = get_signal_model_name(model_name)
    
    all_signals = []
    for model in model_list:
        if signal_model_name == "PytorchModel":
            signal_model = PytorchModelTensor(
                model_obj=model,
                loss_fn=torch.nn.CrossEntropyLoss(),
                device=device,
                batch_size=batch_size,
            )
        elif signal_model_name == "GBDT":
            signal_model = GBDTNumpy(          
                model_obj=model,
                loss_fn=torch.nn.CrossEntropyLoss(),
                model_name=model_name)
        elif signal_model_name == "TextModel":
            signal_model = TextModelTensor(
                model_obj=model,
                loss_fn=torch.nn.CrossEntropyLoss(),
                device=device,
                batch_size=batch_size,
            )
        else:
            raise ValueError(
                f"The {signal_model_name} is not supported."
            )

        signals = []
        if signal_model_name == "TextModel":
            for datas in data_pool:
                pro_dataset = get_batch(datas, model_name)
                if signal_name == "loss":
                    signals.append(_text_get_signals(signal_model, pro_dataset, signal_name))
                elif signal_name == "correctness":
                    signals.append(-_text_get_signals(signal_model, pro_dataset, signal_name))
                else:
                    raise ValueError(
                        f"The {signal_name} is not supported."
                    )  
        elif signal_model_name in ["GBDT", "PytorchModel"]:
            for datas in data_pool:
                if signal_name == "loss":
                    signals.append(_get_signals(signal_model, datas, targets, signal_name))
                elif signal_name == "correctness":
                    signals.append(-_get_signals(signal_model, datas, targets, signal_name))
                else:
                    raise ValueError(
                        f"The {signal_name} is not supported."
                    )
        else:
            raise ValueError(
                f"The {signal_model_name} is not supported."
            )
        all_signals.append(signals)
    return all_signals

######################################################################################
################################  Noise based auditing  ##############################
######################################################################################
def add_noise(
    target_dataset: torch.utils.data.Dataset,
    dataset_type: str="image",
    repeat_num: int=10,
    noise_magnitude: float = 0.01,
):
    """
    Add noise to each example in the dataset to create a data pool.
    """
    # assert type(target_batch) != Dict and dataset_type in ["tabular", "image"], "Only supports image and tabular data."
    print(f"The process to add noise is repeated {repeat_num} times for each sample.")
    if dataset_type == "text":
        data_pool = generate_neighbours(target_dataset, repeat_num)
    else:
        target_batch = get_batch(target_dataset, model_name="none")
        datas, targets = target_batch
        datas = copy.deepcopy(datas)
        targets = copy.deepcopy(targets)

        data_pool = [datas]

        # TODO: data type setting
        if set(np.unique(datas.numpy())) == set([0,1]):
            data_type = "discrete"
        else:
            data_type = "continuous"

        if dataset_type == "image":
            feature_scale = (torch.max(datas[0]) - torch.min(datas[0])) / 2
            for _ in range(repeat_num):
                noise = np.array(np.random.normal(0, noise_magnitude * feature_scale, size=datas.shape), dtype=np.float32)
                data_pool.append(
                    copy.deepcopy(datas+torch.Tensor(noise))
                )

        elif data_type == "continuous":
            feature_scale = (torch.max(datas,0).values - torch.min(datas,0).values) / 2
            feature_scale = feature_scale.numpy()
            num_examples, num_features = datas.shape
            assert num_features == len(feature_scale), "The number of features does not match."
            for _ in range(repeat_num):
                noise = np.zeros_like(datas)
                for i in range(num_features):
                    noise[:,i] = np.array(np.random.normal(0, noise_magnitude * feature_scale[i], size=num_examples), dtype=np.float32)
                data_pool.append(
                    copy.deepcopy(datas+torch.Tensor(noise))
                )

        elif data_type == "discrete":
            assert len(datas.shape) == 2, "The shape of the entered tabular data is incorrect."
            # assert set(np.unique(datas.numpy())) == set([0,1]), "The value of the entered tabular data is incorrect."
            
            num_examples, num_features = datas.shape
            for _ in range(repeat_num):
                new_datas = copy.deepcopy(datas)
                for i in range(num_examples):
                    flip_index = np.random.choice(
                        num_features,
                        size=int(num_features*noise_magnitude),
                        replace=False,
                    )
                    new_datas[i][flip_index] = 1 - new_datas[i][flip_index]
                data_pool.append(new_datas)

        else:
            raise NotImplementedError
    print('Noise generation and addition ends.')
    return data_pool

def compare_signal_process(signals, dataset_type):
    assert dataset_type in ["tabular", "image", "text"]
    for i in range(1,len(signals)):
        signals[i,:] = signals[i,:]-signals[0,:]
    if dataset_type == "text":
        membership_signal = -np.sum(signals[1:,:], 0)/(len(signals)-1)
    else:
        membership_signal = -np.sum(signals[1:,:] > 0, 0)/(len(signals)-1)
    return membership_signal

def compare_based_auditing(
    data_pool: List,
    dataset_type: str,
    targets: np.array,
    target_model: torch.nn.Module, 
    membership: np.array,
    configs: Dict,
    signal_name: str="loss",
):  
    target_signals = prepare_signals(
        data_pool=data_pool,
        targets=targets,
        model_list=[target_model],
        model_name=configs["audit"]["target_model"],
        configs=configs,
        signal_name=signal_name,
    )
    target_signals = np.array(target_signals)
    target_signal = compare_signal_process(target_signals[0], dataset_type)

    fpr_list, tpr_list, roc_auc = metric_based_auditing(
        alg="target", 
        membership=membership, 
        target_signal=target_signal, 
        in_signals=None, 
        out_signals=None, 
        population_signals=None,
    )
    return fpr_list, tpr_list, roc_auc

######################################################################################
#########################  Data Augmentation based auditing  #########################
######################################################################################
def data_augmentation(
    dataset: Tuple,
    dataset_type: str="image",
    repeat_num: int=10,
    rotation_step: float = 0.1,
    cut_ratio: float = 0.01,
):
    #assert type(batch) != Dict, "Only supports image and tabular data."

    print(f"The data augmentation process is repeated {repeat_num} times for each sample.")
    if dataset_type == "text":
        data_pool = generate_neighbours(dataset, repeat_num)
    else:
        batch = get_batch(dataset, model_name="none")
        datas, targets = batch
        datas = copy.deepcopy(datas)
        targets = copy.deepcopy(targets)

        data_pool = [datas]
        if dataset_type == "image":
            rotation_zone = np.arange(1, 15+rotation_step, rotation_step)
            rotations = np.hstack((-rotation_zone, rotation_zone))
            select_rotations = np.random.choice(rotations, repeat_num, replace=False)
            for rotation in select_rotations:
                transform = transforms.RandomAffine((rotation,rotation))
                data_pool.append(
                    copy.deepcopy(transform(datas))
                )

        elif dataset_type == "tabular":
            '''
            The possible data augmentation strategies include cutmix, cutout, mixup, and we currently support cutmix for discrete data.
            '''
            
            assert len(datas.shape) == 2, 'tabular data shape error'
            
            num_examples, num_features = datas.shape
            for _ in range(repeat_num):
                new_datas = copy.deepcopy(datas)
                for i in range(num_examples):
                    cutmix_index = np.random.choice(
                        num_features,
                        size=int(num_features*cut_ratio),
                        replace=False
                    )
                    cutmix_sample_id = np.random.choice(num_examples, size=1)[0]
                    new_datas[i][cutmix_index] = new_datas[cutmix_sample_id][cutmix_index]
                data_pool.append(new_datas)

        else:
            raise NotImplementedError
    print('Data augmentation ends.')
    return data_pool

def augmentation_based_auditing(
    target_data_pool: List,
    target_data_labels: np.array,
    target_model: torch.nn.Module,
    membership: np.array,
    reference_data_pool: List,
    reference_data_labels: np.array,
    reference_model: torch.nn.Module,
    reference_membership: np.array,
    configs: Dict,
    signal_name: str="correctness",
):
    target_signals = prepare_signals(
        data_pool=target_data_pool,
        targets=target_data_labels,
        model_list=[target_model],
        model_name=configs["audit"]["target_model"],
        configs=configs,
        signal_name=signal_name,
    )
    target_signals = np.array(target_signals[0])
    target_features = torch.tensor(target_signals).to(torch.float32).transpose(1,0)

    reference_signals = prepare_signals(
        data_pool=reference_data_pool,
        targets=reference_data_labels,
        model_list=[reference_model],
        model_name=configs["audit"]["reference_model"],
        configs=configs,
        signal_name=signal_name,
    )
    reference_signals = np.array(reference_signals[0])
    reference_features = torch.tensor(reference_signals).to(torch.float32).transpose(1,0)

    train_loader, val_loader, test_loader, num_features = _prepare_dataloaders(
        target_features=target_features,
        membership=membership,
        reference_features=reference_features, 
        reference_membership=reference_membership, 
        val_ratio=configs["audit"]["val_ratio"],
    )

    fpr_list, tpr_list, roc_auc = nn_based_auditing(
        alg="augmentation_based_auditing",
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader, 
        num_features=num_features,
    )
    return fpr_list, tpr_list, roc_auc
    
######################################################################################
################################  Boundary based auditing  ###########################
######################################################################################
def distance_to_boundary(
    target_dataset: torch.utils.data.Dataset,
    target_batch: Tuple,
    target_model: torch.nn.Module,
    nb_classes: int,
    norm_p: int=2,
):
    assert type(target_batch) != Dict, "Only supports image data."
    datas, targets = target_batch
    datas = copy.deepcopy(datas)
    targets = copy.deepcopy(targets)

    num_examples, example_shape = len(datas), datas[0].shape

    data_loader = torch.utils.data.DataLoader(
        target_dataset,
        batch_size=1, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )

    target_model = copy.deepcopy(target_model)
    target_model.eval()
    ARTclassifier = PyTorchClassifier(
        model=target_model,
        clip_values=(torch.min(datas), torch.max(datas)),
        loss=F.cross_entropy,
        input_shape=example_shape,
        nb_classes=nb_classes,
    )

    Lp_dist = []
    Attack = HopSkipJump(
        classifier=ARTclassifier, 
        targeted=False, 
        max_iter=50, 
        max_eval=10000
    )

    not_success_distance = torch.norm(
        torch.ones_like(datas[0]) * (torch.max(datas[0])-torch.min(datas[0])), 
        p=norm_p,
    )
    for idx, (data, target) in enumerate(data_loader): 
        logits = target_model(data.to(ARTclassifier._device))
        logits_predict = torch.argmax(logits,1)
        if logits_predict.item() != target.item():
            success = 1
            data_adv = data
        else:
            data_adv = Attack.generate(x=np.array(data)) 
            success = compute_success(ARTclassifier, data, [target.item()], data_adv)
            data_adv = torch.Tensor(data_adv) 
        print('-------------Training DataSize: {} current img index:{}---------------'.format(num_examples, idx))

        if success == 1:
            Lp_dist.append(torch.norm(data-data_adv, p=norm_p).item())
        else:
            Lp_dist.append(not_success_distance)
    Lp_dist = np.asarray(Lp_dist)
    return Lp_dist

def perturb_distance_to_boundary(
    target_dataset: torch.utils.data.Dataset,
    target_model: torch.nn.Module,
    configs: Dict,
):
    data_pool = add_noise(
        target_dataset=target_dataset,
        dataset_type=configs["data"]["dataset_type"],
        repeat_num=configs["audit"]["query_num"],
    )

    if configs["data"]["dataset_type"] != "text":
        target_batch = get_batch(target_dataset, model_name="none")
        _, target_dataset_label = target_batch
    else:
        target_dataset_label = torch.tensor(target_dataset['label'])

    target_signals = prepare_signals(
        data_pool=data_pool,
        targets=target_dataset_label,
        model_list=[target_model],
        model_name=configs["audit"]["target_model"],
        configs=configs,
        signal_name="correctness",
    )

    target_signals = np.array(target_signals[0])
    target_signal = np.mean(target_signals,0)
    return target_signal

def boundary_based_auditing(
    eva_dataset: torch.utils.data.Dataset,
    eva_membership: np.array,
    target_model: torch.nn.Module, 
    configs: Dict,
    norm_p: int=2,
    dataset_type: str="image",
):  
    '''
    Estimate distances from sample to model decision boundary -> greater distance means sample is a member of related model. Boundary-based auditing is time consuming, so we choose a few samples to evaluate its performance.
    '''
    if dataset_type == "image":
        eva_batch = get_batch(eva_dataset, model_name="none")
        nb_classes = INPUT_OUTPUT_SHAPE[configs["audit"]["dataset"]][1]
        Lp_dist = distance_to_boundary(
            target_dataset=eva_dataset,
            target_batch=eva_batch,
            target_model=target_model,
            nb_classes=nb_classes,
            norm_p=norm_p,
        )
    elif dataset_type == "tabular" or dataset_type == "text":
        Lp_dist = perturb_distance_to_boundary(
            target_dataset=eva_dataset,
            target_model=target_model,
            configs=configs,
        )
    else:
        raise ValueError(f"Dataset type {dataset_type} is not supported!")

    fpr_list, tpr_list, roc_auc = metric_based_auditing(
        alg="target",
        membership=eva_membership, 
        target_signal=-Lp_dist, 
        in_signals=None, 
        out_signals=None, 
        population_signals=None,
    )
    return fpr_list, tpr_list, roc_auc

######################################################################################
################################  Transfer based auditing  ###########################
######################################################################################

def _label_by_target_model(
    label_dataset: torch.utils.data.Dataset,
    prediction: np.array,
):
    if hasattr(label_dataset, "targets"):
        label_dataset.targets = prediction
    elif hasattr(label_dataset, "label"):
        label_dataset.label = prediction
    elif hasattr(label_dataset, "labels"):
        label_dataset.labels = prediction
    else:
        raise ValueError("label_dataset should have attribute target or label.")
    return label_dataset
    
def label_by_target_model(
    target_model: torch.nn.Module,
    target_model_name: str,
    dataset: torch.utils.data.Dataset,
    device: str="cuda:1",
):
    ''' 
    For transfer attack.
    '''
    print("Label the input dataset with the given model.")
    label_dataset = copy.deepcopy(dataset)

    if target_model_name in DOMAIN_SUPPORT_MODEL["text"]:
        batch = get_batch(label_dataset, target_model_name)
        signal_model = TextModelTensor(
            model_obj=target_model,
            loss_fn=torch.nn.CrossEntropyLoss(),
            device=device,
            batch_size=128,
        )
        confidence_vector = _text_get_signals(signal_model, batch, "confidence_vector")
        prediction = np.argmax(confidence_vector,1).astype(int)

        assert len(np.unique(prediction)) > 1, "Check that the signal model has undergone enough training."
        prediction = list(prediction)

        label_dataset = Dataset.from_dict(
            {
                "text":  copy.deepcopy(dataset["text"]),
                "label": prediction,
            }
        )

        label_dataset.dataset_type = dataset.dataset_type

    elif target_model_name not in ["catboost", "lightgbm", "lr"]:
        if target_model_name == "resnet50":
            data_loader = get_dataloader(
                label_dataset,
                batch_size=128,
                shuffle=False,
            )
        else:
            data_loader = get_dataloader(
                label_dataset,
                batch_size=512,
                shuffle=False,
            )
        target_model.eval()
        target_model.to(device)

        preds = []
        for data, _ in data_loader:
            output = target_model(data.to(device))
            pred = output.data.max(1, keepdim=True)[1]
            pred = pred.detach().cpu().numpy()
            preds.append(pred)
        label_dataset = _label_by_target_model(label_dataset, np.concatenate(preds).squeeze())

    else:
        label_dataset = _label_by_target_model(label_dataset, target_model.predict(label_dataset.data).squeeze())

    return label_dataset

def prepare_transfer_auditing_config(
    log_dir: str, 
    reference_data_index: np.array, 
    val_ratio: float=0.2,
):
    transfer_log_dir = f"{log_dir}/Query-transfer"
    Path(transfer_log_dir).mkdir(parents=True, exist_ok=True)

    model_metadata_list = {"model_metadata": {}}

    index_list = []
    keep = np.random.uniform(0, 1, size=(len(reference_data_index))) <= val_ratio
    index_list.append(
        {
            "train": reference_data_index[~keep],
            "test":  reference_data_index[keep],
            "audit": None,
        }
    )
    dataset_splits = {"split": index_list, "split_method": f"random_{1-val_ratio}"}
    return transfer_log_dir, model_metadata_list, dataset_splits

def transfer_based_auditing(
    target_model: torch.nn.Module,
    target_dataset: torch.utils.data.Dataset,
    dataset: torch.utils.data.Dataset,
    reference_data_index: np.array,
    membership: np.array,
    configs: Dict,
    report_dir: str,
):
    '''
    Proxy model is trained to mimic the target model and then be used to provide "confidence" signal.
    '''
    # Relabel the whole dataset based on target model's prediction
    label_dataset = label_by_target_model(
        target_model=target_model,
        target_model_name=configs["audit"]["target_model"],
        dataset=dataset,
        device=configs["audit"]["device"],
    )

    # Config for training proxy model
    t_log_dir, t_meta_dict, t_split = prepare_transfer_auditing_config(
        log_dir=report_dir,
        reference_data_index=reference_data_index, 
        val_ratio=configs['audit']['val_ratio'],
    )

    # Train proxy model in relabeled target dataset
    configs["train"]["model_name"] = configs["audit"]["reference_model"]
    print("Training the proxy model for transfer based auditing.")
    proxy_model_list, _, _ = prepare_models(
        model_ids=[0],
        model_metadata_dict=t_meta_dict,
        log_dir=t_log_dir,
        dataset_name=configs["audit"]["dataset"],
        dataset=label_dataset,
        data_split=t_split,
        configs=configs["train"],
        save_model=False,
    )
    configs["train"]["model_name"] = "none"

    # Generate "confidence" signal using proxy model and transfer-based MIA
    target_transfer_signals, _ = get_signals(
        model_list=proxy_model_list, 
        model_name=configs["audit"]["reference_model"], 
        signal_name="loss", 
        device=configs["audit"]["device"],
        batch_size=configs["audit"]["audit_batch_size"],
        target_dataset=target_dataset, 
        population_dataset=None, 
        get_population_signal=False,
    )

    fpr_list, tpr_list, roc_auc = metric_based_auditing(
        alg="target", 
        membership=membership, 
        target_signal=target_transfer_signals[0], 
        in_signals=None, 
        out_signals=None, 
        population_signals=None,
    )
    return fpr_list, tpr_list, roc_auc

######################################################################################
################################  Query based benchmark  #############################
######################################################################################
def query_based_benchmark(
    alg_list: List[str],
    target_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    target_dataset: torch.utils.data.Dataset,
    reference_dataset: torch.utils.data.Dataset,
    dataset: torch.utils.data.Dataset,
    target_data_index: np.array,
    reference_data_index: np.array,
    membership: np.array,
    reference_membership: np.array,
    configs: Dict,
    report_dir: str,
):
    #configs["data"]["data_type"] = "continuous"

    start_time = time.time()
    print(f"Query based auditing benchmark starts.")
    print(100 * "#")
    all_fpr_list = []
    all_tpr_list = []
    all_auc_list = []     
    membership_list = []
    all_alg_names = []

    # assert configs["data"]["dataset_type"] != "text", "Query-based auditing does not support text data now!"
    if configs["data"]["dataset_type"] != "text":
        target_batch = get_batch(target_dataset, model_name="none")
        reference_batch = get_batch(reference_dataset, model_name="none")
        _, target_dataset_label = target_batch
        _, reference_dataset_label = reference_batch
    else:
        target_dataset_label = torch.tensor(target_dataset['label'])
        reference_dataset_label = torch.tensor(reference_dataset['label'])

    target_dataset.dataset_name = configs["audit"]["dataset"]
    target_dataset.select_data_index = target_data_index

    reference_dataset.dataset_name = configs["audit"]["dataset"]
    reference_dataset.select_data_index = reference_data_index

    # Query-based data auditing
    for alg_full_name in alg_list:
        alg = alg_full_name.split('+')[1]
        alg_start_time = time.time()
        print(f"Runing auditing algorithm {METHODS_ALIASES[alg_full_name]}.")

        if alg == "noise":
            data_pool = add_noise(
                target_dataset=target_dataset,
                dataset_type=configs["data"]["dataset_type"],
                repeat_num=configs["audit"]["query_num"],
            )
            
            fpr_list, tpr_list, roc_auc = compare_based_auditing(
                data_pool=data_pool,
                dataset_type=configs["data"]["dataset_type"],
                targets=target_dataset_label,
                target_model=target_model, 
                membership=membership,
                configs=configs,
                signal_name="loss",
            )
            membership_list.append(membership)

        elif alg == "augmented":
            target_data_pool = data_augmentation(
                dataset=target_dataset,
                dataset_type=configs["data"]["dataset_type"],
                repeat_num=configs["audit"]["query_num"],
                rotation_step=0.1,
                cut_ratio=0.01,
            )
            reference_data_pool = data_augmentation(
                dataset=reference_dataset,
                dataset_type=configs["data"]["dataset_type"],
                repeat_num=configs["audit"]["query_num"],
                rotation_step=0.1,
                cut_ratio=0.01,
            )

            fpr_list, tpr_list, roc_auc = augmentation_based_auditing(
                target_data_pool=target_data_pool,
                target_data_labels=target_dataset_label,
                target_model=target_model,
                membership=membership,
                reference_data_pool=reference_data_pool,
                reference_data_labels=reference_dataset_label,
                reference_model=reference_model,
                reference_membership=reference_membership,
                configs=configs,
                signal_name="correctness",
            )
            membership_list.append(membership)

        elif alg == "adversarial" or alg == "adversarial_for_query_num":
            # For boundary-based audting
            boundary_eva_num = configs["audit"]["boundary_eva_num"]
            if boundary_eva_num >= len(target_data_index):
                eva_dataset = target_dataset
                eva_membership = membership
            else:
                eva_dataset = get_dataset_subset(
                    dataset=dataset, 
                    index=target_data_index[:boundary_eva_num],
                )
                eva_membership = membership[:boundary_eva_num]

                eva_dataset.dataset_name = configs["audit"]["dataset"]
                eva_dataset.select_data_index = target_data_index[:boundary_eva_num]

            fpr_list, tpr_list, roc_auc = boundary_based_auditing(
                eva_dataset=eva_dataset,
                eva_membership=eva_membership,
                target_model=target_model,
                configs=configs,
                norm_p=2,
                dataset_type=configs["data"]["dataset_type"] if alg == "adversarial" else "tabular",
            )
            membership_list.append(eva_membership)
        
        elif alg == "transfer":
            fpr_list, tpr_list, roc_auc = transfer_based_auditing(
                target_model=target_model,
                target_dataset=target_dataset,
                dataset=dataset,
                reference_data_index=reference_data_index,
                membership=membership,
                configs=configs,
                report_dir=report_dir,
            )
            membership_list.append(membership)

        else:
            raise ValueError(f"Algorithm {alg} is not supported.")
        
        # Save results
        all_fpr_list.append(fpr_list)
        all_tpr_list.append(tpr_list)
        all_auc_list.append(roc_auc)
        all_alg_names.append(alg_full_name)
        print(
            f"Algorithm {alg_full_name} costs {time.time()-alg_start_time:0.5f} seconds."
        )
        print(100 * "#")

    print(f"Query based auditing benchmark cost {time.time() - start_time:.5f} seconds.")
    print(100 * "#")

    save_audit_results(membership_list, all_fpr_list, all_tpr_list, all_alg_names, report_dir)
    return all_fpr_list, all_tpr_list, all_auc_list, membership_list

