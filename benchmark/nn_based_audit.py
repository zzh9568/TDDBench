import copy
import time
import numpy as np
import torch

from info import METHODS_ALIASES
from typing import List, Tuple, Dict, Union
from scipy.special import softmax
from sklearn.metrics import auc, roc_curve
from dataset import TabularDataset, get_dataloader, get_batch
from signals import get_signals
from utils import save_audit_results


class NNAudit(torch.nn.Module):
    """
    Inherits from torch.nn.Module class, an neural network (denoted as auditing model later) to predict whether the data is used by the target model based the prediction vector return by the target model.
    """

    def __init__(self, num_fea, hiddens = [64,32], device="cuda:1"):
        """Constructor

        Args:
            num_fea: The number of input features of the auditing model, that is, the length of the prediction vector returned by the target model.
            hiddens: The number of hidden units of auditing model's different layers.
            device: Indicate the device of the audit model.
        """

        # Initializes the parent model
        super(NNAudit, self).__init__()

        self.num_fea = num_fea
        self.hiddens = [num_fea] + hiddens + [2]
        self.layers = torch.nn.ModuleList()
        self.device = device
        for i in range(0, len(self.hiddens)-1):
            self.layers.append(torch.nn.Linear(self.hiddens[i], self.hiddens[i+1]))
            if i < len(self.hiddens)-2:
                self.layers.append(torch.nn.ReLU())

    def forward(self, batch):
        """Function to get the model output from a given input.

        Args:
            batch: Model input.

        Returns:
            Model output.
        """
        h = batch
        for i in range(0,len(self.layers)):
            h = self.layers[i](h)
        return h
    
    def fit(
        self, 
        train_loader: torch.utils.data.DataLoader, 
        val_loader: torch.utils.data.DataLoader=None, 
        early_stop_patience: int=30, 
        epochs: int=500,
        optim: str='Adam', 
        lr: float=0.001, 
        weight_decay: float=0, #1e-7
    ):
        """Train the model based on on the train loader. Evaluate and stop the training process based on the validation loader.

        Args:
            train_loader (torch.utils.data.DataLoader): Data loader for training.
            val_loader (torch.utils.data.DataLoader): Data loader for evaluation.
            early_stop_patience (int): When the number of rounds in which the performance on the validation set does not improve exceeds the patience value, the training is terminated early.
            optim (str): Optimizer for the given model. We support Adam and SGD.
            lr (float): Learning rate for training the audit model.
            weight_decay (float): Weight decay for training the audit model.

        """

        # Move the model to the device
        self.to(self.device)

        # Initialize the loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        if optim == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optim == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError
        
        # Initialize the track variable for early stop strategy
        val_check = True if early_stop_patience >= 0 else False
        best_val_acc, early_stop_track = 0, 0

        print(f"Train the audit model for nn based auditing.")
        # Loop over each epoch
        for epoch_idx in range(epochs):
            start_time = time.time()
            train_loss, train_acc = 0, 0

            # Set the PyTorch model to training mode
            self.train()

            for data, target in train_loader:
                # Move data to the device
                data   = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                # Cast target to long tensor
                target = target.long()

                # Set the gradients to zero
                optimizer.zero_grad(set_to_none=True)

                # Get the model output
                output = self.forward(data)

                # Calculate the loss
                loss = criterion(output, target)

                # Calculate the training accurancy based on prediction
                pred = output.data.max(1, keepdim=True)[1]
                train_acc += pred.eq(target.data.view_as(pred)).sum()
                
                # Perform the backward pass and take a step using optimizer
                loss.backward()
                optimizer.step()

                # Add the loss to the total loss
                train_loss += loss.item()

            print(f"Epoch: {epoch_idx+1}/{epochs} |", end=" ")
            print(f"Train Loss: {train_loss/len(train_loader):.8f} ", end=" ")
            print(f"Train Acc: {float(train_acc)/len(train_loader.dataset):.8f} ", end=" ")

            if val_loader is not None:
                # Set the PyTorch model to evaluation mode
                self.eval()

                # Disable gradient calculation to save memory
                with torch.no_grad():
                    val_loss, val_acc = 0, 0
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)

                        # Cast target to long tensor
                        target = target.long()

                        # Computing output and loss
                        output = self.forward(data)
                        val_loss += criterion(output, target).item()

                        # Computing accuracy
                        pred = output.data.max(1, keepdim=True)[1]
                        val_acc += pred.eq(target.data.view_as(pred)).sum()

                print(f"Val Loss: {val_loss/len(val_loader):.8f} ", end=" ")
                print(f"Val Acc: {float(val_acc)/len(val_loader.dataset):.8f} ", end=" ")

                # If the model performance is improved, save the current optimal model parameters. If the performance of the model is not improved, the value of the track is increased by 1.
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    early_stop_track = 0
                    best_parameters = copy.deepcopy(self.state_dict())
                else:
                    early_stop_track += 1

            print(f"One step uses {time.time() - start_time:.2f} seconds")
            if val_check and (early_stop_track > early_stop_patience):
                print("Early stopped. ACC didn't improve for {} epochs.".format(early_stop_patience))
                break

        if val_check and val_loader is not None: 
            print("Load best model parameters based on the validation set.")
            self.load_state_dict(best_parameters)

    def evaluate(self, loader):
        """Evaluate the model based on on the given loader.

        Args:
            loader (torch.utils.data.DataLoader): Data loader for evaluation.

        Returns:
            loss (float): The average loss of the model on the evaluation data.
            acc (float): The average accurancy of the model on the evaluation data. 
            audit_confidence (np.array(float)): The auditing confidence for predicting the audit data as member.
            membership (np.array(int)): The data's membership ground truth label.
        
        """
        self.eval()
        self.to(self.device)

        # Assigning variables for computing loss and accuracy
        loss, acc, criterion = 0, 0, torch.nn.CrossEntropyLoss()

        # Initialize confidence vector list and ground truth label (data's menbership) in the audit process.
        audit_confidence = [] 
        membership = []

        # Disable gradient calculation to save memory
        with torch.no_grad():
            for data, target in loader:
                # Moving data and target to the device
                data, target = data.to(self.device), target.to(self.device)

                # Cast target to long tensor
                target = target.long()

                # Computing output and loss
                output = self.forward(data)
                loss += criterion(output, target).item()

                # Save confidence vector and ground truth label
                confidence = softmax(output.detach().cpu().numpy(), axis=1)
                audit_confidence.append(confidence[:, 1])
                membership.append(target.detach().cpu().numpy().astype(bool))

                # Computing accuracy
                pred = output.data.max(1, keepdim=True)[1]
                acc += pred.eq(target.data.view_as(pred)).sum()

        # Averaging the losses
        loss /= len(loader)

        # Calculating accuracy
        acc = float(acc) / len(loader.dataset)

        # Move model back to CPU
        self.to("cpu")

        # Return loss and accuracy
        print(f"Test Loss: {loss:.8f} ", end=" ")
        print(f"Test Acc: {acc:.8f} ")

        audit_confidence = np.concatenate(audit_confidence)
        membership = np.concatenate(membership)
        return loss, acc, audit_confidence, membership

def prepare_confidence_vectors(
    target_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    target_dataset: torch.utils.data.Dataset,
    reference_dataset: torch.utils.data.Dataset,
    configs: Dict,
):
    """Prepare confidence vectors as audit model's input features.

    Args:
        target_model (torch.nn.Module): The target model to be audited. We support pytorch models as input.
        reference_model (torch.nn.Module): The reference model to help generate audit model's training set.
        target_dataset (torch.utils.data.Dataset): The audit/target data. # Dict for text data and Tuple (eg. (images, labels)) for other data.
        reference_dataset (torch.utils.data.Dataset): The reference data, which may be target/subtarget/shadow/sub_shadow dataset.
        configs (Dict): Target model name, reference model name, device, and audit batch size.
    
    Returns:
        target_confidence_vectors (np.array[np.float]): Target model's confidence vector on the audit(target) data. It is an array with dimension [N, C], where N represents the number of audit data/samples, and C is the number of classes.  
        reference_confidence_vectors (np.array[np.float]): Reference model's confidence vector on the reference data.
    """
    target_model_name, reference_model_name = configs["audit"]["target_model"], configs["audit"]["reference_model"]
    device, batch_size = configs["audit"]["device"], configs["audit"]["audit_batch_size"]

    target_confidence_vectors, _ = get_signals(
        model_list=[target_model], 
        model_name=target_model_name, 
        signal_name="confidence_vector", 
        device=device, 
        batch_size=batch_size, 
        target_dataset=target_dataset,
        population_dataset=None, 
        get_population_signal=False,
    )
    target_confidence_vectors = target_confidence_vectors[0]

    reference_confidence_vectors, _ = get_signals(
        model_list=[reference_model], 
        model_name=reference_model_name, 
        signal_name="confidence_vector", 
        device=device, 
        batch_size=batch_size, 
        target_dataset=reference_dataset, 
        population_dataset=None, 
        get_population_signal=False,
    )
    reference_confidence_vectors = reference_confidence_vectors[0]

    return target_confidence_vectors, reference_confidence_vectors

def _prepare_mixed_signals(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    model_name: str,
    configs: Dict,  
):
    """Prepare different membership signals as audit model's input features.
    Reference: SoK: Reducing the Vulnerability of Fine-tuned Language Models to Membership Inference Attacks.

    Returns:
        features (np.array[np.float])
    """
    device, batch_size = configs["audit"]["device"], configs["audit"]["audit_batch_size"]
    if configs["data"]["dataset_type"] != "text":
        batch = get_batch(dataset, model_name)
        _, true_dataset_label = batch
    else:
        true_dataset_label = torch.tensor(dataset['label'])
    true_dataset_label = true_dataset_label.numpy()

    features = np.eye(configs["train"]["num_classes"])[true_dataset_label]
    
    for signal_name in ["loss", "mentr", "unnormalized_logit_difference", "rescaled_logits", "confidence_vector"]:
        feature, _ = get_signals(
            model_list=[model], 
            model_name=model_name, 
            signal_name=signal_name, 
            device=device, 
            batch_size=batch_size, 
            target_dataset=dataset, 
            population_dataset=None, 
            get_population_signal=False,
        )
        feature = feature[0]
        if signal_name != "confidence_vector":
            feature = np.expand_dims(feature,1)
        else:
            feature = np.argmax(feature,1)
            feature = np.expand_dims(feature,1)
        features = np.hstack((features, feature))
    return features

def prepare_mixed_signals(
    target_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    target_dataset: torch.utils.data.Dataset,
    reference_dataset: torch.utils.data.Dataset,
    configs: Dict,
):
    """Contact different membership signals as audit model's input features.
    Reference: SoK: Reducing the Vulnerability of Fine-tuned Language Models to Membership Inference Attacks.

    Args:
        target_model (torch.nn.Module): The target model to be audited. We support pytorch models as input.
        reference_model (torch.nn.Module): The reference model to help generate audit model's training set.
        target_dataset (torch.utils.data.Dataset): The audit/target data. # Dict for text data and Tuple (eg. (images, labels)) for other data.
        reference_dataset (torch.utils.data.Dataset): The reference data, which may be target/subtarget/shadow/sub_shadow dataset.
        configs (Dict): Target model name, reference model name, device, and audit batch size.
    
    Returns:
        target_features (np.array[np.float]): Target model's mixed signals on the audit(target) data. It is an array with dimension [N, C], where N represents the number of audit data/samples, and C is the number of features.  
        reference_features (np.array[np.float]): Reference model's mixed signals on the reference data.
    """
    target_model_name, reference_model_name = configs["audit"]["target_model"], configs["audit"]["reference_model"]

    target_features = _prepare_mixed_signals(
        model=target_model,
        dataset=target_dataset,
        model_name=target_model_name,
        configs=configs,  
    )
    reference_features = _prepare_mixed_signals(
        model=reference_model,
        dataset=reference_dataset,
        model_name=reference_model_name,
        configs=configs,  
    )

    return target_features, reference_features

def prepare_dataloaders(
    num_classes: int,
    alg: str, 
    target_dataset_label: np.array,
    membership: np.array,
    target_confidence_vectors: np.array,
    reference_dataset_label: np.array,
    reference_membership: np.array,
    reference_confidence_vectors: np.array,
    val_ratio: float=0, 
    topk: int=3,
):
    """Prepare the audit model's dataloaders.

    Args:
        alg (str): The algorithm name. In nn-based auditing algorithms, it represents different ways of preprocessing the confidence vector.
        target_dataset_label (np.array(int)): The target data's label (e.g. cat, dog, fish).
        membership (np.array(int)): Whether the target model used the target data.
        target_confidence_vectors (np.array[np.float]): Target model's confidence vector on the audit(target) data. It is an array with dimension [N, C], where N represents the number of audit data/samples, and C is the number of classes.  
        reference_dataset_label (np.array(int)): The reference data's label (e.g. cat, dog, fish).
        reference_membership (np.array(int)): Whether the reference model used the reference data.
        reference_confidence_vectors (np.array[np.float]): Reference model's confidence vector on the reference data.
        val_ratio: Percentage of validation sets in audit model's training process.
        topk: Top-k confidence will be used instead of the whole confidence vector in algorithm "topk".
        
    Returns:
        train_loader (torch.utils.data.DataLoader): Training data loader, which is generated from reference data.
        val_loader (torch.utils.data.DataLoader): Validation data loader, which is generated from reference data.
        test_loader (torch.utils.data.DataLoader): Test data loader, which is generated from audit(target) data.
        num_features (int): The number of features the audit model used.
    """
        
    assert val_ratio >= 0 and val_ratio <=1
    if alg == 'normal':
        target_features = target_confidence_vectors
        reference_features = reference_confidence_vectors
    elif alg == 'sorted':
        target_features = -np.sort(-target_confidence_vectors,1)
        reference_features = -np.sort(-reference_confidence_vectors,1)
    elif alg == 'topk':
        target_features = -np.sort(-target_confidence_vectors,1)[:,:topk]
        reference_features = -np.sort(-reference_confidence_vectors,1)[:,:topk]
    elif alg == 'normal_label':
        target_one_hot = np.eye(num_classes)[target_dataset_label]
        target_features = np.hstack((target_confidence_vectors,target_one_hot))
        reference_one_hot = np.eye(num_classes)[reference_dataset_label]
        reference_features = np.hstack((reference_confidence_vectors,reference_one_hot))
    else:
        raise ValueError(f"{alg} is not supported for nn-based auditing.")
    
    target_features = target_features.astype(np.float32)
    reference_features = reference_features.astype(np.float32)

    # membership = membership.astype(np.int32)
    # reference_membership = reference_membership.astype(np.int32)
    
    train_loader, val_loader, test_loader, num_features = _prepare_dataloaders(
        target_features=target_features,
        membership=membership,
        reference_features=reference_features, 
        reference_membership=reference_membership, 
        val_ratio=val_ratio,
    )
    return train_loader, val_loader, test_loader, num_features

def _prepare_dataloaders(
    target_features: np.array,
    membership: np.array,
    reference_features: np.array,
    reference_membership: np.array,
    val_ratio: float=0,
):
    """Prepare the audit model's dataloaders.

    Args:
        target_features (np.array[np.float]): The audit(target) data's input features in the audit model. It is an array with dimension [N, F], where N represents the number of audit data/samples, and F is the number of features.  
        membership (np.array(int)): Whether the target model used the target data.
        reference_features (np.array[np.float]): The reference data's input features in the audit model.
        reference_membership (np.array(int)): Whether the reference model used the reference data.
        val_ratio: Percentage of validation sets in audit model's training process.
        
    Returns:
        train_loader (torch.utils.data.DataLoader): Training data loader, which is generated from reference data.
        val_loader (torch.utils.data.DataLoader): Validation data loader, which is generated from reference data.
        test_loader (torch.utils.data.DataLoader): Test data loader, which is generated from audit(target) data.
        num_features (int): The number of features the audit model used.
    """
    
    num_features = len(target_features[0])
    target_dataset = TabularDataset(target_features, membership)
    reference_dataset = TabularDataset(reference_features, reference_membership)

    if val_ratio > 0:
        reference_val_indexes = np.random.choice(
            len(reference_dataset), int(len(reference_dataset)*val_ratio), replace=False
        )
        reference_train_indexes = np.setdiff1d(np.arange(len(reference_dataset)), reference_val_indexes)
        train_loader = get_dataloader(
            dataset=torch.utils.data.Subset(reference_dataset, reference_train_indexes),
            batch_size=128,
            shuffle=True,
        )
        val_loader = get_dataloader(
            dataset=torch.utils.data.Subset(reference_dataset, reference_val_indexes),
            batch_size=128,
            shuffle=False,
        )
    else:
        train_loader = get_dataloader(
            dataset=reference_dataset,
            batch_size=128,
            shuffle=True,
        )
        val_loader = None
    test_loader = get_dataloader(
        dataset=target_dataset,
        batch_size=256,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader, num_features

def nn_based_auditing(
    alg: str, 
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    num_features: int,
):
    """Neural network based data auditing algorithm.

    Args:
        alg (str): Auditing algorithm name.
        train_loader (torch.utils.data.DataLoader): Training data loader for the audit model.
        val_loader (torch.utils.data.DataLoader): Validation data loader for the audit model.
        test_loader (torch.utils.data.DataLoader): Test data loader for the audit model.
        num_features (int): The number of features the audit model used.
        
    Returns:
        fpr_list (np.array[np.float]): False positive rate of auditing results based on different decision thresholds.
        tpr_list (np.array[np.float]): True positive rate of auditing results based on different decision thresholds. 
        roc_auc (float): Area under TPR-FPR Curve.
    """
    start_time = time.time()

    # Train audit model to obtain audit results
    audit_model = NNAudit(num_fea=num_features)
    audit_model.fit(train_loader, val_loader)
    _, _, audit_confidence, membership  = audit_model.evaluate(test_loader)

    # Calculate auc, acc, and tpr@fpr based on audit results
    fpr_list, tpr_list, _ = roc_curve(membership, audit_confidence)
    acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)
    roc_auc = auc(fpr_list, tpr_list)
    tpr_at_low_fpr  = tpr_list[np.where(fpr_list < 0.001)[0][-1]]

    print(
        f"{alg} AUC: %.4f, Accuracy: %.4f, TPR@0.1%%FPR: %.4f, Time cost: %.4f"
        % (roc_auc, acc, tpr_at_low_fpr, time.time()-start_time)
    )
    return fpr_list, tpr_list, roc_auc

def nn_based_benchmark(
    alg_list: List[str],
    target_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    target_dataset: torch.utils.data.Dataset,
    reference_dataset: torch.utils.data.Dataset,
    membership: np.array,
    reference_membership: np.array,
    configs: Dict,
    report_dir: str,
):  
    """Metric based auditing benchmark.

    Args:
        alg_list (List[str]): The data auditing algorithms list.
        target_model (torch.nn.Module): The target model to be audited. We support pytorch models as input.
        reference_model_list (List[torch.nn.Module]): The reference model list to help compute in signals and out signals.
        target_dataset (torch.utils.data.Dataset): The audit/target data. # Dict for text data and Tuple (eg. (images, labels)) for other data.
        reference_dataset (torch.utils.data.Dataset): The reference data, which may be target/subtarget/shadow/sub_shadow dataset.
        target_data_indexes (one-dimensional np.array(int)): The target data's index in the whole dataset.
        reference_data_indexes (two-dimensional np.array(int)): The reference data's index in the whole dataset.
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
    print(f"NN based auditing benchmark starts.")
    print(100 * "#")
    all_fpr_list = []
    all_tpr_list = []
    all_auc_list = []     
    membership_list = []
    all_alg_names = []

    if configs["data"]["dataset_type"] != "text":
        target_batch = get_batch(target_dataset, model_name="none")
        reference_batch = get_batch(reference_dataset, model_name="none")
        _, target_dataset_label = target_batch
        _, reference_dataset_label = reference_batch
    else:
        target_dataset_label = torch.tensor(target_dataset['label'])
        reference_dataset_label = torch.tensor(reference_dataset['label'])
        
    target_dataset_label = target_dataset_label.numpy()
    reference_dataset_label = reference_dataset_label.numpy()

    val_ratio, nn_topk =configs["audit"]["val_ratio"], configs["audit"]["nn_topk"]

    # Prepare confidence vectors for neural network based auditing algorithms.
    if len(alg_list) > 0:
        target_confidence_vectors, reference_confidence_vectors = prepare_confidence_vectors(
            target_model=target_model,
            reference_model=reference_model,
            target_dataset=target_dataset,
            reference_dataset=reference_dataset,
            configs=configs,
        )
        print(
            f"Prepare the confidence vectors for nn-attack costs {time.time() - start_time:.5f} seconds."
        )

        # NN-based data auditing
        for alg_full_name in alg_list:
            alg = alg_full_name.split('+')[1]
            alg_start_time = time.time()
            print(f"Runing auditing algorithm {METHODS_ALIASES[alg_full_name]}.")

            if alg == "mix":
                target_mix_features, reference_mix_features = prepare_mixed_signals(
                    target_model=target_model,
                    reference_model=reference_model,
                    target_dataset=target_dataset,
                    reference_dataset=reference_dataset,
                    configs=configs,
                )

                train_loader, val_loader, test_loader, num_features = prepare_dataloaders(
                    num_classes=configs["train"]["num_classes"],
                    alg="normal", 
                    target_dataset_label=target_dataset_label,
                    membership=membership,
                    target_confidence_vectors=target_mix_features,
                    reference_dataset_label=reference_dataset_label,
                    reference_membership=reference_membership,
                    reference_confidence_vectors=reference_mix_features,
                    val_ratio=val_ratio, 
                    topk=nn_topk,
                )

            else:
                # Prepare data loaders for the audit model which can distinguish between members and non-members.
                train_loader, val_loader, test_loader, num_features = prepare_dataloaders(
                    num_classes=configs["train"]["num_classes"],
                    alg=alg, 
                    target_dataset_label=target_dataset_label,
                    membership=membership,
                    target_confidence_vectors=target_confidence_vectors,
                    reference_dataset_label=reference_dataset_label,
                    reference_membership=reference_membership,
                    reference_confidence_vectors=reference_confidence_vectors,
                    val_ratio=val_ratio, 
                    topk=nn_topk,
                )

            # Audit model training and inference
            fpr_list, tpr_list, roc_auc = nn_based_auditing(
                alg=alg,
                train_loader=train_loader, 
                val_loader=val_loader, 
                test_loader=test_loader, 
                num_features=num_features,
            )

            # Save results
            all_fpr_list.append(fpr_list)
            all_tpr_list.append(tpr_list)
            all_auc_list.append(roc_auc)
            membership_list.append(membership)
            all_alg_names.append(alg_full_name)
            print(
                f"Algorithm {METHODS_ALIASES[alg_full_name]} costs {time.time()-alg_start_time:0.5f} seconds."
            )
            print(100 * "#")

    print(f"Neural network based auditing benchmark cost {time.time() - start_time:.5f} seconds.")
    print(100 * "#")

    save_audit_results(membership_list, all_fpr_list, all_tpr_list, all_alg_names, report_dir)
    return all_fpr_list, all_tpr_list, all_auc_list, membership_list