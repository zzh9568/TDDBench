"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
import torch
from quantile_mia.image_QMIA.optimizer_utils import build_optimizer
from quantile_mia.image_QMIA.scheduler_utils import build_scheduler
from timm.utils import accuracy
#from quantile_mia.image_QMIA.torch_models import get_model
from torchmetrics.utilities.data import to_onehot
from quantile_mia.image_QMIA.train_utils import (
    gaussian_loss_fn,
    label_logit_and_hinge_scoring_fn,
    pinball_loss_fn,
    rearrange_quantile_fn,
)
from transformers import (
    AutoModelForImageClassification,
    ViTConfig,
    ViTForImageClassification,
)
from transformers import AutoModelForSequenceClassification
from catboost import CatBoostClassifier
import joblib

# base utilities
import sys
sys.path.insert(0, "../basic/")
sys.path.insert(0, "../basic/extra_baselines")
from models import get_model
import pickle
import copy
from typing import Any, NamedTuple, Optional, Tuple
from torch.utils.data import DataLoader
from dataset import get_dataset, get_dataset_subset
import numpy as np
from core import prepare_datasets_for_reference_in_attack
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

def get_optimizer_params(optimizer_params):
    "convenience function to add default options to optimizer params if not provided"
    # optimizer
    optimizer_params.setdefault("opt_type", "adamw")
    optimizer_params.setdefault("weight_decay", 0.0)
    optimizer_params.setdefault("lr", 1e-3)

    # scheduler
    optimizer_params.setdefault("scheduler", None)
    optimizer_params.setdefault("epochs", 100)  # needed for CosineAnnealingLR
    optimizer_params.setdefault("step_gamma", 0.1)  # decay fraction in step scheduler
    optimizer_params.setdefault(
        "step_fraction", 0.33
    )  # fraction of total epochs before step decay

    return optimizer_params

import copy
def get_batch(batch):
    if type(batch) == dict:
        '''
        Passing in the label to variables 'samples' and 'base_samples' will give an error, because the label of the quantile model is not the label of the original dataset (sizes do not match), we only need to input_id, that is, the inputed data.
        '''
        targets = batch['labels']
        samples = dict((k, v) for k, v in batch.items() if k in ['input_ids', 'attention_mask'])
        base_samples = dict((k[5:], v) for k, v in batch.items() if k in ['base_input_ids', 'base_attention_mask'])
    else:
        if len(batch) == 2:
            samples, targets = batch
            base_samples = samples
        else:
            samples, targets, base_samples = batch
    return samples, targets, base_samples

class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        torch.save(
            predictions,
            os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"),
        )


# Lightning wrapper for MIA/QR model
class LightningQMIA(pl.LightningModule):
    def __init__(
        self,
        dataset_name,
        architecture,
        base_architecture,
        num_base_classes,
        image_size,
        hidden_dims,
        freeze_embedding,
        # base_model_name_prefix,
        low_quantile,
        high_quantile,
        n_quantile,
        # cumulative_qr,
        optimizer_params,
        base_model_path=None,
        rearrange_on_predict=True,
        use_target_label=False,
        use_hinge_score=False,
        use_logscale=False,
        use_gaussian=False,
        return_mean_logstd=False,
        use_target_dependent_scoring=False,
        use_target_inputs=False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.use_target_dependent_scoring = use_target_dependent_scoring
        assert not (
            use_target_dependent_scoring and use_target_inputs
        ), "target_dependent scoring should not be used with use_target_inputs"

        self.use_target_inputs = use_target_inputs
        self.num_base_classes = num_base_classes
        self.base_n_outputs = 2 if use_gaussian else n_quantile
        if self.use_target_dependent_scoring:
            n_outputs = self.base_n_outputs * self.num_base_classes
        else:
            n_outputs = self.base_n_outputs

        model, base_model = model_setup(
            architecture=architecture,
            base_architecture=base_architecture,
            image_size=image_size,
            num_quantiles=n_outputs,
            num_base_classes=num_base_classes,
            hidden_dims=hidden_dims,
            freeze_embedding=freeze_embedding,
            base_model_path=base_model_path,
            extra_inputs=num_base_classes if self.use_target_inputs else None,
            dataset_name=dataset_name,
        )

        self.model = model
        self.base_model = base_model
        self.base_model_path = base_model_path
        self.use_gaussian = use_gaussian
        self.return_mean_logstd = return_mean_logstd

        if base_architecture not in ["catboost", "lightgbm"]:
            for parameter in self.base_model.parameters():
                parameter.requires_grad = False

        if use_logscale:
            self.QUANTILE = torch.sort(
                1
                - torch.logspace(
                    low_quantile, high_quantile, n_quantile, requires_grad=False
                )
            )[0].reshape([1, -1])
        else:
            self.QUANTILE = torch.sort(
                torch.linspace(
                    low_quantile, high_quantile, n_quantile, requires_grad=False
                )
            )[0].reshape([1, -1])

        if self.use_gaussian:
            self.loss_fn = gaussian_loss_fn
            self.target_scoring_fn = label_logit_and_hinge_scoring_fn
            self.rearrange_on_predict = False
        else:
            self.loss_fn = pinball_loss_fn
            self.target_scoring_fn = label_logit_and_hinge_scoring_fn
            self.rearrange_on_predict = rearrange_on_predict and not use_logscale
            if not use_target_label or not use_hinge_score:
                raise NotImplementedError

        optimizer_params.update(**kwargs)
        self.optimizer_params = get_optimizer_params(optimizer_params)

        self.validation_step_outputs = []

    def forward(
        self, samples, targets: torch.LongTensor = None
    ) -> torch.Tensor:
        if self.use_target_inputs:
            oh_targets = to_onehot(targets, self.num_base_classes)
            if type(samples) == dict:
                raise ValueError("Text model do not support use_target_inputs now")
            else:
                scores = self.model(samples, oh_targets)
            return scores

        if type(samples) == dict:
            scores = self.model(**samples).logits
        else:
            scores = self.model(samples)

        if self.use_target_dependent_scoring:
            oh_targets = to_onehot(targets, self.num_base_classes).unsqueeze(1)
            scores = (
                scores.reshape(
                    [
                        -1,
                        self.base_n_outputs,
                        self.num_base_classes,
                    ]
                )
                * oh_targets
            ).sum(-1)

        return scores

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        samples, targets, base_samples = get_batch(batch)
        scores = self.forward(samples, targets)
        target_score, target_logits = self.target_scoring_fn(
            base_samples, targets, self.base_model
        )
        loss = self.loss_fn(
            scores, target_score, self.QUANTILE.to(scores.device)
        ).mean()
        self.log("ptl/train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        samples, targets, base_samples = get_batch(batch)
        # print('VALIDATION STEP', self.model.training), print(self.base_model.training)
        scores = self.forward(samples, targets)

        if self.rearrange_on_predict and not self.use_gaussian:
            scores = rearrange_quantile_fn(
                scores, self.QUANTILE.to(scores.device).flatten()
            )


        target_score, target_logits = self.target_scoring_fn(
            base_samples, targets, self.base_model
        )

        loss = self.loss_fn(
            scores, target_score, self.QUANTILE.to(scores.device)
        ).mean()

        rets = {
            "val_loss": loss,
            "scores": scores,
            "targets": target_score,
        }
        self.validation_step_outputs.append(rets)
        return rets

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(
            [x["val_loss"] for x in self.validation_step_outputs]
        ).mean()
        targets = torch.concatenate(
            [x["targets"] for x in self.validation_step_outputs], dim=0
        )
        scores = torch.concatenate(
            [x["scores"] for x in self.validation_step_outputs], dim=0
        )

        self.validation_step_outputs.clear()  # free memory
        #self.log("ptl/val_loss", avg_loss, sync_dist=True, prog_bar=True)
        self.log("ptl/val_loss", avg_loss, sync_dist=False, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        samples, targets, base_samples = get_batch(batch)

        scores = self.forward(samples, targets)
        if self.rearrange_on_predict and not self.use_gaussian:
            scores = rearrange_quantile_fn(
                scores, self.QUANTILE.to(scores.device).flatten()
            )
        target_score, target_logits = self.target_scoring_fn(
            base_samples, targets, self.base_model
        )
        loss = self.loss_fn(scores, target_score, self.QUANTILE.to(scores.device))
        base_acc1, base_acc5 = accuracy(target_logits, targets, topk=(1, 5))

        if self.use_gaussian and not self.return_mean_logstd:
            # use torch distribution to output quantiles
            mu = scores[:, 0]
            log_std = scores[:, 1]
            scores = mu.reshape([-1, 1]) + torch.exp(log_std).reshape(
                [-1, 1]
            ) * torch.erfinv(2 * self.QUANTILE.to(scores.device) - 1).reshape(
                [1, -1]
            ) * math.sqrt(
                2
            )
            assert (
                scores.ndim == 2
                and scores.shape[0] == targets.shape[0]
                and scores.shape[1] == self.QUANTILE.shape[1]
            ), "inverse cdf quantiles have the wrong shape, got {} {} {}".format(
                scores.shape, targets.shape, self.QUANTILE.size()
            )

        return scores, target_score, loss, base_acc1, base_acc5

    def configure_optimizers(self):
        optimizer = build_optimizer(
            self.model,
            opt_type=self.optimizer_params["opt_type"],
            lr=self.optimizer_params["lr"],
            weight_decay=self.optimizer_params["weight_decay"],
        )
        interval = "epoch"

        lr_scheduler = build_scheduler(
            scheduler=self.optimizer_params["scheduler"],
            epochs=self.optimizer_params["epochs"],
            step_fraction=self.optimizer_params["step_fraction"],
            step_gamma=self.optimizer_params["step_gamma"],
            optimizer=optimizer,
            mode="min",
            lr=self.optimizer_params["lr"],
        )
        opt_and_scheduler_config = {
            "optimizer": optimizer,
        }
        if lr_scheduler is not None:
            opt_and_scheduler_config["lr_scheduler"] = {
                # REQUIRED: The scheduler instance
                "scheduler": lr_scheduler,
                "interval": interval,
                "frequency": 1,
                "monitor": "ptl/val_loss",
                "strict": True,
                "name": None,
            }

        return opt_and_scheduler_config


# Convenience function to create models and potentially load weights for base classifier
def model_setup(
    architecture,
    base_architecture,
    image_size,
    num_quantiles,
    num_base_classes,
    hidden_dims,
    freeze_embedding,
    base_model_path=None,
    extra_inputs=None,
    dataset_name='cifar10',
):
    # Get forward function of regression model
    model = get_quantile_model(
        architecture,
        num_quantiles,
        hidden_dims=hidden_dims,
        extra_inputs=extra_inputs,
        dataset_name=dataset_name,
    )

    ## Create base model, load params from pickle, then define the scoring function and the logit embedding function
    base_model = get_base_model(
        base_architecture, 
        num_base_classes,
        base_model_path,
        dataset_name,
    )
    return model, base_model

def get_text_model_path(model_name):
    if model_name == "roberta":
        model_path = f"../huggingface/models/FacebookAI/roberta-base"
    elif model_name == "flan-t5":
        model_path = f"../huggingface/models/google/flan-t5-base"
    elif model_name == "distilbert":
        model_path = f"../huggingface/models/distilbert/distilbert-base-uncased"
    elif model_name == "bert":
        model_path = f"../huggingface/models/google-bert/bert-base-uncased"
    elif model_name == "gpt2":
        model_path = f"../huggingface/models/openai-community/gpt2"
    elif model_name == "longformer":
        model_path = f"../huggingface/models/allenai/longformer-base-4096"
    else:
        raise ValueError(
            f"The {model_name} is not supported for the given dataset."
        )
    return model_path

def get_quantile_model(
    architecture,
    num_classes,
    hidden_dims=[],
    extra_inputs=None,
    dataset_name='cifar10',
):
    print(f"quantitle model, model_name: {architecture}, dataset name: {dataset_name}")
    if architecture == 'convnext-tiny-224':
        from transformers import AutoModelForImageClassification
        model_base = AutoModelForImageClassification.from_pretrained(
            "../models/facebook/convnext-tiny-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        model = HugginFaceTupleWrapper(
            model_base, hidden_dims=hidden_dims, extra_inputs=extra_inputs
        )
    elif architecture in ["roberta", "flan-t5", "distilbert", "bert", "gpt2", "longformer"]:
        model_path = get_text_model_path(architecture)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_classes)
    elif architecture in ["catboost", "lightgbm"]:
        raise ValueError(f"run tabular_quantile_gdbt in extra_audit.py when quantile_model is catboost or lightgbm!")
    else:
        model = get_model(architecture, dataset_name, num_classes)
    return model

def get_base_model(
    architecture,
    num_classes,
    base_model_path,
    dataset_name,
):
    print(f"base model, model_name: {architecture}, dataset name: {dataset_name}")
    #print(base_model_path)
    if architecture == 'convnext-tiny-224':
        raise ValueError(f"We do not support convnext-tiny-224 as base model!")
    elif architecture in ["roberta", "flan-t5", "distilbert", "bert", "gpt2", "longformer"]:
        base_model = AutoModelForSequenceClassification.from_pretrained(base_model_path, num_labels=num_classes)
    elif architecture == 'catboost':
        base_model = CatBoostClassifier()
        base_model.load_model(base_model_path)
    elif architecture == 'lightgbm':
        base_model = joblib.load(base_model_path)
    else:
        base_model = get_model(architecture, dataset_name, num_classes)
        with open(base_model_path, "rb") as file:
            model_weight = pickle.load(file)
        base_model.load_state_dict(model_weight)
    base_model.architecture = architecture
    return base_model
class CustomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        configs,
        mode="mia",
        batch_size: int = 16,
        num_workers: int = 16,
        image_size: int = -1,
        data_root: str = "../data",
        use_augmentation: bool = True,
    ):
        super().__init__()
        self.configs = configs
        self.dataset_name = configs["audit"]["dataset"]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.image_size = image_size
        self.data_root = data_root
        self.use_augmentation = use_augmentation

        self.dataset = get_dataset(self.configs["audit"]["dataset"], self.configs["data"]["data_dir"])

        self.num_base_classes = configs["train"]["num_classes"]
        self.dataset_type = self.dataset.dataset_type
        if self.dataset_type == "text":
            if self.configs["audit"]["reference_model"] == "flan-t5":
                self.batch_size = int(self.configs["train"]["per_device_train_batch_size"] / 2)
            else:
                self.batch_size = self.configs["train"]["per_device_train_batch_size"]
        #self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        configs = self.configs
        dataset = self.dataset
        base_architecture, architecture = configs["audit"]["target_model"], configs["audit"]["reference_model"]


        if dataset.dataset_type == 'image' and configs.get("architecture",1) == "convnext-tiny-224":
            import torchvision.transforms as transforms
            transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            dataset.transform = transform

        with open(configs["audit"]["target_model_split_info_dir"], "rb") as f:
            split_info = pickle.load(f)

        num_population = len(split_info['audit_split'])
        mia_val_index = np.random.choice(num_population, int(num_population * configs['audit']['val_ratio']), replace=False)
        mia_train_index = np.setdiff1d(np.arange(num_population), mia_val_index)

        from dataset import text_dataset_preprocess
        if dataset.dataset_type == "text":
            pro_dataset = text_dataset_preprocess(model_name=architecture, all_data=dataset, padding=True)
            pro_base_dataset = text_dataset_preprocess(model_name=base_architecture, all_data=dataset, padding=True)

            pro_base_dataset = pro_base_dataset.rename_column("labels", "base_labels")
            pro_base_dataset = pro_base_dataset.rename_column("input_ids", "base_input_ids")
            pro_base_dataset = pro_base_dataset.rename_column("attention_mask", "base_attention_mask")
            

            dataset = copy.deepcopy(pro_dataset)
            del pro_dataset
            dataset = dataset.add_column("base_labels", pro_base_dataset["base_labels"].tolist())
            dataset = dataset.add_column("base_input_ids", pro_base_dataset["base_input_ids"].tolist())
            dataset = dataset.add_column("base_attention_mask", pro_base_dataset["base_attention_mask"].tolist())



            if self.mode == "mia":
                self.train_dataset = dataset.select(indices=split_info['audit_split'][mia_train_index])  # large dataset of samples that were not used to train the original network
                self.test_dataset = dataset.select(indices=split_info['train_split'])  # This was used to train the nw
                self.val_dataset = dataset.select(indices=split_info['audit_split'][mia_val_index])  # test samples to test generalization of score model
            elif self.mode == 'eval':
                self.train_dataset = dataset.select(indices=split_info['audit_split'])  # large dataset of samples that were not used to train the original network
                self.test_dataset = dataset.select(indices=split_info['train_split'])  # This was used to train the nw
                self.val_dataset = dataset.select(indices=split_info['test_split'])  # test samples to test generalization of score model
            else:
                raise ValueError
            
        else:
            if self.mode == "mia":
                if dataset.dataset_type == "image":
                    from core import get_train_transform
                    augmentation = self.configs["train"]["augmentation"].lower()
                    assert augmentation in ["none", "crop", "flip", "both"], f"Augmentaion {augmentation} is not supported!"
                    image_size =  dataset[0][0].size()[-1]
                    train_transform = get_train_transform(dataset, self.dataset_name, augmentation, image_size)
                    dataset_copy = copy.deepcopy(dataset)
                    dataset_copy.transform = train_transform
                    self.train_dataset = torch.utils.data.Subset(dataset_copy, split_info['audit_split'][mia_train_index])  
                    self.test_dataset = torch.utils.data.Subset(dataset, split_info['train_split']) 
                    self.val_dataset = torch.utils.data.Subset(dataset, split_info['audit_split'][mia_val_index])  
                else:
                    self.train_dataset = torch.utils.data.Subset(dataset, split_info['audit_split'][mia_train_index])  # large dataset of samples that were not used to train the original network
                    self.test_dataset = torch.utils.data.Subset(dataset, split_info['train_split'])  # This was used to train the nw
                    self.val_dataset = torch.utils.data.Subset(dataset, split_info['audit_split'][mia_val_index])  # test samples to test generalization of score model
                
            
            elif self.mode == 'eval':
                self.train_dataset = torch.utils.data.Subset(dataset, split_info['audit_split'])  # large dataset of samples that were not used to train the original network
                self.test_dataset = torch.utils.data.Subset(dataset, split_info['train_split'])  # This was used to train the nw
                self.val_dataset = torch.utils.data.Subset(dataset, split_info['test_split'])  # test samples to test generalization of score model
                print(self.test_dataset[0][0].shape)
            else:
                raise ValueError



    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )
            
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return [
            DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers,
            ),
            DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers,
            ),
            DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers,
            ),
        ]

class HugginFaceTupleWrapper(torch.nn.Module):
    def __init__(self, model_base, hidden_dims=[], extra_inputs=None):
        super().__init__()
        self.model_base = model_base

        # Replaces the linear layer of the default classifier with an MLP
        if isinstance(self.model_base.classifier, torch.nn.Sequential):
            self.classifier = self.model_base.classifier
            self.model_base.classifier = torch.nn.Identity()
        else:
            prev_size = self.model_base.classifier.in_features
            if extra_inputs is not None:
                prev_size += extra_inputs
            num_classes = self.model_base.classifier.out_features
            mlp_list = []
            for hd in hidden_dims:
                mlp_list.append(torch.nn.Linear(prev_size, hd))
                mlp_list.append(torch.nn.LeakyReLU())  # TODO!
                prev_size = hd
            mlp_list.append(torch.nn.Linear(prev_size, num_classes))
            self.classifier = torch.nn.Sequential(*mlp_list)
            self.model_base.classifier = torch.nn.Identity()

        # self.linear =torch.nn.Linear(embedding_size, num_classes)
        super(HugginFaceTupleWrapper, self).add_module("model_base", self.model_base)
        super(HugginFaceTupleWrapper, self).add_module("classifier", self.classifier)

    def forward(self, input, extra_inputs=None):
        embedding = self.model_base(input).logits
        if extra_inputs is not None:
            assert (
                extra_inputs.shape[0] == embedding.shape[0]
                and extra_inputs.ndim == embedding.ndim
            ), "extra inputs and embedding need to have the same batch dimension"
            embedding = torch.concatenate([embedding, extra_inputs], dim=1)
            # print(embedding.shape)
        logits = self.classifier(embedding)
        return logits

    def freeze_base_model(self):
        for p in self.model_base.parameters():
            p.requires_grad = False
        for p in self.model_base.classifier.parameters():
            p.requires_grad = True

    def unfreeze_base_model(self):
        for p in self.model_base.parameters():
            p.requires_grad = True


