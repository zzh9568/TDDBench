import copy
import torch
import numpy as np
from utils import _m_entr_comp
from scipy.special import softmax
from scipy.stats import entropy
from torch import nn
from privacy_meter.model import PytorchModelTensor
from dataset import get_batch
from tqdm import tqdm

# Signal model class for GBDT models
class GBDTNumpy():
    def __init__(self, model_obj, loss_fn, model_name):
        self.model_obj  = model_obj
        self.loss_fn    = loss_fn
        self.loss_fn.reduction = "none"
        self.model_name = model_name
        assert self.model_name in ['catboost', 'lightgbm', 'lr'], 'Other models are not yet available'
    
    def get_logits(self, X):
        if self.model_name == 'catboost':
            y_pred = self.model_obj.predict(X.numpy(), prediction_type="LogProbability") #Probability
        elif self.model_name in ['lightgbm', 'lr']:
            y_pred = self.model_obj.predict_log_proba(X.numpy()) #predict_proba
        else:
            raise ValueError
        return y_pred
    
    def get_loss(self, X, Y):
        y_pred = self.get_logits(X)
        loss = self.loss_fn(torch.Tensor(y_pred), Y).numpy()
        return loss

    def get_rescaled_logits(self, X, Y):
        COUNT = len(X)
        y_pred = self.get_logits(X)
        confi = softmax(y_pred, axis=1)
        confi_corret = confi[np.arange(COUNT), Y]
        confi[np.arange(COUNT), Y] = 0
        confi_wrong = np.sum(confi, axis=1)
        logit = np.log(confi_corret + 1e-45) - np.log(confi_wrong + 1e-45)
        return logit

# Signal model class for text classification models
class TextModelTensor():
    def __init__(self, model_obj, loss_fn, device="cpu", batch_size=25):
        self.model_obj  = model_obj
        self.loss_fn    = loss_fn
        self.loss_fn.reduction = "none"
        self.device = device
        self.batch_size = batch_size

    def get_logits(self, batch):
        self.model_obj.to(self.device)
        self.model_obj.eval()

        batch_idxs = torch.LongTensor(np.arange(len(batch["input_ids"])))
        split_idxs = torch.split(batch_idxs, self.batch_size)
        with torch.no_grad():
            logits_list = []
            for split_idx in tqdm(split_idxs):
                sub_batch = {k: v[split_idx].to(self.device) for k, v in batch.items()}
                out = self.model_obj(**sub_batch)
                logits_list.append(out.logits.detach())  # to avoid the OOM
            all_logits = torch.cat(logits_list).detach().cpu().numpy()
        self.model_obj.to("cpu")
        return all_logits
    
    def get_loss(self, batch):
        y_pred = self.get_logits(batch)
        loss = self.loss_fn(torch.Tensor(y_pred), batch["labels"]).numpy()
        return loss

    def get_rescaled_logits(self, batch):
        Y = batch["labels"]
        COUNT = len(Y)
        y_pred = self.get_logits(batch)
        confi = softmax(y_pred, axis=1)
        confi_corret = confi[np.arange(COUNT), Y]
        confi[np.arange(COUNT), Y] = 0
        confi_wrong = np.sum(confi, axis=1)
        logit = np.log(confi_corret + 1e-45) - np.log(confi_wrong + 1e-45)
        return logit

# get signal name from input algorithm names
def get_signal_name(algs):
    assert isinstance(algs, list) or isinstance(algs, str)
    signal_names = []
    support_signal_names = ['loss', 'confidence', 'correctness', 'entropy', 'mentr',\
                            'rescaled_logits', 'unnormalized_logit_difference']
    if isinstance(algs, list):
        for alg in algs:
            if alg.split('+')[1] in support_signal_names:
                signal_names.append(alg.split('+')[1])
    else:
        if algs.split('+')[1] in support_signal_names:
            signal_names.append(algs.split('+')[1])
    return list(set(signal_names))

# get signal model name from input model name
from info import DOMAIN_SUPPORT_MODEL
def get_signal_model_name(model_name):
    gbdt_model = ["catboost", "lightgbm", "lr"]
    tabular_torch_model = set(DOMAIN_SUPPORT_MODEL["tabular"]) - set(gbdt_model) 
    SIGNAL_MODEL_MAP = {
        "PytorchModel": DOMAIN_SUPPORT_MODEL["image"] + list(tabular_torch_model),
        "GBDT": gbdt_model,
        "TextModel": DOMAIN_SUPPORT_MODEL["text"],
    }
    for key, value in SIGNAL_MODEL_MAP.items():
        if model_name in value:
            return key
        
# Get data's membership signal from signal model
# Notice: make sure that members' mean signal < non-members' mean signal for every signal name (except confidence vector)!
# Confidence means the prediction score (SoftMax) of output of the model for true label (target)
def _get_signals(signal_model, datas, targets, signal_name):
    if signal_name == 'loss':
        return signal_model.get_loss(datas, targets)
    elif signal_name == 'rescaled_logits':
        return -signal_model.get_rescaled_logits(datas, targets)
    elif signal_name == 'confidence':
        logits = signal_model.get_logits(datas)
        confidence = softmax(logits, axis=1)
        return -confidence[np.arange(len(confidence)),targets.numpy()]
    elif signal_name == 'confidence_vector':
        logits = signal_model.get_logits(datas)
        confidence_vector = softmax(logits, axis=1)
        return confidence_vector
    elif signal_name == 'correctness':
        logits = signal_model.get_logits(datas)
        logits_predict = np.argmax(logits,1)
        correctness = logits_predict == targets.numpy()
        return -correctness.astype(logits.dtype)
    elif signal_name == 'entropy':
        logits = signal_model.get_logits(datas)
        confidence = softmax(logits, axis=1)
        return entropy(confidence.T, base=2)
    elif signal_name == 'mentr':
        logits = signal_model.get_logits(datas)
        confidence = softmax(logits, axis=1)
        return _m_entr_comp(confidence, targets.numpy())
    elif signal_name == 'unnormalized_logit_difference':
        targets = targets.numpy()
        logits = signal_model.get_logits(datas)
        logits_target = copy.deepcopy(logits[np.arange(len(logits)),targets])
        logits[np.arange(len(logits)),targets] = -1
        logits_second_target = np.max(logits,1)
        return logits_second_target - logits_target
    else:
        raise ValueError(
            f"The {signal_name} is not supported."
        )
    
# Get text data's membership signal from text signal model
# Notice: make sure that members' mean signal < non-members' mean signal for every signal name (except confidence vector)!
# Confidence means the prediction score (SoftMax) of output of the model for true label (target)
def _text_get_signals(signal_model, batch, signal_name):
    targets = batch["labels"]
    if signal_name == 'loss':
        return signal_model.get_loss(batch)
    elif signal_name == 'rescaled_logits':
        return -signal_model.get_rescaled_logits(batch)
    elif signal_name == 'confidence':
        logits = signal_model.get_logits(batch)
        confidence = softmax(logits, axis=1)
        return -confidence[np.arange(len(confidence)),targets.numpy()]
    elif signal_name == 'confidence_vector':
        logits = signal_model.get_logits(batch)
        confidence_vector = softmax(logits, axis=1)
        return confidence_vector
    elif signal_name == 'correctness':
        logits = signal_model.get_logits(batch)
        logits_predict = np.argmax(logits,1)
        correctness = logits_predict == targets.numpy()
        return -correctness.astype(logits.dtype)
    elif signal_name == 'entropy':
        logits = signal_model.get_logits(batch)
        confidence = softmax(logits, axis=1)
        return entropy(confidence.T, base=2)
    elif signal_name == 'mentr':
        logits = signal_model.get_logits(batch)
        confidence = softmax(logits, axis=1)
        return _m_entr_comp(confidence, targets.numpy())
    elif signal_name == 'unnormalized_logit_difference':
        targets = targets.numpy()
        logits = signal_model.get_logits(batch)
        logits_target = copy.deepcopy(logits[np.arange(len(logits)),targets])
        logits[np.arange(len(logits)),targets] = -1
        logits_second_target = np.max(logits,1)
        return logits_second_target - logits_target
    else:
        raise ValueError(
            f"The {signal_name} is not supported for text data."
        )

def get_signals(model_list, model_name, signal_name, device, batch_size, target_dataset, population_dataset, get_population_signal=False):
    model_names = [model.__class__.__name__ for model in model_list]
    assert len(np.unique(np.array(model_names))) == 1, "The categories of the models in the model list should be consistent!"

    signals = []
    population_signals = []
    signal_model_name = get_signal_model_name(model_name)

    target_batch = get_batch(target_dataset, model_name)
    population_batch = get_batch(population_dataset, model_name)

    print("Start to get signals.")
    for model in model_list:
        if model is None:
            pass
        elif signal_model_name == "TextModel":
            signal_model = TextModelTensor(
                model_obj=model,
                loss_fn=nn.CrossEntropyLoss(),
                device=device,
                batch_size=batch_size,
            )
            signals.append(_text_get_signals(signal_model, target_batch, signal_name))
            if get_population_signal:
                population_signals.append(_text_get_signals(signal_model, population_batch, signal_name))
        elif signal_model_name == "PytorchModel":
            data, targets = target_batch
            signal_model = PytorchModelTensor(
                model_obj=model,
                loss_fn=nn.CrossEntropyLoss(),
                device=device,
                batch_size=batch_size,
            )
            signals.append(_get_signals(signal_model, data, targets, signal_name))
            if get_population_signal:
                p_data, p_targets = population_batch
                population_signals.append(_get_signals(signal_model, p_data, p_targets, signal_name))
        elif signal_model_name == "GBDT":
            data, targets = target_batch
            signal_model = GBDTNumpy(          
                model_obj=model,
                loss_fn=torch.nn.CrossEntropyLoss(),
                model_name=model_name)
            signals.append(_get_signals(signal_model, data, targets, signal_name))
            if get_population_signal:
                p_data, p_targets = population_batch
                population_signals.append(_get_signals(signal_model, p_data, p_targets, signal_name))
        else:
            raise ValueError(
                f"The {signal_model_name} is not supported."
            )
    population_signals = np.array(population_signals) if get_population_signal else None
    return np.array(signals), population_signals



