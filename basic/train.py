"""This file contains functions for training and testing the model."""
import time
import copy
import shutil
from ast import Tuple

import numpy as np
import torch
from torch import nn
from util import get_optimizer, get_text_model_path

import optuna
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score

import evaluate
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, Seq2SeqTrainingArguments, Seq2SeqTrainer
from models import INPUT_OUTPUT_SHAPE

def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    configs: dict,
    val_loader: torch.utils.data.DataLoader = None,
):
    """Train the model based on on the train loader
    Args:
        model(nn.Module): Model for evaluation.
        train_loader(torch.utils.data.DataLoader): Data loader for training.
        configs (dict): Configurations for training.
    Return:
        nn.Module: Trained model.
    """
    # Get the device for training
    device = configs.get("device", "cpu")
    patience = configs.get("early_stop_patience", 30)
    early_stop = True if patience > 0 else False

    # Set the model to the device
    model.to(device)
    model.train()
    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, configs)
    # Get the number of epochs for training
    epochs = configs.get("epochs", 1)
    scheduler_mode = configs["scheduler"] #configs.get("scheduler", "none")
    assert scheduler_mode in ["CosineAnnealingLR", "None"]
    if scheduler_mode == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_acc, early_stop_track = 0, 0

    # Loop over each epoch
    for epoch_idx in range(epochs):
        start_time = time.time()
        train_loss, train_acc = 0, 0
        # Loop over the training set
        model.train()
        for data, target in train_loader:
            # Move data to the device
            data, target = data.to(device, non_blocking=True), target.to(
                device, non_blocking=True
            )
            # Cast target to long tensor
            target = target.long()

            # Set the gradients to zero
            optimizer.zero_grad(set_to_none=True)

            # Get the model output
            output = model(data)
            
            # Calculate the loss
            loss = criterion(output, target)
            pred = output.data.max(1, keepdim=True)[1]
            train_acc += pred.eq(target.data.view_as(pred)).sum()
            # Perform the backward pass
            loss.backward()
            # Take a step using optimizer
            optimizer.step()
            # Add the loss to the total loss
            train_loss += loss.item()

        print(f"Epoch: {epoch_idx+1}/{epochs} |", end=" ")
        print(f"Train Loss: {train_loss/len(train_loader):.8f} ", end=" ")
        print(f"Train Acc: {float(train_acc)/len(train_loader.dataset):.8f} ", end=" ")

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss, val_acc = 0, 0
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    # Cast target to long tensor
                    target = target.long()
                    # Computing output and loss
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    # Computing accuracy
                    pred = output.data.max(1, keepdim=True)[1]
                    val_acc += pred.eq(target.data.view_as(pred)).sum()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    early_stop_track = 0
                    best_parameters = copy.deepcopy(model.state_dict())
                else:
                    early_stop_track += 1
                    
            print(f"Val Loss: {val_loss/len(val_loader):.8f} ", end=" ")
            print(f"Val Acc: {float(val_acc)/len(val_loader.dataset):.8f} ", end=" ")

            if (early_stop_track > patience) and early_stop:
                print("Early stopped. ACC didn't improve for {} epochs.".format(patience))
                break
        print(f"One step uses {time.time() - start_time:.2f} seconds")

        if scheduler_mode == "CosineAnnealingLR":
            scheduler.step()

    if val_loader is not None and early_stop: 
        print("Load best model parameters based on the validation set.")
        model.load_state_dict(best_parameters)

    # Move the model back to the CPU
    model.to("cpu")

    # Return the model
    return model


# Test Function
def inference(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str
) -> Tuple(float, float):
    """Evaluate the model performance on the test loader
    Args:
        model (torch.nn.Module): Model for evaluation
        loader (torch.utils.data.DataLoader): Data Loader for testing
        device (str): GPU or CPU
    Return:
        loss (float): Loss for the given model on the test dataset.
        acc (float): Accuracy for the given model on the test dataset.
    """

    # Setting model to eval mode and moving to specified device
    model.eval()
    model.to(device)

    # Assigning variables for computing loss and accuracy
    loss, acc, criterion = 0, 0, nn.CrossEntropyLoss()

    # Disable gradient calculation to save memory
    with torch.no_grad():
        for data, target in loader:
            # Moving data and target to the device
            data, target = data.to(device), target.to(device)
            # Cast target to long tensor
            target = target.long()

            # Computing output and loss
            output = model(data)
            loss += criterion(output, target).item()

            # Computing accuracy
            pred = output.data.max(1, keepdim=True)[1]
            acc += pred.eq(target.data.view_as(pred)).sum()

        # Averaging the losses
        loss /= len(loader)

        # Calculating accuracy
        acc = float(acc) / len(loader.dataset)

        # Move model back to CPU
        model.to("cpu")

        # Return loss and accuracy
        return loss, acc

def catboost_train(_X_train, _y_train, _X_valid, _y_valid, configs, dataset_name):
    seed = configs["random_seed"]
    #X_test = np.concatenate((_X_train, _X_valid), axis=0)
    device, model_name = configs["device"], configs["model_name"]
    patience = configs.get("early_stop_patience", 30)
    early_stop = True if patience > 0 else False

    if model_name == 'catboost':
        model = CatBoostClassifier(learning_rate=0.05, iterations=10000, loss_function="MultiClass", random_seed=seed, task_type="GPU", devices=device)
        if early_stop:
            model.fit(_X_train, _y_train, eval_set=[(_X_valid, _y_valid)], \
                early_stopping_rounds=patience, verbose=True)
        else:
            model.fit(_X_train, _y_train, verbose=True)

    elif model_name == "svm":
        from sklearn import svm
        model = svm.SVC(verbose=True, max_iter=100)
        model.fit(_X_train, _y_train)

    elif model_name == "lr":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=0, n_jobs=2, verbose=1).fit(_X_train, _y_train)

    elif model_name == 'lightgbm':
        num_class = INPUT_OUTPUT_SHAPE[dataset_name][1] if INPUT_OUTPUT_SHAPE[dataset_name][1] > 2 else 1
        obj = 'multiclass' if INPUT_OUTPUT_SHAPE[dataset_name][1] > 2 else 'binary'
        metric = 'multi_error' if INPUT_OUTPUT_SHAPE[dataset_name][1] > 2 else 'binary_error'
        patience = 100

        model = lgb.LGBMClassifier(
            num_class=num_class,
            objective=obj,
            verbosity=-1,
            random_state=seed,
            max_depth=3,
            n_estimators=5000,
            learning_rate=0.1,
        )
        if early_stop:
            callbacks = [lgb.log_evaluation(period=1), lgb.early_stopping(stopping_rounds=patience)]
            model.fit(_X_train, _y_train, eval_metric=metric, eval_set=[(_X_valid, _y_valid)],
                callbacks=callbacks)
        else:
            model.fit(_X_train, _y_train, eval_metric=metric, eval_set=[(_X_valid, _y_valid)])

    else:
        raise ValueError

    return model, model.get_params()

def catboost_inference(model, X, Y, model_name):
    y_pred = model.predict(X)
    score = accuracy_score(Y, y_pred)
    return score


def text_model_train(log_dir, dataset, train_split, eval_split, configs, train_trainer=True):
    model_name, metric_name = configs["model_name"], configs["metric_name"]
    model_path = get_text_model_path(model_name)
    patience = configs.get("early_stop_patience", 30)
    early_stop = True if patience > 0 else False
    callback = EarlyStoppingCallback(early_stopping_patience=patience) if early_stop else None
    
    output_dir  = f"{log_dir}/check_points"
    batch_size, lr, weight_decay = configs["batch_size"], configs["learning_rate"], configs["weight_decay"]
    per_device_train_batch_size, per_device_eval_batch_size = configs["per_device_train_batch_size"], configs["per_device_test_batch_size"]
    num_labels = configs["num_classes"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    accuracy = evaluate.load(f"../huggingface/metrics/{metric_name}")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    if model_name != "flan-t5":
        train_arg = "TrainingArguments"
        trainer = "Trainer"
    else:
        train_arg = "Seq2SeqTrainingArguments"
        trainer = "Seq2SeqTrainer"

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if type(predictions) == tuple:
            predictions = predictions[0]
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    train_dataset = dataset.select(indices=train_split)
    eval_dataset  = dataset.select(indices=eval_split)

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    training_args = eval(train_arg)(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=int(batch_size/per_device_train_batch_size),
        num_train_epochs=configs["epochs"],
        weight_decay=weight_decay,
        # eval_strategy="steps",
        # save_strategy="steps",
        eval_strategy="epoch",
        save_strategy="epoch",
        # eval_steps=20,
        # save_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_"+metric_name,
        report_to="none",
        logging_steps=1,
        fp16=True,
        optim=configs["optimizer"],
    )

    trainer = eval(trainer)(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[callback],
    )

    if train_trainer:
        trainer.train()
    trainer.model.to("cpu")

    shutil.rmtree(output_dir)

    return trainer

def text_model_inference(trainer, evaluate_dataset):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    trainer.model.to(device)
    eval_results = trainer.evaluate(evaluate_dataset)
    return round(eval_results["eval_loss"], 8), round(eval_results["eval_accuracy"], 8)
    