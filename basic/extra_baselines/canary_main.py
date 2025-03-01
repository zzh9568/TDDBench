import torch
import torch.nn as nn
import numpy as np

import torchvision.transforms as transforms

import os
import argparse
import copy
import wandb
import random
from pynvml import *
import pickle
from pathlib import Path
import sys
sys.path.insert(0, "../basic/")

from canary_mia.utils import (generate_aug_imgs, get_attack_loss, split_shadow_models, get_curr_shadow_models, 
    calculate_loss, get_logits, progress_bar, calibrate_logits, cal_results)
from core import get_model
from dataset import get_dataset as benchmark_get_dataset

def load_model_meta(dir):
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

class InferenceModel(nn.Module):
    def __init__(self, shadow_id, args, model_metadata_dict, model_type="target"):
        assert shadow_id >= 0, 'shadow_id error'
        assert model_type in ["target", "shadow"], "model type error"
        super().__init__()
        self.shadow_id = shadow_id
        self.args = args

        if model_type == "target":
            self.model = get_model(args.target_net, args.configs["audit"]["dataset"])
        else:
            self.model = get_model(args.shadow_net, args.configs["audit"]["dataset"])

        resume_checkpoint = model_metadata_dict["model_metadata"][shadow_id]["model_path"]
        assert os.path.isfile(resume_checkpoint), 'Error: no checkpoint found!'
        with open(resume_checkpoint, "rb") as file:
            model_weight = pickle.load(file)
        self.model.load_state_dict(model_weight)

        print(f"load the saved checkpoint from paths: {resume_checkpoint}")
        train_split = model_metadata_dict["model_metadata"][shadow_id]["train_split"]
        self.in_data = [args.train_test_split.index(i) for i in train_split if i in args.train_test_split]
        self.keep_bool = None
        
        # no grad by default
        self.deactivate_grad()
        self.model.eval()
        self.is_in_model = False # False for out_model

    def forward(self, x):
        return self.model(x)

    def deactivate_grad(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def activate_grad(self):
        for param in self.model.parameters():
            param.requires_grad = True
    
    def load_state_dict(self, checkpoint):
        self.model.load_state_dict(checkpoint)

def generate_class_dict(args):
    dataset_class_dict = [[] for _ in range(args.num_classes)]
    for i in range(len(args.aug_trainset)):
        _, tmp_class = args.aug_trainset[i]
        dataset_class_dict[tmp_class].append(i)

    return dataset_class_dict

def generate_close_imgs(args):
    canaries = []
    target_class_list = args.dataset_class_dict[args.target_img_class]

    if args.aug_strategy and 'same_class_imgs' in args.aug_strategy:
        # assume always use the target img
        canaries = [args.aug_trainset[args.target_img_id][0].unsqueeze(0)]
        for i in range(args.num_gen - 1):
            img_id = random.sample(target_class_list, 1)[0]
            x = args.aug_trainset[img_id][0]
            x = x.unsqueeze(0)

            canaries.append(x)
    elif args.aug_strategy and 'nearest_imgs' in args.aug_strategy:
        similarities = []
        target_img = args.aug_trainset[args.target_img_id][0]
        canaries = []
        for i in target_class_list:
            similarities.append(torch.abs(target_img - args.aug_trainset[i][0]).sum())
        
        top_k_indx = np.argsort(similarities)[:(args.num_gen)]
        target_class_list = np.array(target_class_list)
        final_list = target_class_list[top_k_indx]

        for i in final_list:
            canaries.append(args.aug_trainset[i][0].unsqueeze(0))
    return canaries

def initialize_poison(args):
    """Initialize according to args.init.
    Propagate initialization in distributed settings.
    """
    if args.aug_strategy and ('same_class_imgs' in args.aug_strategy or 'nearest_imgs' in args.aug_strategy):
        if 'dataset_class_dict' not in args:
            args.dataset_class_dict = generate_class_dict(args)

        fixed_target_img = generate_close_imgs(args)
        args.fixed_target_img = torch.cat(fixed_target_img, dim=0).to(args.device)
    else:
        fixed_target_img = generate_aug_imgs(args)
        args.fixed_target_img = torch.cat(fixed_target_img, dim=0).to(args.device)

    # ds has to be placed on the default (cpu) device, not like self.ds
    dm = torch.tensor(args.data_mean)[None, :, None, None]
    ds = torch.tensor(args.data_std)[None, :, None, None]
    if args.init == 'zero':
        init = torch.zeros(args.num_gen, *args.canary_shape)
    elif args.init == 'rand':
        init = (torch.rand(args.num_gen, *args.canary_shape) - 0.5) * 2
        init *= 1 / ds
    elif args.init == 'randn':
        init = torch.randn(args.num_gen, *args.canary_shape)
        init *= 1 / ds
    elif args.init == 'normal':
        init = torch.randn(args.num_gen, *args.canary_shape)
    elif args.init == 'target_img':
        init = copy.deepcopy(args.fixed_target_img)
        init.requires_grad = True
        return init
    else:
        raise NotImplementedError()

    init = init.to(args.device)
    dm = dm.to(args.device)
    ds = ds.to(args.device)

    if args.epsilon:
        x_diff = init.data - args.fixed_target_img.data
        x_diff.data = torch.max(torch.min(x_diff, args.epsilon /
                                                ds / 255), -args.epsilon / ds / 255)
        x_diff.data = torch.max(torch.min(x_diff, (1 - dm) / ds -
                                                args.fixed_target_img), -dm / ds - args.fixed_target_img)
        init.data = args.fixed_target_img.data + x_diff.data
    else:
        init = torch.max(torch.min(init, (1 - dm) / ds), -dm / ds)

    init.requires_grad = True
    return init

def generate_canary_one_shot(shadow_models, args, return_loss=False):
    target_img_class = args.target_img_class

    # get loss functions
    args.in_criterion = get_attack_loss(args.in_model_loss)
    args.out_criterion = get_attack_loss(args.out_model_loss)

    # initialize patch
    x = initialize_poison(args)
    y = torch.tensor([target_img_class] * args.num_gen).to(args.device)

    dm = torch.tensor(args.data_mean)[None, :, None, None].to(args.device)
    ds = torch.tensor(args.data_std)[None, :, None, None].to(args.device)
    
    # initialize optimizer
    if args.opt.lower() in ['adam', 'signadam']:
        optimizer = torch.optim.Adam([x], lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt.lower() in ['sgd', 'signsgd']:
        optimizer = torch.optim.SGD([x], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.opt.lower() in ['adamw']:
        optimizer = torch.optim.AdamW([x], lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduling:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.iter // 2.667, args.iter // 1.6,
                                                                                    args.iter // 1.142], gamma=0.1)
    else:
        scheduler = None
        
    # generate canary
    last_loss = 100000000000
    trigger_times = 0
    loss = torch.tensor([0.0], requires_grad=True)

    for step in range(args.iter):
        # choose shadow models
        curr_shadow_models = get_curr_shadow_models(shadow_models, x, args)

        for _ in range(args.inner_iter):
            loss, in_loss, out_loss, reg_norm = calculate_loss(x, y, curr_shadow_models, args)
            optimizer.zero_grad()
            # loss.backward()
            if loss != 0:
                x.grad,  = torch.autograd.grad(loss, [x])
            if args.opt.lower() in ['signsgd', 'signadam'] and x.grad is not None:
                x.grad.sign_()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()
            
            # projection
            with torch.no_grad():
                if args.epsilon:
                    x_diff = x.data - args.fixed_target_img.data
                    x_diff.data = torch.max(torch.min(x_diff, args.epsilon /
                                                            ds / 255), -args.epsilon / ds / 255)
                    x_diff.data = torch.max(torch.min(x_diff, (1 - dm) / ds -
                                                            args.fixed_target_img), -dm / ds - args.fixed_target_img)
                    x.data = args.fixed_target_img.data + x_diff.data
                else:
                    x.data = torch.max(torch.min(x, (1 - dm) / ds), -dm / ds)
            
            if args.print_step:
                print(f'step: {step}, ' + 'loss: %.3f, in_loss: %.3f, out_loss: %.3f, reg_loss: %.3f' % (loss, in_loss, out_loss, reg_norm))

        if args.stop_loss is not None and loss <= args.stop_loss:
            break

        if args.early_stop and loss > last_loss:
            trigger_times += 1

            if trigger_times >= args.patience:
                break
        else:
            trigger_times = 0

        last_loss = loss.item()

    if return_loss:
        return x.detach(), loss.item()
    else:
        return x.detach()


def generate_canary(shadow_models, args):
    canaries = []

    if args.aug_strategy is not None:
        rand_start = random.randrange(args.num_classes)

        for out_target_class in range(1000): # need to be simplified later
            if args.canary_aug:
                args.target_img, args.target_img_class = args.aug_trainset[args.target_img_id]
                args.target_img = args.target_img.unsqueeze(0).to(args.device)

            if 'try_all_out_class' in args.aug_strategy:
                out_target_class = (rand_start + out_target_class) % args.num_classes
            elif 'try_random_out_class' in args.aug_strategy:
                out_target_class = random.randrange(args.num_classes)
            elif 'try_random_diff_class' in args.aug_strategy:
                pass
            else:
                raise NotImplementedError()

            if out_target_class != args.target_img_class:
                if args.print_step:
                    print(f'Try class: {out_target_class}')

                if 'try_random_diff_class' in args.aug_strategy:
                    out_target_class = []
                    for _ in range(args.num_gen):
                        a = random.randrange(args.num_classes)
                        while a == args.target_img_class:
                            a = random.randrange(args.num_classes)

                        out_target_class.append(a)
                    
                    args.out_target_class = torch.tensor(out_target_class).to(args.device)
                else:
                    args.out_target_class = out_target_class

                x, loss = generate_canary_one_shot(shadow_models, args, return_loss=True)

                canaries.append(x)
                args.canary_losses[-1].append(loss)

            if sum([len(canary) for canary in canaries]) >= args.num_aug:
                break
    else:
        x, loss = generate_canary_one_shot(shadow_models, args, return_loss=True)
        canaries.append(x)
        args.canary_losses[-1].append(loss)
        
    return canaries


def main(args, configs, target_model_idx):
    usewandb = not args.nowandb
    if usewandb:
        wandb.init(project='canary_generation',name=args.save_name)
        wandb.config.update(args)

    ###################################Modify#################################################
    args.configs = configs
    args.device = configs["audit"]["device"]

    # set random seed
    args.seed = configs["random_seed"]
    torch.manual_seed(configs["random_seed"])
    np.random.seed(configs["random_seed"])
    random.seed(configs["random_seed"])
    torch.cuda.manual_seed_all(configs["random_seed"])

    args.num_gen = configs['audit']['query_num']
    args.num_aug = args.num_gen


    tv_dataset = benchmark_get_dataset(configs["audit"]["dataset"], configs["data"]["data_dir"])

    if tv_dataset.dataset_type == 'image':
        args.data_mean = [0.5, 0.5, 0.5]
        args.data_std = [0.5, 0.5, 0.5]
        from models import INPUT_OUTPUT_SHAPE
        args.num_classes = INPUT_OUTPUT_SHAPE[configs["audit"]["dataset"]][1]

    meta_log_dir = f'''{configs["train"]["log_dir"]}/{configs["data"]["dataset_type"]}'''
    dataset_name, ref_data_dis = configs["audit"]["dataset"], configs["audit"]["ref_data_dis"]
    subtar_ratio = configs["audit"]["subtar_ratio"]
    target_model_name, reference_model_name = configs["audit"]["target_model"], configs["audit"]["reference_model"]
    
    if ref_data_dis == "shadow":
        args.offline = True

    if ref_data_dis == "subtar":
        ref_model_dir = f"{meta_log_dir}/{dataset_name}/{reference_model_name}/subtar({subtar_ratio})"
    elif ref_data_dis == "subtar_shadow":
        ref_model_dir = f"{meta_log_dir}/{dataset_name}/{reference_model_name}/subtar({subtar_ratio})_shadow"
    else:
        ref_model_dir = f"{meta_log_dir}/{dataset_name}/{reference_model_name}/{ref_data_dis}"

    target_model_metadata = load_model_meta(f"{meta_log_dir}/{dataset_name}/{target_model_name}/target")
    ref_model_metadata = load_model_meta(ref_model_dir)

    subtar_ratio = configs["audit"]["subtar_ratio"]
    data_split_info_dir = f'''{configs["train"]["log_dir"]}/{configs["data"]["dataset_type"]}/{configs["audit"]["dataset"]}/data_split_info'''

    
    if ref_data_dis in ["subtar", "subtar_shadow"]:
        train_test_split = np.load(f"{data_split_info_dir}/subtar({subtar_ratio})_data_index.npy")
    elif ref_data_dis in ["target_ref", "shadow"]:
        train_test_split = np.load(f"{data_split_info_dir}/target_data_index.npy")
    else:
        raise ValueError(f"We do not support reference data distribution: {ref_data_dis}")
    audit_split = np.setdiff1d(np.arange(len(tv_dataset)), train_test_split)
    np.random.shuffle(train_test_split)

    trainset = torch.utils.data.Subset(tv_dataset, train_test_split)
    args.train_test_split = list(train_test_split)
    args.canary_trainset = None

    args.img_shape = trainset[0][0].shape
    args.canary_shape = trainset[0][0].shape

    from core import get_train_transform
    augmentation = configs["train"]["augmentation"].lower()
    assert augmentation in ["none", "crop", "flip", "both"], f"Augmentaion {augmentation} is not supported!"
    image_size =  tv_dataset[0][0].size()[-1]
    train_transform = get_train_transform(tv_dataset, dataset_name, augmentation, image_size)
    aug_dataset = copy.deepcopy(tv_dataset)
    aug_dataset.transform = train_transform

    args.aug_trainset = torch.utils.data.Subset(aug_dataset, train_test_split)
    args.aug_testset = torch.utils.data.Subset(aug_dataset, audit_split)

    args.start, args.end = 0, min(len(trainset), configs["audit"]["boundary_eva_num"])

    args.num_shadow = configs["audit"]["number_of_reference_models"]

    args.target_net, args.shadow_net = configs["audit"]["target_model"], configs["audit"]["reference_model"]
    # load shadow and target models
    shadow_models = []
    shadow_ids = list(range(args.num_shadow))

    for i in shadow_ids:
        curr_model = InferenceModel(i, args, ref_model_metadata, model_type="shadow").to(args.device)
        shadow_models.append(curr_model)
    
    target_model = InferenceModel(target_model_idx, args, target_model_metadata, model_type="target").to(args.device)

    args.target_model = target_model
    args.shadow_models = shadow_models

    #######################################Modify#####################################################
    
    args.pred_logits = [] # N x (num of shadow + 1) x num_trials x num_class (target at -1)
    args.in_out_labels = [] # N x (num of shadow + 1)
    args.canary_losses = [] # N x num_trials
    args.class_labels = []  # N
    args.img_id = [] # N

    for i in range(args.start, args.end):
        args.target_img_id = i

        args.target_img, args.target_img_class = trainset[args.target_img_id]
        args.target_img = args.target_img.unsqueeze(0).to(args.device)

        args.in_out_labels.append([])
        args.canary_losses.append([])
        args.pred_logits.append([])

        if args.num_val:
            in_models, out_models = split_shadow_models(shadow_models, args.target_img_id)
            num_in = min(int(args.num_val / 2), len(in_models))
            num_out = args.num_val - num_in

            train_shadow_models = random.sample(in_models, num_in)
            train_shadow_models += random.sample(out_models, num_out)

            val_shadow_models = train_shadow_models
        else:
            train_shadow_models = shadow_models
            val_shadow_models = shadow_models

        if args.aug_strategy and 'baseline' in args.aug_strategy:
            curr_canaries = generate_aug_imgs(args)
        else:
            curr_canaries = generate_canary(train_shadow_models, args)

        # get logits
        curr_canaries = torch.cat(curr_canaries, dim=0).to(args.device)
        for curr_model in val_shadow_models:
            args.pred_logits[-1].append(get_logits(curr_canaries, curr_model))
            args.in_out_labels[-1].append(int(args.target_img_id in curr_model.in_data))

        args.pred_logits[-1].append(get_logits(curr_canaries, target_model))
        args.in_out_labels[-1].append(int(args.target_img_id in target_model.in_data))

        args.img_id.append(args.target_img_id)
        args.class_labels.append(args.target_img_class)

        progress_bar(i, args.end - args.start)


    # accumulate results
    pred_logits = np.array(args.pred_logits)
    in_out_labels = np.array(args.in_out_labels)
    class_labels = np.array(args.class_labels)

    in_out_labels = np.swapaxes(in_out_labels, 0, 1).astype(bool)
    pred_logits = np.swapaxes(pred_logits, 0, 1)

    scores = calibrate_logits(pred_logits, class_labels, args.logits_strategy)

    shadow_scores = scores[:-1]
    target_scores = scores[-1:]
    shadow_in_out_labels = in_out_labels[:-1]
    target_in_out_labels = in_out_labels[-1:]

    some_stats = cal_results(shadow_scores, shadow_in_out_labels, target_scores, target_in_out_labels, logits_mul=args.logits_mul, only_off=args.offline)
    if usewandb:
        wandb.log(some_stats)

    report_dir = f'''{configs["audit"]["report_dir"]}/target_model_idx_{target_model_idx}'''

    from info import METHODS_ALIASES
    if args.offline:
        canary_method_name = METHODS_ALIASES["query-adv+out_pdf+fix_var+rescaled_logits"]
        print(f"AUROC:{some_stats['fix_var_offline_auc']}")
        Path(f"{report_dir}/{canary_method_name}").mkdir(parents=True, exist_ok=True)
        np.save(f"{report_dir}/{canary_method_name}/fpr", some_stats['fix_var_offline_FPR'])
        np.save(f"{report_dir}/{canary_method_name}/tpr", some_stats['fix_var_offline_TPR'])
        np.save(f"{report_dir}/{canary_method_name}/positive", some_stats['fix_var_offline_pos'])
        np.save(f"{report_dir}/{canary_method_name}/negative", some_stats['fix_var_offline_neg'])

        canary_method_name = "query-adv+out_pdf+rescaled_logits"
        print(f"AUROC:{some_stats['offline_auc']}")
        Path(f"{report_dir}/{canary_method_name}").mkdir(parents=True, exist_ok=True)
        np.save(f"{report_dir}/{canary_method_name}/fpr", some_stats['offline_FPR'])
        np.save(f"{report_dir}/{canary_method_name}/tpr", some_stats['offline_TPR'])
        np.save(f"{report_dir}/{canary_method_name}/positive", some_stats['offline_pos'])
        np.save(f"{report_dir}/{canary_method_name}/negative", some_stats['offline_neg'])

    else:
        canary_method_name = METHODS_ALIASES["query-adv+in_out_pdf+fix_var+rescaled_logits"]
        print(f"AUROC:{some_stats['fix_var_auc']}")
        Path(f"{report_dir}/{canary_method_name}").mkdir(parents=True, exist_ok=True)
        np.save(f"{report_dir}/{canary_method_name}/fpr", some_stats['fix_var_FPR'])
        np.save(f"{report_dir}/{canary_method_name}/tpr", some_stats['fix_var_TPR'])
        np.save(f"{report_dir}/{canary_method_name}/positive", some_stats['fix_var_pos'])
        np.save(f"{report_dir}/{canary_method_name}/negative", some_stats['fix_var_neg'])

        canary_method_name = "query-adv+in_out_pdf+rescaled_logits"
        print(f"AUROC:{some_stats['auc']}")
        Path(f"{report_dir}/{canary_method_name}").mkdir(parents=True, exist_ok=True)
        np.save(f"{report_dir}/{canary_method_name}/fpr", some_stats['FPR'])
        np.save(f"{report_dir}/{canary_method_name}/tpr", some_stats['TPR'])
        np.save(f"{report_dir}/{canary_method_name}/positive", some_stats['pos'])
        np.save(f"{report_dir}/{canary_method_name}/negative", some_stats['neg'])


#if __name__ == '__main__':
def membership_inference(configs, target_model_idx, offline):
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Gen Canary')
    parser.add_argument('--bs', default=512, type=int)
    parser.add_argument('--size', default=32, type=int)
    parser.add_argument('--canary_size', default=32, type=int)
    parser.add_argument('--name', default='test')
    parser.add_argument('--save_name', default='test')
    #parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--net', default='res18')
    parser.add_argument('--patch', default=4, type=int, help="patch for ViT")
    parser.add_argument('--dimhead', default=512, type=int)
    parser.add_argument('--convkernel', default=8, type=int, help="parameter for convmixer")
    parser.add_argument('--init', default='target_img')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--opt', default='adamw')
    parser.add_argument('--iter', default=30, type=int)
    parser.add_argument('--scheduling', action='store_true')
    #parser.add_argument('--start', default=0, type=int)
    #parser.add_argument('--end', default=50000, type=int)
    parser.add_argument('--in_model_loss', default='target_logits', type=str)
    parser.add_argument('--out_model_loss', default='target_logits', type=str)
    parser.add_argument('--stop_loss', default=None, type=float)
    parser.add_argument('--print_step', action='store_true')
    parser.add_argument('--out_target_class', default=None, type=int)
    parser.add_argument('--aug_strategy', default='try_random_out_class', nargs='+')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--patience', default=3, type=int)
    parser.add_argument('--nowandb', default=True, action='store_true', help='disable wandb')

    parser.add_argument('--logits_mul', default=1, type=int)
    parser.add_argument('--logits_strategy', default='log_logits')
    parser.add_argument('--in_model_loss_weight', default=1, type=float)
    parser.add_argument('--out_model_loss_weight', default=1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--reg_lambda', default=0.001, type=float)
    parser.add_argument('--regularization', default=None)
    parser.add_argument('--stochastic_k', default=2, type=int)
    parser.add_argument('--in_stop_loss', default=None, type=float)
    parser.add_argument('--out_stop_loss', default=None, type=float)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--inner_iter', default=1, type=int)
    parser.add_argument('--canary_aug', action='store_true')
    parser.add_argument('--num_val', default=None, type=int)

    parser.add_argument('--epsilon', default=1, type=float)
    parser.add_argument('--balance_shadow', action='store_true')
    parser.add_argument('--target_logits', default=[10,0]) #, nargs='+', type=float
    parser.add_argument('--save_preds', action='store_true')
    #parser.add_argument('--offline', action='store_true')

    #parser.add_argument('--num_aug', default=10, type=int)
    #parser.add_argument('--num_gen', default=10, type=int) # number of canaries generated during opt
    #parser.add_argument('--num_shadow', default=None, type=int, required=True)
    
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--ref_dis", type=str, default="target_ref", choices=["target_ref", "shadow", \
        "subtar", "subtar_shadow"], help="Data distribution of reference models")
    parser.add_argument("--target_model", type=str, default="wrn28-2", help="Target model name")
    parser.add_argument("--ref_model", type=str, default="wrn28-2", help="Reference model name")
    parser.add_argument("--gpu",  type=str,  default="FromConfigFile", help="GPU device id")
    parser.add_argument("--algs", nargs='+', default="Base", help="MIA algorithms")
    parser.add_argument("--subtar_ratio", type=float, default=0.5, help="Sampling target dataset ratio")

    #parser.add_argument("--number_of_target_models", type=int, default=1, help="The number of target models used in auditing algorithms.")
    parser.add_argument("--target_model_ids", type=int,  nargs='+', default=[0], help="The audited target model id")
    parser.add_argument("--number_of_reference_models", type=int, default=-1, help="The number of reference models used in auditing algorithms.")
    parser.add_argument("--query_num", type=int, default=-1, help="Number of additional queries for each sample in ['query-noise', 'query-augmented'] algorthms.")
    parser.add_argument("--boundary_eva_num", type=int, default=-1, help="Number of evaluation samples in boundary-based auditing methods.")

    args = parser.parse_args()
    
    args.offline = offline
    print(100*'#')
    print('Algorithm Canary Begins')
    print(f"dataset:{args.dataset}, reference data distribution:{args.ref_dis}, subtar_ratio:{args.subtar_ratio}")
    print(f"target model:{args.target_model}, reference model:{args.ref_model}, gpu:{args.gpu}")
    print(100*'-')
    main(args, configs, target_model_idx)

    print(100*'#')
