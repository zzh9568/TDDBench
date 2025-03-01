import os
import warnings

warnings.simplefilter("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

import argparse
import random
import shutil
import sys
import math
import pickle
from pathlib import Path
from glob import glob
sys.path.insert(0, "../basic/")
sys.path.insert(0, "../basic/extra_baselines")

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from models import INPUT_OUTPUT_SHAPE
from dataset import DATASET_TYPE
NUM_CPUS_PER_WORKER = 7

from quantile_utils import LightningQMIA, CustomDataModule, CustomWriter
from quantile_mia.image_QMIA.analysis_utils import plot_performance_curves


def plot_model(
    args,
    checkpoint_path,
    fig_name="best",
    recompute_predictions=True,
    return_mean_logstd=False,
):
    configs = args.configs
    if return_mean_logstd:
        fig_name = "raw_{}".format(fig_name)
        prediction_output_dir = os.path.join(
            args.root_checkpoint_path,
            "raw_predictions",
            fig_name,
        )
    else:
        prediction_output_dir = os.path.join(
            args.root_checkpoint_path,
            "predictions",
            fig_name,
        )
    print("Saving predictions to", prediction_output_dir)

    os.makedirs(prediction_output_dir, exist_ok=True)
    if (
        recompute_predictions
        or len(glob(os.path.join(prediction_output_dir, "*.pt"))) == 0
    ):
        try:
            if os.environ["LOCAL_RANK"] == "0":
                shutil.rmtree(prediction_output_dir)
        except:
            pass
        # Get model and data
        datamodule = CustomDataModule(
            configs=configs,
            mode="eval",
            num_workers=7,
            image_size=args.image_size,
            batch_size=args.batch_size,
            data_root=None,
        )

        # reload quantile model
        print("reloading from", checkpoint_path)
        lightning_model = LightningQMIA.load_from_checkpoint(checkpoint_path)
        if return_mean_logstd:
            lightning_model.return_mean_logstd = True
        pred_writer = CustomWriter(
            output_dir=prediction_output_dir, write_interval="epoch"
        )

        trainer = pl.Trainer(
            logger=False,
            max_epochs=1,
            accelerator="auto" if torch.cuda.is_available() else "cpu",
            callbacks=[pred_writer],
            devices=[int(configs["audit"]["device"][5:])],#-1 for all gpus
            #devices="auto",
            enable_progress_bar=True,
        )
        predict_results = trainer.predict(
            lightning_model, datamodule, return_predictions=True
        )
        trainer.strategy.barrier()
        if trainer.global_rank != 0:
            return

    # Trainer predict in DDP does not return predictions. To use distributed predicting, we instead save the prediciton outputs to file then concatenate manually
    predict_results = None
    for file in glob(os.path.join(prediction_output_dir, "*.pt")):
        rank_predict_results = torch.load(file)
        if predict_results is None:
            predict_results = rank_predict_results
        else:
            for r, p in zip(rank_predict_results, predict_results):
                p.extend(r)

    def join_list_of_tuples(list_of_tuples):
        n_tuples = len(list_of_tuples[0])
        result = []
        for _ in range(n_tuples):
            try:
                result.append(torch.concat([t[_] for t in list_of_tuples]))
            except:
                result.append(torch.Tensor([t[_] for t in list_of_tuples]))
        return result

    print([len(predict_results[i]) for i in range(len(predict_results))])
    (
        private_predicted_quantile_threshold,
        private_target_score,
        private_loss,
        private_base_acc1,
        private_base_acc5,
    ) = join_list_of_tuples(predict_results[-1])
    (
        test_predicted_quantile_threshold,
        test_target_score,
        test_loss,
        test_base_acc1,
        test_base_acc5,
    ) = join_list_of_tuples(predict_results[1])

    model_target_quantiles = np.sort(
        1.0
        - np.logspace(args.low_quantile, args.high_quantile, args.n_quantile).flatten()
        if args.use_log_quantile
        else np.linspace(
            args.low_quantile, args.high_quantile, args.n_quantile
        ).flatten()
    )
    if return_mean_logstd:
        # model_target_quantiles = model_target_quantiles[1:-1]
        dislocated_quantiles = torch.erfinv(
            2 * torch.Tensor(model_target_quantiles) - 1
        ).reshape([1, -1]) * math.sqrt(2)

        public_mu = test_predicted_quantile_threshold[:, 0].reshape([-1, 1])
        public_std = torch.exp(test_predicted_quantile_threshold[:, 1]).reshape([-1, 1])
        test_predicted_quantile_threshold = (
            public_mu + public_std * dislocated_quantiles
        )

        private_mu = private_predicted_quantile_threshold[:, 0].reshape([-1, 1])
        private_std = torch.exp(private_predicted_quantile_threshold[:, 1]).reshape(
            [-1, 1]
        )
        private_predicted_quantile_threshold = (
            private_mu + private_std * dislocated_quantiles
        )
    print(
        "Model accuracy on training set {:.2f}%".format(
            np.mean(private_base_acc1.numpy())
        )
    )
    print("Model accuracy on test set  {:.2f}%".format(np.mean(test_base_acc1.numpy())))

    plot_result = plot_performance_curves(
        np.asarray(private_target_score),
        np.asarray(test_target_score),
        private_predicted_score_thresholds=np.asarray(
            private_predicted_quantile_threshold
        ),
        public_predicted_score_thresholds=np.asarray(test_predicted_quantile_threshold),
        model_target_quantiles=model_target_quantiles,
        model_name="Quantile Regression",
        use_logscale=True,
        fontsize=12,
        savefig_path=f"{args.root_checkpoint_path}/results.png"
    )

    ##############################################################################################
    ################################DELETE Prediction Results#####################################
    ##############################################################################################

    return prediction_output_dir,plot_result


def argparser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(description="QMIA attack")

    # Options for pinball loss attack. Multi head regression 
    # where each head is a different quantile target
    parser.add_argument(
        "--n_quantile", type=int, default=41, help="number of quantile targets"
    )
    parser.add_argument(
        "--low_quantile",
        type=float,
        default=-4,
        help="minimum quantile, in absolute scale if use_log_quantile is false, otherwise just the exponent (0.01 vs -2)",
    )
    parser.add_argument(
        "--high_quantile",
        type=float,
        default=0,
        help="maximum quantile, in absolute scale if use_log_quantile is false, otherwise just the exponent (0.01 vs -2) ",
    )
    parser.add_argument(
        "--use_log_quantile",
        type=str2bool,
        default=True,
        help="use log scale for quantile sweep",
    )
    # Options for gaussian (mean, std) modelling of score distribution
    parser.add_argument(
        "--use_gaussian",
        type=str2bool,
        default=True, #False
        help="use gaussian parametrization",
    )
    # Optionally train a label-dependent regressor q(x,y) instead of q(x)
    parser.add_argument(
        "--use_target_dependent_scoring",
        type=str2bool,
        default=False,
        help="Use target label y for quantile predictor (q(x,y))?",
    )

    # Training arguments
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="l2 regularization"
    )
    parser.add_argument(
        "--opt", type=str, default="adamw", help="otimizer {sgd, adam, adamw}"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate") #1e-4
    parser.add_argument(
        "--scheduler", type=str, default="", help="learning rate scheduler"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="gradient clipping"
    )

    # Score configuration
    parser.add_argument(
        "--use_hinge_score",
        type=str2bool,
        default="True",
        help="use hinge loss of logits as score? otherwise use probability",
    )
    parser.add_argument(
        "--use_target_label",
        type=str2bool,
        default="True",
        help="use target label or argmax label of model output",
    )
    parser.add_argument(
        "--use_target_inputs",
        type=str2bool,
        default=False,
        help="use targets as input to the quantile model",
    )
    parser.add_argument(
        "--return_mean_logstd",
        type=str2bool,
        default=False,
        help="just for plotting, stick to false",
    )

    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--ref_dis", type=str, default="target_ref", choices=["target_ref", "shadow", \
        "subtar", "subtar_shadow"], help="Data distribution of reference models")
    parser.add_argument("--target_model", type=str, default="wrn28-2", help="Target model name")
    parser.add_argument("--ref_model", type=str, default="wrn28-2", help="Reference model name")
    parser.add_argument("--gpu",  type=str,  default="FromConfigFile", help="GPU device id")
    parser.add_argument("--algs", nargs='+', default="Base", help="MIA algorithms")
    parser.add_argument("--subtar_ratio", type=float, default=0.5, help="Sampling target dataset ratio")
    parser.add_argument("--epochs",     type=int,  default=30, help="epochs") #30

    parser.add_argument("--target_model_ids", type=int,  nargs='+', default=[0], help="The audited target model id")
    parser.add_argument("--number_of_reference_models", type=int, default=-1, help="The number of reference models used in auditing algorithms.")
    parser.add_argument("--query_num", type=int, default=-1, help="Number of additional queries for each sample in ['query-noise', 'query-augmented'] algorthms.")
    parser.add_argument("--boundary_eva_num", type=int, default=-1, help="Number of evaluation samples in boundary-based auditing methods.")

    args = parser.parse_args()
    return args

#if __name__ == "__main__":
def membership_inference(configs, target_model_idx):
    args = argparser()

    print(100*'#')
    print('Algorithm Quantile Begins')
    print(f'''dataset:{args.dataset}, input gpu id:{args.gpu}, used gpu id:{int(configs["audit"]["device"][5:])}''')
    print(100*'-')

    seed = configs["random_seed"]
    configs["train"]["random_seed"] = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    meta_log_dir = configs["train"]["log_dir"]
    dataset_name, ref_data_dis = configs["audit"]["dataset"], configs["audit"]["ref_data_dis"]
    subtar_ratio = configs["audit"]["subtar_ratio"]
    target_model_name, reference_model_name = configs["audit"]["target_model"], configs["audit"]["reference_model"]
    meta_log_dir = configs["train"]["log_dir"]

    args.root_checkpoint_path = f'''{configs["quantile_root_checkpoint_path"]}/use_log_quantile_{args.use_log_quantile}/target_model_idx_{target_model_idx}'''
    
    configs["audit"]["target_model_split_info_dir"] = f'''{meta_log_dir}/{configs["data"]["dataset_type"]}/{dataset_name}/{target_model_name}/target/models_metadata/0.pkl'''
        
    if os.path.exists(args.root_checkpoint_path):
        shutil.rmtree(args.root_checkpoint_path)
    args.image_size = None

    args.batch_size = configs["train"]["batch_size"]


    args.base_architecture = target_model_name
    args.architecture = reference_model_name
    args.configs = configs

    ##########  convnext-tiny (slow)  ##################################################
    # setting multi gpu -> may report error
    # if DATASET_TYPE[configs["data"]["dataset"]] == "image":
    #     args.architecture = "convnext-tiny-224" #configs['train']["model_name"]
    #     if args.architecture == "convnext-tiny-224":
    #         args.batch_size = 16
    #         args.lr = 1e-4
    #         args.use_gaussian = False
    #     configs["architecture"] = "convnext-tiny-224"
    ##########  convnext-tiny (slow)  ##################################################

    datamodule = CustomDataModule(
        configs=configs,
        mode="mia",
        num_workers=6,
        image_size=None, #args.image_size,
        batch_size=args.batch_size,
        data_root=None,
    )


    if datamodule.dataset_type == "text":
        gradient_accumulation_steps = int(configs["train"]["batch_size"] / configs["train"]["per_device_train_batch_size"])
    else:
        gradient_accumulation_steps = 1

    num_base_classes = datamodule.num_base_classes


    if datamodule.dataset_type == "text":
        base_model_path = f'''{meta_log_dir}/{configs["data"]["dataset_type"]}/{dataset_name}/{target_model_name}/target/model_{target_model_idx}'''
    elif args.base_architecture == "catboost":
        base_model_path = f'''{meta_log_dir}/{configs["data"]["dataset_type"]}/{dataset_name}/{target_model_name}/target/model_{target_model_idx}.model'''
    else:
        base_model_path = f'''{meta_log_dir}/{configs["data"]["dataset_type"]}/{dataset_name}/{target_model_name}/target/model_{target_model_idx}.pkl'''

    # Create lightning model
    lightning_model = LightningQMIA(
        dataset_name=dataset_name,
        architecture=args.architecture,
        base_architecture=args.base_architecture,
        image_size=None, #args.image_size,
        hidden_dims=[512, 512],
        num_base_classes=num_base_classes,
        freeze_embedding=False,
        low_quantile=args.low_quantile,
        high_quantile=args.high_quantile,
        n_quantile=args.n_quantile,
        use_logscale=args.use_log_quantile,
        optimizer_params={"opt_type": args.opt},
        base_model_path=base_model_path,
        rearrange_on_predict=not args.use_gaussian,
        use_hinge_score=args.use_hinge_score,
        use_target_label=args.use_target_label,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_gaussian=args.use_gaussian,
        use_target_dependent_scoring=args.use_target_dependent_scoring,
        use_target_inputs=args.use_target_inputs,
    )

    metric = "ptl/val_loss"
    mode = "min"
    checkpoint_dir = os.path.dirname(args.root_checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.root_checkpoint_path,
        monitor=metric,
        mode=mode,
        save_top_k=1,
        auto_insert_metric_name=False,
        filename="best_val_loss",
    )
    callbacks = [checkpoint_callback] + [TQDMProgressBar(refresh_rate=100)]
    trainer = pl.Trainer(
        accumulate_grad_batches=gradient_accumulation_steps,
        logger=False,
        max_epochs=args.epochs,
        accelerator="gpu",
        callbacks=callbacks,
        devices=[int(configs["audit"]["device"][5:])],
        gradient_clip_val=args.grad_clip,
        default_root_dir=os.path.join(args.root_checkpoint_path, "tune"),
    )
    trainer.fit(lightning_model, datamodule=datamodule)

    print(checkpoint_callback.best_model_path)

    ####################################plot###########################################
    dst_checkpoint_path = os.path.join(args.root_checkpoint_path, "best_val_loss.ckpt")

    # plot best trial
    prediction_output_dir, _ = plot_model(
        args,
        dst_checkpoint_path,
        "best",
        recompute_predictions=False,
        return_mean_logstd=args.return_mean_logstd,
    )

    with open(f"{args.root_checkpoint_path}/results.pkl", 'rb') as f:
        results = pickle.load(f)

    results = results['model']
    print(f"AUROC:{results['auc']}")
    fpr = 1 - results['tnr']
    tpr = results['tpr']
    positive_sample = int(np.unique(np.array([results['positive']])))
    negative_sample = int(np.unique(np.array([results['negative']])))

    from info import METHODS_ALIASES


    quantile_method_name = METHODS_ALIASES["query-quantile+rescaled_logits"]
    report_dir = f'''{configs["audit"]["report_dir"]}/target_model_idx_{target_model_idx}'''
    
    Path(f"{report_dir}/{quantile_method_name}").mkdir(parents=True, exist_ok=True)
    np.save(f"{report_dir}/{quantile_method_name}/fpr", fpr)
    np.save(f"{report_dir}/{quantile_method_name}/tpr", tpr)
    np.save(f"{report_dir}/{quantile_method_name}/positive", positive_sample)
    np.save(f"{report_dir}/{quantile_method_name}/negative", negative_sample)
    
    print(args.root_checkpoint_path)
    shutil.rmtree(args.root_checkpoint_path)
    os.makedirs(prediction_output_dir)

    print(100*'#')
