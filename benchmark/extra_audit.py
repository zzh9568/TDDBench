import numpy as np
import lightgbm as lgb

from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import auc, roc_curve
from catboost import CatBoostRegressor
from extra_baselines.quantile_main import membership_inference as quantile_mi
from extra_baselines.canary_main   import membership_inference as canary_mi
from info import DOMAIN_SUPPORT_DATASET


def extra_audit_benchmark(alg, configs, target_model_idx):
    if alg == "query-quantile+rescaled_logits":
        if configs["audit"]["reference_model"] not in ['catboost', 'lightgbm', 'lr']:
            quantile_mi(configs, target_model_idx)
    elif alg == "query-adv+in_out_pdf+fix_var+rescaled_logits":
        if configs["audit"]["dataset"] in DOMAIN_SUPPORT_DATASET["image"]:
            canary_mi(configs, target_model_idx, offline=False)
        else:
            print(f"canary.py only supports image dataset now. Although no error is reported, no audit results are returned.")
    elif alg == "query-adv+out_pdf+fix_var+rescaled_logits":
        if configs["audit"]["dataset"] in DOMAIN_SUPPORT_DATASET["image"]:
            canary_mi(configs, target_model_idx, offline=True)
        else:
            print(f"canary.py only supports image dataset now. Although no error is reported, no audit results are returned.")
    else:
        pass

def tabular_quantile_gdbt(
    quantile_model_name, 
    membership, 
    target_signal,
    target_data, 
    population_signal, 
    population_data, 
    seed,
    device,
):
    assert quantile_model_name in ['catboost', 'lightgbm', 'lr'], f"{quantile_model_name} is not supported in function tabular_quantile_gbdt"
    if quantile_model_name == 'catboost':
        # quantile_model = CatBoostRegressor(learning_rate=0.05, iterations=10000, loss_function="RMSEWithUncertainty", \
        #     posterior_sampling=True, random_seed=seed, )
        quantile_model = CatBoostRegressor(learning_rate=0.05, iterations=10000, loss_function="RMSEWithUncertainty", \
            posterior_sampling=True, random_seed=seed, thread_count=2, ) #task_type="GPU", devices=device
        quantile_model.fit(population_data.numpy(), population_signal, verbose=True)
        conf_test = quantile_model.predict(target_data.numpy())
        mu = conf_test[:, 0]
        sigma = conf_test[:, 1]
    
    elif quantile_model_name == 'lr':
        quantile_model = LinearRegression(n_jobs=2).fit(population_data.numpy(), population_signal)
        mu = quantile_model.predict(target_data.numpy())
        sigma = np.var(population_signal)

    elif quantile_model_name == 'lightgbm':
        quantile_model = lgb.LGBMRegressor(random_seed=seed, n_jobs=2, max_depth=3, learning_rate=0.1, verbosity=-1)
        quantile_model.fit(population_data.numpy(), population_signal)
        mu = quantile_model.predict(target_data.numpy())
        sigma = np.var(population_signal)

    else:
        raise ValueError
    
    #pr_out = -norm.logpdf(target_signal[0], mu, np.sqrt(sigma) + 1e-30)
    pr_out = norm.cdf(target_signal, mu, np.sqrt(sigma) + 1e-30)
    score = pr_out
    fpr_list, tpr_list, _ = roc_curve(membership, -score)

    ref_in_acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)
    ref_in_roc_auc = auc(fpr_list, tpr_list)
    ref_in_low = tpr_list[np.where(fpr_list < 0.001)[0][-1]]

    print(
        f"quantile_gdbt AUC: %.4f, Accuracy: %.4f, TPR@0.1%%FPR: %.4f"
        % (ref_in_roc_auc, ref_in_acc, ref_in_low)
    )
    return fpr_list, tpr_list, ref_in_roc_auc
