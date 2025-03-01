##################################################################################
######################  Data auditing Method Information  ########################
##################################################################################

# Auditing algorithm names presented in the paper
METHODS_ALIASES = {
    "target+loss":                  "Metric-loss",
    "target+confidence":            "Metric-conf",
    "target+correctness":           "Metric-corr",
    "target+entropy":               "Metric-ent",
    "target+mentr":                 "Metric-ment",
    
    "network+normal":               "Learn-original",
    "network+topk":                 "Learn-top3",
    "network+sorted":               "Learn-sorted",
    "network+normal_label":         "Learn-label",
    "network+mix":                  "Learn-merge",

    "reference_in_out-non+loss":                    "Model-loss",
    "reference_out-non+loss":                       "Model-calibration",
    "reference_in_out-pdf_fix_var+rescaled_logits": "Model-lira",
    "reference_out-percentile+rescaled_logits":     "Model-fpr",
    "reference_in_out-population_rmia+confidence":  "Model-robust", 

    "query+augmented":                  "Query-augment",
    "query+transfer":                   "Query-transfer",  
    "query+adversarial":                "Query-adv",
    "query+noise":                      "Query-neighbor",
    "query-quantile+rescaled_logits":   "Query-qrm",    
    "query-adv+in_out_pdf+fix_var+rescaled_logits": "Query-ref",
}

METHODS_ALIASES_REVERSE = {METHODS_ALIASES[key]:key for key in METHODS_ALIASES.keys()}


METHODS_REF = {
    "target+loss":                  "Yeom et al. (2018)",
    "target+confidence":            "Song et al. (2019)",
    "target+correctness":           "Leino et al. (2019)",
    "target+entropy":               "Shokri et al. (2017) & Song et al. (2021)",
    "target+mentr":                 "Song et al. (2021)",

    "network+normal":                   "Shokri et al. (2017)",
    "network+topk":                     "Salem et al. (2019)+TopK",
    "network+sorted":                   "Salem et al. (2019)+Sort",
    "network+normal_label":             "Nasr et al. (2018)",
    "network+mix":                      "Amit et al. (2024)",

    "reference_in_out-pdf_fix_var+rescaled_logits":      "Carlini et al. (2022)",
    "reference_out-percentile+rescaled_logits":  "Ye et al. (2022)",
    "reference_in_out-non+loss":        "Sablayrolles et al. (2019)",
    "reference_out-non+loss":           "Watson et al. (2021)",
    "reference_in_out-population_rmia+confidence": "Zarifzadeh et al. (2024)", 

    "query-quantile+rescaled_logits":  "Bertran et al. (2023)",
    "query-adv+in_out_pdf+fix_var+rescaled_logits": "Wen et al. (2023)",
    "query+noise":                      "Jayaraman et al. (2021) & Mattern et al. (2023)",
    "query+augmented":                  "Choquette-Choo et al. (2021)",
    "query+transfer":                   "Li et al. (2021)",  
    "query+adversarial":                "Li et al. (2021) & Choquette-Choo et al. (2021)",
}

ALL_ALG_LIST = list(METHODS_REF.keys())

TARGET_ALG_LIST = ["target+loss", "target+confidence", "target+correctness", "target+entropy", "target+mentr"]
METRIC_ALG_LIST = TARGET_ALG_LIST

REFERENCE_ALG_LIST = ["reference_in_out-pdf_fix_var+rescaled_logits", "reference_out-percentile+rescaled_logits", "reference_in_out-non+loss", "reference_out-non+loss", "reference_in_out-population_rmia+confidence"]
REFERENCE_ALG_OFFLINE_LIST = ["reference_out-pdf_fix_var+rescaled_logits", "reference_out-pdf+rescaled_logits", "reference_out-percentile+rescaled_logits", "reference_out-non+loss", "reference_out-population_rmia+confidence"]

NN_ALG_LIST = ["network+normal", "network+topk", "network+sorted", "network+normal_label", "network+mix"]
QUERY_ALG_LIST = ["query+noise", "query+augmented", "query+adversarial", "query+transfer"]

# algorithms in EXTRA_ALG_LIST are implemented using the code from the original paper.
EXTRA_ALG_LIST = ["query-quantile+rescaled_logits", "query-adv+in_out_pdf+fix_var+rescaled_logits"]
BASE_ALG_LIST = list(set(ALL_ALG_LIST) - set(EXTRA_ALG_LIST))


##################################################################################
################################ Domain Information    ###########################
##################################################################################

DOMAIN_SUPPORT_DATASET = {
    "image": ["minist", "fashionminist", "cifar10-demo", "cifar10", "cifar100", "celeba", \
        "lfw", "cinic10", "imagenet", "pathmnist", "octmnist", "chestmnist", "breastmnist", \
            "dermamnist", "retinamnist", "bloodmnist", "organamnist"], \
    "tabular": ["purchase100", "texas100", "credit", "abalone", "adult", "student", \
         "location", "diabete", "cancer", "insurance", "iris", "breast_cancer"], \
    "text": ["tweet_eval_hate", "rotten_tomatoes", "ag_news", "cola", "sst2", \
        "ecthr_articles", "contract_types",
        "scotus", "imdb", "medical_institutions", "medical_meadow", "twitter_sentiment"],
}

DOMAIN_SUPPORT_MODEL = {
    "image": ["cnn16", "cnn32", "cnn64", "cnn128", "cnn256", "lenet", "alexnet", "wrn28-1", "wrn28-2", "wrn28-10", 
              "vgg16", "vgg11", "mobilenet-v2", "densenet121", "inception-v3", "resnet10", "resnet18", "resnet34", "resnet50"], 
    "tabular": ["catboost", "lightgbm", "lr", "mlp", "mlp16", "mlp32", "mlp64", "mlp128", "mlp256"],
    "text": ["roberta", "flan-t5", "distilbert", "bert", "gpt2", "longformer"],
}