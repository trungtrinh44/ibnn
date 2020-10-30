# iBNN

Official Pytorch implementation of the implicit BNN model from the paper

[Scalable Bayesian neural networks by layer-wise input augmentation](https://arxiv.org/abs/2010.13498)

by Trung Trinh, Samuel Kaski, Markus Heinonen

## Installation

```bash
pip install -r requirements.txt
```

## File Structure

```
.
+-- models/ (Folder containing all model definitions)
|   +-- resnet.py (containing the WideResNet model)
|   +-- vgg.py (containing the VGG model)
|   +-- utils.py (utility functions and modules)
+-- datasets.py (containing functions to load data)
+-- train.py (script for training)
+-- test.py (script for testing)
+-- ood_test.py (script for out-of-distribution testing)
```

## Command to replicate the result

Training WideResNet-28x10 on CIFAR-10
```bash
python train.py with  model_name=StoWideResNet28x10 validation=False \
        num_epochs=300 validate_freq=15 logging_freq=1 'kl_weight.kl_min=0.0' 'kl_weight.kl_max=1.0' 'kl_weight.last_iter=200' \
        lr_ratio_det=0.01 lr_ratio_sto=1.0 prior_std=0.1 prior_mean=1.0 'det_params.weight_decay=5e-4' \
        num_test_sample=1 dropout=0.0 n_components=<NUMBER_OF_COMPONENTS> dataset=cifar10 \
        'det_params.lr=0.1' 'sto_params.lr'=2.4 'sto_params.weight_decay=0.0' \
        'sto_params.momentum=0.9' 'sto_params.nesterov=True' num_train_sample=2 \
        'sgd_params.nesterov'=True 'milestones=(0.50,0.90)' name=<UNIQUE_NAME_FOR_THE_EXPERIMENT> seed=<RANDOM_SEED>
```
Training WideResNet-28x10 on CIFAR-100
```bash
python train.py with  model_name=StoWideResNet28x10 validation=False \
        num_epochs=300 validate_freq=15 logging_freq=1 'kl_weight.kl_min=0.0' 'kl_weight.kl_max=1.0' 'kl_weight.last_iter=200' \
        lr_ratio_det=0.01 lr_ratio_sto=1.0 prior_std=0.1 prior_mean=1.0 'det_params.weight_decay=5e-4' \
        num_test_sample=1 dropout=0.0 n_components=<NUMBER_OF_COMPONENTS> dataset=cifar100 seed=<RANDOM_SEED> \
        'det_params.lr=0.1' 'sto_params.momentum=0.9' 'sto_params.nesterov=True' 'sto_params.lr'=4.8 'sto_params.weight_decay=0.0' 'num_train_sample'=2 \
        'sgd_params.nesterov'=True 'milestones=(0.50,0.90)' name=<UNIQUE_NAME_FOR_THE_EXPERIMENT>
```
Training VGG16 on CIFAR-10
```bash
python train.py with model_name=StoVGG16 validation=False \
        num_epochs=300 validate_freq=15 logging_freq=1 'kl_weight.kl_min=0.0' 'kl_weight.kl_max=1.0' 'kl_weight.last_iter=200' \
        lr_ratio_det=0.01 lr_ratio_sto=1.0 prior_std=0.3 prior_mean=1.0 'det_params.weight_decay=5e-4' num_test_sample=1 \
        n_components=<NUMBER_OF_COMPONENTS> dataset=vgg_cifar10 'posterior_mean_init=(1.0,0.75)' \
        'det_params.lr=0.05' 'sto_params.lr'=1.2 'sto_params.weight_decay=0.0' 'sto_params.momentum=0.9' 'sto_params.nesterov=True' 'num_train_sample'=2 \
        'sgd_params.nesterov'=True 'milestones=(0.50,0.90)' name=<UNIQUE_NAME_FOR_THE_EXPERIMENT> seed=<RANDOM_SEED>
```
Training VGG16 on CIFAR-100
```bash
python train.py with model_name=StoVGG16 validation=False \
    num_epochs=300 validate_freq=15 logging_freq=1 'kl_weight.kl_min=0.0' 'kl_weight.kl_max=1.0' 'kl_weight.last_iter=200' \
    lr_ratio_det=0.01 lr_ratio_sto=1.0 prior_std=0.3 prior_mean=1.0 'det_params.weight_decay=3e-4' num_test_sample=1 \
    n_components=<NUMBER_OF_COMPONENTS> dataset=vgg_cifar100 'posterior_mean_init=(1.0,0.75)' 'posterior_std_init=(0.05,0.02)' \
    'det_params.lr=0.05' 'sto_params.lr'=1.6 'sto_params.weight_decay=0.0' 'sto_params.momentum=0.9' 'sto_params.nesterov=True' 'num_train_sample'=2 \
    'sgd_params.nesterov'=True 'milestones=(0.50,0.90)' name=<UNIQUE_NAME_FOR_THE_EXPERIMENT> seed=<RANDOM_SEED>
```
For more information on each training option, please read the comments in the `train.py` file.
Each experiment will be stored in a subfolder of the `experiments` folder.

To test the model
```bash
python test.py <EXPERIMENT_FOLDER> -n 5 -b 128
```
where `-n` option defines the number of samples to use in each component, and `-b` option defines the batch size. The test result will be in the `<EXPERIMENT_FOLDER>/<DATASET>/result.json` file.