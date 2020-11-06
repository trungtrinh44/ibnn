import logging
import os
from bisect import bisect
from itertools import chain

import numpy as np
import torch
import torch.distributions as D
import torchvision
from sacred import Experiment
from sacred.observers import FileStorageObserver, RunObserver

from datasets import get_data_loader, infinite_wrapper
from models import DetWideResNet28x10, StoWideResNet28x10, StoVGG16, DetVGG16, BayesianVGG16, BayesianWideResNet28x10 ,count_parameters

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

class SetID(RunObserver):
    priority = 50  # very high priority

    def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):
        return f"{config['model_name']}_{config['seed']}_{config['dataset']}_{config['name']}"

EXPERIMENT = 'experiments'
BASE_DIR = EXPERIMENT
ex = Experiment(EXPERIMENT)
ex.observers.append(SetID())
ex.observers.append(FileStorageObserver(BASE_DIR))


@ex.config
def my_config():
    seed = 1 # Random seed
    name = 'name' # Unique name for the folder of the experiment
    model_name = 'StoWideResNet28x10' # Choose with model to train
    # the KL weight will increase from <kl_min> to <kl_max> for <last_iter> iterations.
    kl_weight = {
        'kl_min': 0.0,
        'kl_max': 1.0,
        'last_iter': 200
    }
    batch_size = 128 # Batch size
    prior_mean = 1.0 # Mean of the Gaussian prior
    prior_std = 0.5 # Std of the Gaussian prior
    n_components = 2 # Number of components in the posterior
    # Options of the deterministic weights for the SGD
    det_params = {
        'lr': 0.1, 'weight_decay': 5e-4
    }
    # Options of the variational parameters for the SGD
    sto_params = {
        'lr': 0.1, 'weight_decay': 0.0, 'momentum': 0.0, 'nesterov': False
    }
    # Universal options for the SGD
    sgd_params = {
        'momentum': 0.9, 
        'dampening': 0.0,
        'nesterov': True
    }
    # Options for the Adam optimizer used by the BNN-VI model.
    adam_params = {
        'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0
    }
    num_epochs = 300 # Number of training epoch
    validation = True # Whether of not to use a validation set
    validation_fraction = 0.1 # Size of the validation set
    validate_freq = 5  # Frequency of testing on the validation set
    num_train_sample = 1 # Number of samples drawn from each component during training
    num_test_sample = 1 # Number of samples drawn from each component during testing
    logging_freq = 1 # Logging frequency
    device = 'cuda'
    dropout = 0.3 # for MC-dropout model
    lr_ratio_det = 0.01 # For annealing the learning rate of the deterministic weights
    lr_ratio_sto = 1/3 # For annealing the learning rate of the variational parameters
    milestones = (0.5, 0.9) # First value chooses which epoch to start decreasing the learning rate and the second value chooses which epoch to stop. See the schedule function for more information.
    if not torch.cuda.is_available():
        device = 'cpu'
    posterior_mean_init = (1.0, 0.5) # Mean and std to init the component means in the posterior
    posterior_std_init = (0.05, 0.02) # Mean and std to init the component stds in the posterior
    dataset = 'cifar100' # Dataset of the experiment
    if dataset == 'cifar100' or dataset == 'vgg_cifar100':
        num_classes = 100
    elif dataset == 'cifar10' or dataset == 'vgg_cifar10':
        num_classes = 10

@ex.capture(prefix='kl_weight')
def get_kl_weight(epoch, kl_min, kl_max, last_iter):
    value = (kl_max-kl_min)/last_iter
    return min(kl_max, kl_min + epoch*value)

def schedule(num_epochs, epoch, milestones, lr_ratio):
    t = epoch / num_epochs
    m1, m2 = milestones
    if t <= m1:
        factor = 1.0
    elif t <= m2:
        factor = 1.0 - (1.0 - lr_ratio) * (t - m1) / (m2 - m1)
    else:
        factor = lr_ratio
    return factor

@ex.capture
def get_model(model_name, num_classes, prior_mean, prior_std, n_components, device, sgd_params, det_params, sto_params, dropout, num_epochs, milestones, lr_ratio_det, lr_ratio_sto, posterior_mean_init, posterior_std_init, adam_params):
    if model_name == 'StoWideResNet28x10':
        model = StoWideResNet28x10(num_classes, n_components, prior_mean, prior_std)
        detp = []
        stop = []
        for name, param in model.named_parameters():
            if 'posterior' in name or 'prior' in name:
                stop.append(param)
            else:
                detp.append(param)
        optimizer = torch.optim.SGD(
            [{
                'params': detp,
                **det_params
            },{
                'params': stop,
                **sto_params
            }], **sgd_params)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lambda e: schedule(num_epochs, e, milestones, lr_ratio_det), lambda e: schedule(num_epochs, e, milestones, lr_ratio_sto)])
    elif model_name == 'StoVGG16':
        model = StoVGG16(num_classes, n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init)
        detp = []
        stop = []
        for name, param in model.named_parameters():
            if 'posterior' in name or 'prior' in name:
                stop.append(param)
            else:
                detp.append(param)
        optimizer = torch.optim.SGD(
            [{
                'params': detp,
                **det_params
            },{
                'params': stop,
                **sto_params
            }], **sgd_params)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lambda e: schedule(num_epochs, e, milestones, lr_ratio_det), lambda e: schedule(num_epochs, e, milestones, lr_ratio_sto)])
    elif model_name == 'BayesianVGG16':
        model = BayesianVGG16(num_classes, prior_mean, prior_std)
        optimizer = torch.optim.Adam(
            [{
                'params': model.parameters()
            }], **adam_params)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lambda e: schedule(num_epochs, e, milestones, lr_ratio_det)])
    elif model_name == 'BayesianWideResNet28x10':
        model = BayesianWideResNet28x10(num_classes, prior_mean, prior_std)
        optimizer = torch.optim.Adam(
            [{
                'params': model.parameters()
            }], **adam_params)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lambda e: schedule(num_epochs, e, milestones, lr_ratio_det)])
    
    elif model_name == 'DetVGG16':
        model = DetVGG16(num_classes)
        optimizer = torch.optim.SGD(
            [{
                'params': model.parameters(),
                **det_params
            }], **sgd_params)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lambda e: schedule(num_epochs, e, milestones, lr_ratio_det)])
    elif model_name == 'DetWideResNet28x10':
        model = DetWideResNet28x10(num_classes, dropout_rate=dropout)
        optimizer = torch.optim.SGD(
            [{
                'params': model.parameters(),
                **det_params
            }], **sgd_params)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lambda e: schedule(num_epochs, e, milestones, lr_ratio_det)])
    model.to(device)
    return model, optimizer, scheduler


@ex.capture
def get_dataloader(batch_size, validation, validation_fraction, seed, dataset):
    return get_data_loader(dataset, batch_size, validation, validation_fraction, seed)


@ex.capture
def get_logger(_run, _log):
    fh = logging.FileHandler(os.path.join(BASE_DIR, _run._id, 'train.log'))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    _log.addHandler(fh)
    return _log


@ex.capture
def test_nll(model, loader, device, num_test_sample):
    model.eval()
    with torch.no_grad():
        nll = 0
        acc = 0
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            bnll, pred = model.nll(bx, by, num_test_sample)
            nll += bnll.item() * bx.size(0)
            acc += (pred.exp().mean(1).argmax(-1) == by).sum().item()
        acc /= len(loader.dataset)
        nll /= len(loader.dataset)
    return nll, acc


@ex.automain
def main(_run, model_name, num_train_sample, num_test_sample, device, validation, validate_freq, num_epochs, logging_freq):
    logger = get_logger()
    if validation:
        train_loader, valid_loader, test_loader = get_dataloader()
        logger.info(f"Train size: {len(train_loader.dataset)}, validation size: {len(valid_loader.dataset)}, test size: {len(test_loader.dataset)}")
    else:
        train_loader, test_loader = get_dataloader()
        logger.info(f"Train size: {len(train_loader.dataset)}, test size: {len(test_loader.dataset)}")
    n_batch = len(train_loader)
    model, optimizer, scheduler = get_model()
    count_parameters(model, logger)
    logger.info(str(model))
    checkpoint_dir = os.path.join(BASE_DIR, _run._id, 'checkpoint.pt')
    if model_name.startswith('Sto'):
        model.train()
        best_nll = float('inf')
        for i in range(num_epochs):
            for bx, by in train_loader:
                bx = bx.to(device)
                by = by.to(device)
                optimizer.zero_grad()
                loglike, kl = model.vb_loss(bx, by, num_train_sample)
                klw = get_kl_weight(epoch=i)
                loss = loglike + klw*kl/(n_batch*bx.size(0))
                loss.backward()
                optimizer.step()
                ex.log_scalar('loglike.train', loglike.item(), i)
                ex.log_scalar('kl.train', kl.item(), i)
            scheduler.step()
            if (i+1) % logging_freq == 0:
                logger.info("VB Epoch %d: loglike: %.4f, kl: %.4f, kl weight: %.4f, lr1: %.4f, lr2: %.4f",
                            i, loglike.item(), kl.item(), klw, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
            if (i+1) % validate_freq == 0:
                if validation:
                    with torch.no_grad():
                        nll, acc = test_nll(model, valid_loader)
                    if best_nll >= nll:
                        best_nll = nll
                        torch.save(model.state_dict(), checkpoint_dir)
                        logger.info('Save checkpoint')
                    ex.log_scalar('nll.valid', nll, i)
                    ex.log_scalar('acc.valid', acc, i)
                    logger.info("VB Epoch %d: validation NLL %.4f, acc %.4f", i, nll, acc)
                else:
                    torch.save(model.state_dict(), checkpoint_dir)
                    logger.info('Save checkpoint')
                with torch.no_grad():
                    nll, acc = test_nll(model, test_loader)
                ex.log_scalar('nll.test', nll, i)
                ex.log_scalar('acc.test', acc, i)
                logger.info("VB Epoch %d: test NLL %.4f, acc %.4f", i, nll, acc)
                model.train()
        model.load_state_dict(torch.load(checkpoint_dir, map_location=device))
        tnll = 0
        acc = 0
        nll_miss = 0
        model.eval()
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device)
                by = by.to(device)
                indices = torch.empty(bx.size(0)*num_test_sample, dtype=torch.long, device=bx.device)
                prob = torch.cat([model.forward(bx, num_test_sample, indices=torch.full((bx.size(0)*num_test_sample,), idx, out=indices, device=bx.device, dtype=torch.long)) for idx in range(model.n_components)], dim=1)
                y_target = by.unsqueeze(1).expand(-1, num_test_sample*model.n_components)
                bnll = D.Categorical(logits=prob).log_prob(y_target)
                bnll = torch.logsumexp(bnll, dim=1) - torch.log(torch.tensor(num_test_sample*model.n_components, dtype=torch.float32, device=bnll.device))
                tnll -= bnll.sum().item()
                vote = prob.exp().mean(dim=1)
                pred = vote.argmax(dim=1)

                y_miss = pred != by
                if y_miss.sum().item() > 0:
                    nll_miss -= bnll[y_miss].sum().item()
                acc += (pred == by).sum().item()
        nll_miss /= len(test_loader.dataset) - acc
        tnll /= len(test_loader.dataset)
        acc /= len(test_loader.dataset)
        logger.info("Test data: acc %.4f, nll %.4f, nll miss %.4f",
                    acc, tnll, nll_miss)
    elif model_name.startswith('Bayesian'):
        model.train()
        best_nll = float('inf')
        for i in range(num_epochs):
            for bx, by in train_loader:
                bx = bx.to(device)
                by = by.to(device)
                optimizer.zero_grad()
                loglike, kl = model.vb_loss(bx, by, num_train_sample)
                klw = get_kl_weight(epoch=i)
                loss = loglike + klw*kl/(n_batch*bx.size(0))
                loss.backward()
                optimizer.step()
                ex.log_scalar('loglike.train', loglike.item(), i)
                ex.log_scalar('kl.train', kl.item(), i)
            scheduler.step()
            if (i+1) % logging_freq == 0:
                logger.info("VB Epoch %d: loglike: %.4f, kl: %.4f, kl weight: %.4f, lr: %.4f",
                            i, loglike.item(), kl.item(), klw, optimizer.param_groups[0]['lr'])
            if (i+1) % validate_freq == 0:
                if validation:
                    with torch.no_grad():
                        nll, acc = test_nll(model, valid_loader)
                    if best_nll >= nll:
                        best_nll = nll
                        torch.save(model.state_dict(), checkpoint_dir)
                        logger.info('Save checkpoint')
                    ex.log_scalar('nll.valid', nll, i)
                    ex.log_scalar('acc.valid', acc, i)
                    logger.info("VB Epoch %d: validation NLL %.4f, acc %.4f", i, nll, acc)
                else:
                    torch.save(model.state_dict(), checkpoint_dir)
                    logger.info('Save checkpoint')
                with torch.no_grad():
                    nll, acc = test_nll(model, test_loader)
                ex.log_scalar('nll.test', nll, i)
                ex.log_scalar('acc.test', acc, i)
                logger.info("VB Epoch %d: test NLL %.4f, acc %.4f", i, nll, acc)
                model.train()
        model.load_state_dict(torch.load(checkpoint_dir, map_location=device))
        tnll = 0
        acc = 0
        nll_miss = 0
        model.eval()
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device)
                by = by.to(device)
                prob = model.forward(bx, num_test_sample)
                y_target = by.unsqueeze(1).expand(-1, num_test_sample)
                bnll = D.Categorical(logits=prob).log_prob(y_target)
                bnll = torch.logsumexp(bnll, dim=1) - torch.log(torch.tensor(num_test_sample, dtype=torch.float32, device=bnll.device))
                tnll -= bnll.sum().item()
                vote = prob.exp().mean(dim=1)
                pred = vote.argmax(dim=1)

                y_miss = pred != by
                if y_miss.sum().item() > 0:
                    nll_miss -= bnll[y_miss].sum().item()
                acc += (pred == by).sum().item()
        nll_miss /= len(test_loader.dataset) - acc
        tnll /= len(test_loader.dataset)
        acc /= len(test_loader.dataset)
        logger.info("Test data: acc %.4f, nll %.4f, nll miss %.4f",
                    acc, tnll, nll_miss)
    elif model_name.startswith('Det'):
        model.train()
        best_nll = float('inf')
        for i in range(num_epochs):
            for bx, by in train_loader:
                optimizer.zero_grad()
                bx = bx.to(device)
                by = by.to(device)
                pred = model(bx)
                loss = torch.nn.functional.nll_loss(pred, by)
                loss.backward()
                optimizer.step()
                ex.log_scalar("nll.train", loss.item(), i)
            scheduler.step()
            if (i+1) % logging_freq == 0:
                logger.info("Epoch %d: train %.4f, lr %.4f", i, loss.item(), optimizer.param_groups[0]['lr'])
            if (i+1) % validate_freq == 0:
                model.eval()
                if validation:
                    with torch.no_grad():
                        nll = 0
                        acc = 0
                        for bx, by in valid_loader:
                            bx = bx.to(device)
                            by = by.to(device)
                            pred = model(bx)
                            nll += torch.nn.functional.nll_loss(pred, by).item() * len(by)
                            acc += (pred.argmax(1) == by).sum().item()
                        nll /= len(valid_loader.dataset)
                        acc /= len(valid_loader.dataset)
                    if best_nll >= nll:
                        best_nll = nll
                        torch.save(model.state_dict(), checkpoint_dir)
                        logger.info('Save checkpoint')
                    ex.log_scalar('nll.valid', nll, i)
                    ex.log_scalar('acc.valid', acc, i)
                    logger.info("Epoch %d: validation %.4f, %.4f", i, nll, acc)
                else:
                    torch.save(model.state_dict(), checkpoint_dir)
                    logger.info('Save checkpoint')
                with torch.no_grad():
                    nll = 0
                    acc = 0
                    for bx, by in test_loader:
                        bx = bx.to(device)
                        by = by.to(device)
                        pred = model(bx)
                        nll += torch.nn.functional.nll_loss(
                            pred, by).item() * len(by)
                        acc += (pred.argmax(1) == by).sum().item()
                    nll /= len(test_loader.dataset)
                    acc /= len(test_loader.dataset)
                ex.log_scalar('nll.test', nll, i)
                ex.log_scalar('acc.test', acc, i)
                logger.info("Epoch %d: test %.4f, acc %.4f", i, nll, acc)
                model.train()
        model.load_state_dict(torch.load(checkpoint_dir, map_location=device))
        tnll = 0
        acc = 0
        nll_miss = 0
        model.eval()
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device)
                by = by.to(device)
                prob = model(bx)
                pred = prob.argmax(dim=1)
                tnll += torch.nn.functional.nll_loss(
                    prob, by).item() * len(by)
                y_miss = pred != by
                if y_miss.sum().item() > 0:
                    prob_miss = prob[y_miss]
                    by_miss = by[y_miss]
                    nll_miss += torch.nn.functional.nll_loss(
                        prob_miss, by_miss).item() * len(by_miss)
                acc += (pred == by).sum().item()
        nll_miss /= len(test_loader.dataset) - acc
        tnll /= len(test_loader.dataset)
        acc /= len(test_loader.dataset)
        logger.info("Test data: acc %.4f, nll %.4f, nll miss %.4f",
                    acc, tnll, nll_miss)
