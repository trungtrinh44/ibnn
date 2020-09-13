import logging
import os
from bisect import bisect

import numpy as np
import torch
import torch.distributions as D

import torchvision
from sacred import Experiment
from sacred.observers import FileStorageObserver

from datasets import get_data_loader, infinite_wrapper
from models import DetWideResNet, StoWideResNet, DropWideResNet, count_parameters

EXPERIMENT = 'wrn_cifar100'
BASE_DIR = os.path.join('implicit_runs', EXPERIMENT)
ex = Experiment(EXPERIMENT)
ex.observers.append(FileStorageObserver(BASE_DIR))


@ex.config
def my_config():
    seed = 1
    model_type = 'stochastic'
    kl_weight = {
        'kl_min': 0.0,
        'kl_max': 1.0,
        'last_iter': 62560
    }
    batch_size = 128
    init_prior_mean = 0.0
    init_prior_std = 1.0
    n_components = 2
    det_params = {
        'lr': 0.1, 'weight_decay': 5e-4
    }
    sto_params = {
        'lr': 0.1, 'weight_decay': 0.0
    }
    lr_scheduler = {
        'milestones': [23460, 46920, 62560],
        'gamma': 0.2
    }
    sgd_params = {
        'momentum': 0.9, 
        'dampening': 0.0,
        'nesterov': True
    }
    num_iterations = 62600
    n_per_block = 4
    k_factor = 2
    init_method = 'normal'
    activation = 'relu'
    validation = True
    validation_fraction = 0.2
    validate_freq = 1000  # calculate validation frequency
    num_train_sample = 1
    num_kl_sample = 1
    num_test_sample = 1
    logging_freq = 500
    posterior_type = 'mixture_gaussian'
    device = 'cuda'
    kl_div_nbatch = True
    no_kl = False
    dropout = 0.3 # for mc-dropout model
    if not torch.cuda.is_available():
        device = 'cpu'
    start_from_deterministic_checkpoint = ''

@ex.capture(prefix='kl_weight')
def get_kl_weight(kl_min, kl_max, last_iter):
    kl = kl_min
    value = (kl_max-kl_min)/last_iter
    while 1:
        yield min(kl_max, kl)
        kl += value

@ex.capture
def get_model(model_type, n_per_block, k_factor, init_method, activation, init_prior_mean, init_prior_std, n_components,
              device, sgd_params, lr_scheduler, det_params, sto_params, dropout):
    if model_type == 'stochastic':
        model = StoWideResNet(32, 3, 100, n_per_block=n_per_block, k=k_factor, init_method=init_method, 
                              prior_mean=init_prior_mean, prior_std=init_prior_std, n_components=n_components)
        detp = []
        stop = []
        for name, param in model.named_parameters():
            if 'posterior_mean' in name or 'posterior_std' in name:
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
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lambda e: lr_scheduler['gamma']**(bisect(lr_scheduler['milestones'], e)), lambda e: 1])
    elif model_type == 'dropout':
        model = DropWideResNet(32, 3, dropout, 100, n_per_block, k_factor, init_method)
        optimizer = torch.optim.SGD(
            [{
                'params': model.parameters(),
                **det_params
            }], **sgd_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **lr_scheduler)
    else:
        model = DetWideResNet(32, 3, dropout, 100, n_per_block, k_factor, init_method)
        optimizer = torch.optim.SGD(
            [{
                'params': model.parameters(),
                **det_params
            }], **sgd_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **lr_scheduler)
    model.to(device)
    return model, optimizer, scheduler


@ex.capture
def get_dataloader(batch_size, validation, validation_fraction, seed):
    return get_data_loader('wrn_cifar100', batch_size, validation, validation_fraction, seed)


@ex.capture
def get_logger(_run, _log):
    fh = logging.FileHandler(os.path.join(BASE_DIR, _run._id, 'train.log'))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    _log.addHandler(fh)
    return _log


@ex.capture
def test_nll(model, loader, device, num_test_sample, model_type):
    model.eval()
    with torch.no_grad():
        nll = 0
        acc = 0
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            bnll, pred = model.negative_loglikelihood(bx, by, num_test_sample, True)
            nll += bnll.item() * bx.size(0)
            acc += (pred.exp().mean(1).argmax(-1) == by).sum().item()
        acc /= len(loader.dataset)
        nll /= len(loader.dataset)
    return nll, acc


@ex.automain
def main(_run, model_type, num_train_sample, num_test_sample, device, validation, validate_freq, num_iterations, logging_freq, kl_div_nbatch, no_kl, num_kl_sample, start_from_deterministic_checkpoint):
    torch.backends.cudnn.benchmark = True
    logger = get_logger()
    if validation:
        train_loader, valid_loader, test_loader = get_dataloader()
        logger.info(f"Train size: {len(train_loader.dataset)}, validation size: {len(valid_loader.dataset)}, test size: {len(test_loader.dataset)}")
    else:
        train_loader, test_loader = get_dataloader()
        logger.info(f"Train size: {len(train_loader.dataset)}, test size: {len(test_loader.dataset)}")
    n_batch = len(train_loader) if kl_div_nbatch else len(train_loader.dataset)
    train_loader = infinite_wrapper(train_loader)
    model, optimizer, scheduler = get_model()
    if start_from_deterministic_checkpoint != '':
        model.load_state_dict(
            torch.load(start_from_deterministic_checkpoint, map_location=device), strict=False
        )
        logger.info(f'Load from deterministic checkpoint {start_from_deterministic_checkpoint}')
    count_parameters(model, logger)
    logger.info(str(model))
    checkpoint_dir = os.path.join(BASE_DIR, _run._id, 'checkpoint.pt')
    kl_weight = get_kl_weight()
    # First train mll
    if model_type == 'stochastic':
        model.train()
        train_iter = enumerate(train_loader, 1)
        best_nll = float('inf')
        for i, (bx, by) in train_iter:
            if i > num_iterations:
                break
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            loglike, kl = model.vb_loss(bx, by, num_train_sample, num_kl_sample, no_kl)
            klw = next(kl_weight)
            loss = loglike + klw*kl/n_batch
            loss.backward()
            optimizer.step()
            scheduler.step()
            ex.log_scalar('loglike.train', loglike.item(), i)
            ex.log_scalar('kl.train', kl.item(), i)
            if i % logging_freq == 0:
                logger.info("VB Epoch %d: loglike: %.4f, kl: %.4f, kl weight: %.4f, lr: %.4f",
                            i, loglike.item(), kl.item(), klw, optimizer.param_groups[0]['lr'])
            if i % validate_freq == 0:
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
                y_target = by.unsqueeze(1).repeat(1, num_test_sample*model.n_components)
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
    elif model_type == 'dropout':
        model.train()
        train_iter = enumerate(train_loader, 1)
        best_nll = float('inf')
        for i, (bx, by) in train_iter:
            if i > num_iterations:
                break
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            loss = model.train_loss(bx, by, num_train_sample)
            loss.backward()
            optimizer.step()
            scheduler.step()
            ex.log_scalar('nll.train', loss.item(), i)
            if i % logging_freq == 0:
                logger.info("Epoch %d: loss: %.4f",
                            i, loss.item())
            if i % validate_freq == 0:
                if validation:
                    with torch.no_grad():
                        nll = test_nll(model, valid_loader)
                    if best_nll >= nll:
                        best_nll = nll
                        torch.save(model.state_dict(), checkpoint_dir)
                        logger.info('Save checkpoint')
                    ex.log_scalar('nll.valid', nll, i)
                    logger.info("Epoch %d: validation NLL %.4f", i, nll)
                else:
                    torch.save(model.state_dict(), checkpoint_dir)
                    logger.info('Save checkpoint')
                with torch.no_grad():
                    nll = test_nll(model, test_loader)
                ex.log_scalar('nll.test', nll, i)
                logger.info("Epoch %d: test NLL %.4f", i, nll)
                model.train()
        model.load_state_dict(torch.load(checkpoint_dir, map_location=device))
        tnll = 0
        acc = 0
        nll_miss = 0
        model.eval()
        ll_func = model.negative_loglikelihood
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device)
                by = by.to(device)
                prob = model(bx, num_test_sample)
                tnll += ll_func(bx, by, num_test_sample).item() * len(by)
                vote = prob.argmax(2)
                onehot = torch.zeros(
                    (vote.size(0), vote.size(1), 10), device=vote.device)
                onehot.scatter_(2, vote.unsqueeze(2), 1)
                vote = onehot.sum(dim=1)
                vote /= vote.sum(dim=1, keepdims=True)
                pred = vote.argmax(dim=1)

                y_miss = pred != by
                if y_miss.sum().item() > 0:
                    by_miss = by[y_miss]
                    bx_miss = bx[y_miss]
                    nll_miss += ll_func(bx_miss, by_miss,
                                        num_test_sample).item() * len(by_miss)
                acc += (pred == by).sum().item()
        nll_miss /= len(test_loader.dataset) - acc
        tnll /= len(test_loader.dataset)
        acc /= len(test_loader.dataset)
        logger.info("Test data: acc %.4f, nll %.4f, nll miss %.4f",
                    acc, tnll, nll_miss)
    else:
        model.train()
        best_nll = float('inf')
        for i, (bx, by) in enumerate(train_loader, 1):
            if i > num_iterations:
                break
            optimizer.zero_grad()
            bx = bx.to(device)
            by = by.to(device)
            pred = model(bx)
            loss = torch.nn.functional.nll_loss(pred, by)
            loss.backward()
            optimizer.step()
            scheduler.step()
            ex.log_scalar("nll.train", loss.item(), i)
            if i % logging_freq == 0:
                logger.info("Epoch %d: train %.4f, lr %.4f", i, loss.item(), optimizer.param_groups[0]['lr'])
            if i % validate_freq == 0:
                model.eval()
                if validation:
                    with torch.no_grad():
                        nll = 0
                        for bx, by in valid_loader:
                            bx = bx.to(device)
                            by = by.to(device)
                            pred = model(bx)
                            nll += torch.nn.functional.nll_loss(
                                pred, by).item() * len(by)
                        nll /= len(valid_loader.dataset)
                    if best_nll >= nll:
                        best_nll = nll
                        torch.save(model.state_dict(), checkpoint_dir)
                        logger.info('Save checkpoint')
                    ex.log_scalar('nll.valid', nll, i)
                    logger.info("Epoch %d: validation %.4f", i, nll)
                else:
                    torch.save(model.state_dict(), checkpoint_dir)
                    logger.info('Save checkpoint')
                with torch.no_grad():
                    nll = 0
                    for bx, by in test_loader:
                        bx = bx.to(device)
                        by = by.to(device)
                        pred = model(bx)
                        nll += torch.nn.functional.nll_loss(
                            pred, by).item() * len(by)
                    nll /= len(test_loader.dataset)
                ex.log_scalar('nll.test', nll, i)
                logger.info("Epoch %d: test %.4f", i, nll)
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
