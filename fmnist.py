import logging
import os

import numpy as np
import torch
import torchvision
from sacred import Experiment
from sacred.observers import FileStorageObserver

from datasets import get_data_loader, infinite_wrapper
from models import DeterministicLeNet, StochasticLeNet

EXPERIMENT = 'fmnist'
BASE_DIR = os.path.join('runs', EXPERIMENT)
ex = Experiment(EXPERIMENT)
ex.observers.append(FileStorageObserver(BASE_DIR))


@ex.config
def my_config():
    seed = 1
    p = 0.75
    model_type = 'stochastic'
    kl_weight = 5
    batch_size = 128
    conv_hiddens = [32, 64]
    fc_hidden = 256
    init_mean = 0.0
    init_log_std = -2.3
    weight_decay = 0.0
    lr = 1e-3
    mll_iteration = 12000
    vb_iteration = 14000
    noise_type = 'full'
    noise_size = [28, 28]
    orthogonal_init = False
    activation = 'relu'
    validation = True
    validation_fraction = 0.2
    validate_freq = 1000  # calculate validation frequency
    num_train_sample = 20
    num_test_sample = 200
    logging_freq = 500
    device = 'cuda'
    if not torch.cuda.is_available():
        device = 'cpu'

@ex.capture
def get_model(model_type, conv_hiddens, fc_hidden, orthogonal_init, activation, init_mean, init_log_std, p, noise_type, noise_size, lr, weight_decay, device):
    if model_type == 'stochastic':
        model = StochasticLeNet(28, 28, 1, conv_hiddens, fc_hidden, 10, orthogonal_init,
                                activation, init_mean, init_log_std, p, noise_type, noise_size)
    else:
        model = DeterministicLeNet(
            28, 28, 1, conv_hiddens, fc_hidden, 10, orthogonal_init, activation)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer


@ex.capture
def get_dataloader(batch_size, validation, validation_fraction, seed):
    return get_data_loader('fmnist', batch_size, validation, validation_fraction, seed)


@ex.capture
def get_logger(_run, _log):
    fh = logging.FileHandler(os.path.join(BASE_DIR, _run._id, 'train.log'))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    _log.addHandler(fh)
    return _log


@ex.automain
def main(_run, model_type, num_train_sample, num_test_sample, device, validate_freq, mll_iteration, vb_iteration, logging_freq, kl_weight):
    logger = get_logger()
    train_loader, valid_loader, test_loader = get_dataloader()
    logger.info(
        f"Train size: {len(train_loader.dataset)}, validation size: {len(valid_loader.dataset)}, test size: {len(test_loader.dataset)}")
    train_loader = infinite_wrapper(train_loader)
    model, optimizer = get_model()
    logger.info(str(model))
    checkpoint_dir = os.path.join(BASE_DIR, _run._id, 'checkpoint.pt')

    # First train mll
    if model_type == 'stochastic':
        best_mll = float('inf')
        model.train()
        train_iter = enumerate(train_loader, 1)
        for i, (bx, by) in train_iter:
            if i > mll_iteration:
                break
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            mll = model.marginal_loglikelihood_loss(bx, by, num_train_sample)
            mll.backward()
            optimizer.step()
            ex.log_scalar('mll.train', mll.item(), i)
            if i % logging_freq == 0:
                logger.info("MLL Epoch %d: train %.4f", i, mll)
            if i % validate_freq == 0:
                model.eval()
                with torch.no_grad():
                    mll = 0
                    for bx, by in valid_loader:
                        bx = bx.to(device)
                        by = by.to(device)
                        mll += model.marginal_loglikelihood_loss(
                            bx, by, num_test_sample).item() * len(by)
                    mll /= len(valid_loader.dataset)
                if best_mll >= mll:
                    best_mll = mll
                    torch.save(model.state_dict(), checkpoint_dir)
                    logger.info('Save checkpoint')
                ex.log_scalar('mll.valid', mll, i)
                logger.info("MLL Epoch %d: validation %.4f", i, mll)
                model.train()
        model.load_state_dict(torch.load(checkpoint_dir, map_location=device))
    # Second train using VB
        vb_iteration += mll_iteration
        best_nll = float('inf')
        for i, (bx, by) in train_iter:
            if i > vb_iteration:
                break
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            loglike, kl, wn = model.vb_loss(bx, by, num_train_sample)
            loss = loglike + kl_weight*kl - wn
            loss.backward()
            optimizer.step()
            ex.log_scalar('loglike.train', loglike.item(), i)
            ex.log_scalar('kl.train', kl.item(), i)
            ex.log_scalar('weight_norm', wn.item(), i)
            if i % logging_freq == 0:
                logger.info("VB Epoch %d: loglike: %.4f, kl: %.4f, wn: %.2f", i, loglike.item(), kl.item(), wn.item())
            if i % validate_freq == 0:
                model.eval()
                with torch.no_grad():
                    nll = 0
                    for bx, by in valid_loader:
                        bx = bx.to(device)
                        by = by.to(device)
                        nll += model.negative_loglikelihood(
                            bx, by, num_test_sample).item() * len(by)
                    nll /= len(valid_loader.dataset)
                if best_nll >= nll:
                    best_nll = nll
                    torch.save(model.state_dict(), checkpoint_dir)
                    logger.info('Save checkpoint')
                ex.log_scalar('nll.valid', nll, i)
                logger.info("VB Epoch %d: validation NLL %.4f", i, nll)
                model.train()
        model.load_state_dict(torch.load(checkpoint_dir, map_location=device))
        test_nll = 0
        acc = 0
        nll_miss = 0
        model.eval()
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device)
                by = by.to(device)
                prob = model(bx, num_test_sample)
                test_nll += model.negative_loglikelihood(bx, by, num_test_sample).item() * len(by)
                vote = prob.argmax(2)
                onehot = torch.zeros((vote.size(0), vote.size(1), 10), device=vote.device)
                onehot.scatter_(2, vote.unsqueeze(2), 1)
                vote = onehot.sum(dim=1)
                vote /= vote.sum(dim=1, keepdims=True)
                pred = vote.argmax(dim=1)
                
                y_miss = pred != by
                if y_miss.sum().item() > 0:
                    by_miss = by[y_miss]
                    bx_miss = bx[y_miss]
                    nll_miss += model.negative_loglikelihood(bx_miss, by_miss, num_test_sample).item() * len(by_miss)
                acc += (pred == by).sum().item()
        nll_miss /= len(test_loader.dataset) - acc
        test_nll /= len(test_loader.dataset)
        acc /= len(test_loader.dataset)
        logger.info("Test data: acc %.4f, nll %.4f, nll miss %.4f", acc, test_nll, nll_miss)
    else:
        model.train()
        best_nll = float('inf')
        for i, (bx, by) in enumerate(train_loader, 1):
            if i > vb_iteration:
                break
            optimizer.zero_grad()
            bx = bx.to(device)
            by = by.to(device)
            pred = model(bx)
            loss = torch.nn.functional.nll_loss(pred, by)
            loss.backward()
            optimizer.step()
            ex.log_scalar("nll.train", loss.item(), i)
            if i % logging_freq == 0:
                logger.info("Epoch %d: train %.4f", i, loss.item())
            if i % validate_freq == 0:
                model.eval()
                with torch.no_grad():
                    nll = 0
                    for bx, by in valid_loader:
                        bx = bx.to(device)
                        by = by.to(device)
                        pred = model(bx)
                        nll += torch.nn.functional.nll_loss(pred, by).item() * len(by)
                    nll /= len(valid_loader.dataset)
                if best_nll >= nll:
                    best_nll = nll
                    torch.save(model.state_dict(), checkpoint_dir)
                    logger.info('Save checkpoint')
                ex.log_scalar('nll.valid', nll, i)
                logger.info("Epoch %d: validation %.4f", i, nll)
                model.train()
        model.load_state_dict(torch.load(checkpoint_dir, map_location=device))
        test_nll = 0
        acc = 0
        nll_miss = 0
        model.eval()
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device)
                by = by.to(device)
                prob = model(bx)
                pred = prob.argmax(dim=1)
                test_nll += torch.nn.functional.nll_loss(prob, by).item() * len(by)
                y_miss = pred != by
                if y_miss.sum().item() > 0:
                    prob_miss = prob[y_miss]
                    by_miss = by[y_miss]
                    nll_miss += torch.nn.functional.nll_loss(prob_miss, by_miss).item() * len(by_miss)
                acc += (pred == by).sum().item()
        nll_miss /= len(test_loader.dataset) - acc
        test_nll /= len(test_loader.dataset)
        acc /= len(test_loader.dataset)
        logger.info("Test data: acc %.4f, nll %.4f, nll miss %.4f", acc, test_nll, nll_miss)