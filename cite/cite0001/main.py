from random import shuffle
import wandb

import copy
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc

import math
import collections
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
from sklearn.model_selection import train_test_split, KFold, GroupKFold, StratifiedGroupKFold
import sklearn.preprocessing
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.utils import load_csr_data_to_gpu, make_coo_batch, make_coo_batch_slice, AverageMeter


## Loss
def partial_correlation_score_torch_faster(y_true, y_pred):
    y_true_centered = y_true - torch.mean(y_true, dim=1)[:,None]
    y_pred_centered = y_pred - torch.mean(y_pred, dim=1)[:,None]
    cov_tp = torch.sum(y_true_centered*y_pred_centered, dim=1)/(y_true.shape[1]-1)
    var_t = torch.sum(y_true_centered**2, dim=1)/(y_true.shape[1]-1)
    var_p = torch.sum(y_pred_centered**2, dim=1)/(y_true.shape[1]-1)
    return cov_tp/torch.sqrt(var_t*var_p)

def correlation_loss(pred, tgt):
    return -torch.mean(partial_correlation_score_torch_faster(tgt, pred))


## Dataset
def load_data(data_dir, device):
    # 訓練データの入力の読み込み
    train_input = scipy.sparse.load_npz(data_dir / 'train_cite_inputs_values.sparse.npz')
    train_input = load_csr_data_to_gpu(train_input)
    gc.collect()
    ## 最大値で割って0-1に正規化
    max_input = torch.from_numpy(np.load(data_dir / 'train_cite_inputs_max_values.npz')['max_input'])[0].to(device)
    train_input.data[...] /= max_input[train_input.indices.long()]

    # 訓練データのターゲットの読み込み
    train_target = scipy.sparse.load_npz(data_dir / 'train_cite_targets_values.sparse.npz')
    train_target = load_csr_data_to_gpu(train_target)
    gc.collect()

    return train_input, train_target

def create_fold(cfg, data_dir, n_samples):
    if cfg.fold == 'GroupKFold':
        train_idx = np.load(data_dir / 'train_cite_inputs_idxcol.npz', allow_pickle=True)['index']
        train_meta = pd.read_parquet(data_dir / 'metadata.parquet')
        train_meta = train_meta.query('cell_id in @train_idx').reset_index(drop=True)
        kfold = GroupKFold(n_splits=cfg.n_folds)
        fold_list = list(kfold.split(X=range(n_samples), groups=train_meta[cfg.group].values))
    else:
        kfold = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
        fold_list = list(kfold.split(X=range(n_samples)))
    return fold_list


## DataLoader
class DataLoaderCOO:
    def __init__(self, train_inputs, train_target, train_idx=None, 
                 *,
                batch_size=512, shuffle=False, drop_last=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.train_inputs = train_inputs
        self.train_target = train_target
        
        self.train_idx = train_idx
        
        self.nb_examples = len(self.train_idx) if self.train_idx is not None else train_inputs.shape[0]
        
        self.nb_batches = self.nb_examples//batch_size
        if not drop_last and not self.nb_examples%batch_size==0:
            self.nb_batches +=1
    
    def __iter__(self):
        if self.shuffle:
            shuffled_idx = torch.randperm(self.nb_examples)
            # shuffled_idx = torch.randperm(self.nb_examples, device=cfg.device)
            if self.train_idx is not None:
                idx_array = self.train_idx[shuffled_idx]
            else:
                idx_array = shuffled_idx
        else:
            if self.train_idx is not None:
                idx_array = self.train_idx
            else:
                idx_array = None
        
        for i in range(self.nb_batches):
            slc = slice(i*self.batch_size, (i+1)*self.batch_size)
            if idx_array is None:
                inp_batch = make_coo_batch_slice(self.train_inputs, i*self.batch_size, (i+1)*self.batch_size)
                if self.train_target is not None:
                    tgt_batch = make_coo_batch_slice(self.train_target, i*self.batch_size, (i+1)*self.batch_size)
                else:
                    tgt_batch = None
            else:
                idx_batch = idx_array[slc]
                inp_batch = make_coo_batch(self.train_inputs, idx_batch)
                if self.train_target is not None:
                    tgt_batch = make_coo_batch(self.train_target, idx_batch)
                else:
                    tgt_batch = None
            yield inp_batch, tgt_batch
            
    def __len__(self):
        return self.nb_batches


## Model
class MsciModel(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_channel, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_channel),
            # nn.ReLU(),
            nn.Softplus()
        )
    
    def forward(self, x):
        return self.mlp(x)

## Train Function
def train_one_epoch(cfg, epoch, train_loader, model, loss_fn, optimizer, scheduler):
    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    losses = AverageMeter()

    for step, (input, target) in pbar:
        bs = input.shape[0]
        input = input.to(cfg.device)
        target = target.to_dense().to(cfg.device)

        optimizer.zero_grad()

        pred = model(input)

        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), bs)

        if scheduler:
            scheduler.step()

        description = f'TRAIN epoch: {epoch}, loss: {loss.item():.4f}'
        pbar.set_description(description)
    return {'loss': losses.avg}


## Valid Function
def valid_one_epoch(cfg, epoch, valid_loader, model, loss_fn):
    model.eval()

    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))

    losses = AverageMeter()
    scores = AverageMeter()

    partial_correlation_scores = []

    for step, (input, target) in pbar:
        bs = input.shape[0]
        input = input.to(cfg.device)
        target = target.to_dense().to(cfg.device)

        with torch.no_grad():
            pred = model(input)

        pred = (pred - torch.mean(pred, dim=1, keepdim=True)) / (torch.std(pred, dim=1, keepdim=True) + 1e-7)

        batch_score = partial_correlation_score_torch_faster(target, pred)
        for score in batch_score:
            scores.update(score.item())

        loss = -torch.mean(batch_score) # =loss_fnの出力

        losses.update(loss.item(), bs)

        description = f'VALID epoch: {epoch}, loss: {loss.item():.4f}'
        pbar.set_description(description)
    
    return {'loss': losses.avg, 'correlation': scores.avg}


## main
@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig):
    # 初期設定
    if cfg.wandb:
        wandb.login()
    
    exp_name = Path.cwd().parents[2].name
    data_dir = Path.cwd().parents[5] / 'data' / 'data'
    save_dir = Path.cwd().parents[5] / 'output' / 'cite' / exp_name
    save_dir.mkdir(exist_ok=True)

    # データのロードと整形
    train_input, train_target = load_data(data_dir, cfg.device)
    n_samples = train_input.shape[0]
    input_size = train_input.shape[1]
    output_size = train_target.shape[1]
    fold_list = create_fold(cfg, data_dir, n_samples)

    # foldごとに学習
    for fold in range(cfg.n_folds):
        if fold not in cfg.use_fold:
            continue
        
        if cfg.wandb:
            wandb.config = OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True)
            wandb.config['fold'] = fold
            wandb.cnfig['exp_num'] = exp_name
            wandb.init(project=cfg.wandb_project, entity='luka-magic', name=f'{exp_name}_fold{fold}', config=wandb.config)

        best_fold_score = {'correlation': -1.}

        train_indices, valid_indices = fold_list[fold]

        train_loader = DataLoaderCOO(train_input, train_target, train_idx=train_indices, batch_size=cfg.train_bs, shuffle=True, drop_last=True)
        valid_loader = DataLoaderCOO(train_input, train_target, train_idx=valid_indices, batch_size=cfg.valid_bs, shuffle=True, drop_last=False)

        model = MsciModel(input_size, output_size).to(cfg.device)

        if cfg.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters())
        
        if cfg.loss == 'correlation':
            loss_fn = correlation_loss
        
        if cfg.scheduler == None:
            scheduler = None
        else:
            scheduler = None

        # 学習開始
        for epoch in range(cfg.n_epochs):
            train_result = train_one_epoch(cfg, epoch, train_loader, model, loss_fn, optimizer, scheduler)
            valid_result = valid_one_epoch(cfg, epoch, valid_loader, model, loss_fn)

            print('='*40)
            print(f"TRAIN {epoch}, loss: {train_result['loss']}")
            print(f"VALID {epoch}, loss: {valid_result['loss']}, score: {valid_result['correlation']}")
            print('='*40)

            if cfg.wandb:
                wandb.log(dict(
                    epoch = epoch,
                    train_loss = train_result['loss'],
                    valid_loss = valid_result['loss'],
                    correlation = valid_result['correlation']
                ))
            
            if best_fold_score['correlation'] < valid_result['correlation']:
                best_fold_score['correlation'] = valid_result['correlation']
                if cfg.wandb:
                    wandb.run.summary['best_correlation'] = best_fold_score['correlation']
                torch.save(model.state_dict(), save_dir / f'{exp_name}_fold{fold}.pth')
            
        print(f"BEST CORRELATION: {best_fold_score['correlation']}")
        
        del model, loss_fn, optimizer, scheduler, train_result, valid_result, train_indices, valid_indices, best_fold_score
        wandb.finish()
        gc.collect()
        torch.cuda.empty_cache()
    
    print('ALL FINISHED')

if __name__ == '__main__':
    main()