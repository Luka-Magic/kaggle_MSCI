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
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
from sklearn.model_selection import train_test_split, KFold, GroupKFold, StratifiedGroupKFold
import sklearn.preprocessing
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA, TruncatedSVD

from utils.utils import seed_everything, load_csr_data_to_gpu, make_coo_batch, make_coo_batch_slice, AverageMeter, EarlyStopping
from model import MsciModel

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
def load_data(cfg, data_dir, compressed_data_dir):
    data_dict = {}
    # 訓練データの入力の読み込み
    compressed_input_path = compressed_data_dir / f'train_multi_input_tsvd{cfg.latent_input_dim}_seed{cfg.seed}.pkl'
    compressed_input_model_path = compressed_data_dir / f'train_multi_input_tsvd{cfg.latent_input_dim}_seed{cfg.seed}_model.pkl'
    if cfg.pca_input:
        ## 入力データをpcaする場合
        ##   PCAモデル・圧縮データ共に既に存在する場合は圧縮データをロード
        ##   そうでない場合は元の入力データをロードし次元削減を行い、PCAモデルと圧縮データを共に保存 (modelはinferで使用するため)
        ##   data_dictに圧縮データだけ加える
        if compressed_input_path.exists() and compressed_input_model_path.exists():
            print('PCA input data already exists, now loading...')
            with open(compressed_input_path, 'rb') as f:
                train_input_compressed = pickle.load(f)
        else:
            train_input = scipy.sparse.load_npz(data_dir / 'train_multi_inputs_values.sparse.npz')
            print('PCA input now...')
            pca_train_input_model = TruncatedSVD(n_components=cfg.latent_input_dim, random_state=cfg.seed)
            train_input_compressed = pca_train_input_model.fit_transform(train_input)
            with open(str(compressed_input_path), 'wb') as f:
                pickle.dump(train_input_compressed, f)
            with open(str(compressed_input_model_path), 'wb') as f:
                pickle.dump(pca_train_input_model, f)
            del pca_train_input_model, train_input
        data_dict['train_input_compressed'] = train_input_compressed
        del train_input_compressed
        print('PCA input complate')
    else:
        ## PCAしない場合
        train_input = scipy.sparse.load_npz(data_dir / 'train_multi_inputs_values.sparse.npz')
        train_input = load_csr_data_to_gpu(train_input)
        gc.collect()
        ## 最大値で割って0-1に正規化
        max_input = torch.from_numpy(np.load(data_dir / 'train_multi_inputs_max_values.npz')['max_input'])[0].to(cfg.device)
        train_input.data[...] /= max_input[train_input.indices.long()]
        data_dict['train_input'] = train_input
        del train_input, max_input
    gc.collect()

    # 訓練データのターゲットの読み込み
    compressed_target_path = compressed_data_dir / f'train_multi_target_tsvd{cfg.latent_target_dim}_seed{cfg.seed}.pkl'
    compressed_target_model_path = compressed_data_dir / f'train_multi_target_tsvd{cfg.latent_target_dim}_seed{cfg.seed}_model.pkl'
    
    train_target = scipy.sparse.load_npz(data_dir / 'train_multi_targets_values.sparse.npz')
    if cfg.pca_target:
        ## ターゲットをpcaする場合
        ##   PCAモデル・圧縮データ共に既に存在する場合は圧縮データをロード
        ##   そうでない場合は元のターゲットデータをロードし次元削減を行い、PCAモデルと圧縮データを共に保存
        ##   data_dictに圧縮データ "と元データ" を加える。
        if compressed_target_path.exists() and compressed_target_model_path.exists():
            print('PCA target data already exists, now loading...')
            with open(compressed_target_path, 'rb') as f:
                train_target_compressed = pickle.load(f)
        else:
            print('PCA target now...')
            pca_train_target_model = TruncatedSVD(n_components=cfg.latent_target_dim, random_state=cfg.seed)
            train_target_compressed = pca_train_target_model.fit_transform(train_target)
            with open(str(compressed_target_path), 'wb') as f:
                pickle.dump(train_target_compressed, f)
            with open(str(compressed_target_model_path), 'wb') as f:
                pickle.dump(pca_train_target_model, f)
            del pca_train_target_model
        data_dict['train_target_compressed'] = train_target_compressed
        del train_target_compressed
        print('PCA target complate')
    train_target = load_csr_data_to_gpu(train_target)
    data_dict['train_target'] = train_target
    del train_target
    gc.collect()
    return data_dict


def create_fold(cfg, data_dir, n_samples):
    if cfg.fold == 'GroupKFold':
        train_idx = np.load(data_dir / 'train_multi_inputs_idxcol.npz', allow_pickle=True)['index']
        train_meta = pd.read_parquet(data_dir / 'metadata.parquet')
        train_meta = train_meta.query('cell_id in @train_idx').reset_index(drop=True)
        kfold = GroupKFold(n_splits=cfg.n_folds)
        fold_list = list(kfold.split(X=range(n_samples), groups=train_meta[cfg.group].values))
    else:
        kfold = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
        fold_list = list(kfold.split(X=range(n_samples)))
    return fold_list


## DataLoader
class DataLoader:
    def __init__(self, cfg, data_dict, train_idx=None, 
                 *,
                batch_size=512, shuffle=False, drop_last=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pca_target = cfg.pca_target
        self.pca_input = cfg.pca_input
        
        if cfg.pca_input:
            self.train_inputs = data_dict['train_input_compressed']
        else:
            self.train_inputs = data_dict['train_input']
        
        if cfg.pca_target:
            self.train_target_compressed = data_dict['train_target_compressed']
        self.train_target = data_dict['train_target']

        self.train_idx = train_idx
        
        self.nb_examples = len(self.train_idx) if self.train_idx is not None else self.train_inputs.shape[0]
        
        self.nb_batches = self.nb_examples//batch_size
        if not drop_last and not self.nb_examples%batch_size==0:
            self.nb_batches +=1
    
    def __iter__(self):
        if self.shuffle:
            shuffled_idx = torch.randperm(self.nb_examples)
            if self.train_idx is not None:
                idx_array = self.train_idx[shuffled_idx]
            else:
                idx_array = shuffled_idx
        else:
            if self.train_idx is not None:
                idx_array = self.train_idx
            else:
                idx_array = None
        
        batch_dict = {}
        for i in range(self.nb_batches):
            slc = slice(i*self.batch_size, (i+1)*self.batch_size)
            if idx_array is None:
                if not self.pca_input:
                    batch_dict['input'] = make_coo_batch_slice(self.train_inputs, i*self.batch_size, (i+1)*self.batch_size)
                else:
                    batch_dict['input_compressed'] = torch.from_numpy(self.train_inputs[i*self.batch_size:(i+1)*self.batch_size, :])
                if self.train_target is not None:
                    batch_dict['target'] = make_coo_batch_slice(self.train_target, i*self.batch_size, (i+1)*self.batch_size)
                    if self.pca_target:
                        batch_dict['target_compressed'] = torch.from_numpy(self.train_target_compressed[i*self.batch_size:(i+1)*self.batch_size, :])
                else:
                    batch_dict['target'] = None
                    if self.pca_target:
                        batch_dict['target_compressed'] = None
            else:
                idx_batch = idx_array[slc]
                if not self.pca_input:
                    batch_dict['input'] = make_coo_batch(self.train_inputs, idx_batch)
                else:
                    batch_dict['input_compressed'] = torch.from_numpy(self.train_inputs[idx_batch, :])
                if self.train_target is not None:
                    batch_dict['target'] = make_coo_batch(self.train_target, idx_batch)
                    if self.pca_target:
                        batch_dict['target_compressed'] = torch.from_numpy(self.train_target_compressed[idx_batch, :])
                else:
                    batch_dict['target'] = None
                    if self.pca_target:
                        batch_dict['target_compressed'] = None
            yield batch_dict
            
    def __len__(self):
        return self.nb_batches

## Train Function
def train_one_epoch(cfg, epoch, train_loader, model, loss_fn, optimizer, scheduler):
    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    losses = AverageMeter()

    for step, (batch_dict) in pbar:
        if cfg.pca_input:
            input = batch_dict['input_compressed'].to(cfg.device)
        else:
            input = batch_dict['input'].to(cfg.device)
        if cfg.pca_target:
            target = batch_dict['target_compressed'].to(cfg.device)
        else:
            target = batch_dict['target'].to_dense().to(cfg.device)
        bs = input.shape[0]

        optimizer.zero_grad()

        pred = model(input)

        pred = (pred - torch.mean(pred, dim=1, keepdim=True)) / (torch.std(pred, dim=1, keepdim=True) + 1e-10)

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
def valid_one_epoch(cfg, epoch, valid_loader, model, pca_train_target_model=None):
    model.eval()

    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))

    losses = AverageMeter()
    scores = AverageMeter()

    for step, (batch_dict) in pbar:
        if cfg.pca_input:
            input = batch_dict['input_compressed'].to(cfg.device)
        else:
            input = batch_dict['input'].to(cfg.device)
        if cfg.pca_target:
            # target = batch_dict['target_compressed'].to(cfg.device)
            target = batch_dict['target'].to_dense().detach().cpu()
        else:
            target = batch_dict['target'].to_dense().detach().cpu()
        bs = input.shape[0]

        with torch.no_grad():
            pred = model(input)
        
        if cfg.pca_target:
            pred = pca_train_target_model.inverse_transform(pred.detach().cpu().numpy())
            pred = torch.from_numpy(pred)
        else:
            pred = pred.detach().cpu()
        
        pred = (pred - torch.mean(pred, dim=1, keepdim=True)) / (torch.std(pred, dim=1, keepdim=True) + 1e-10)

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
    compressed_data_dir = Path.cwd().parents[5] / 'data' / 'compressed_data'
    save_dir = Path.cwd().parents[5] / 'output' / 'multi' / exp_name
    save_dir.mkdir(exist_ok=True)

    # データのロードと整形
    data_dict = load_data(cfg, data_dir, compressed_data_dir)
    if cfg.pca_target:
        compressed_target_model_path = compressed_data_dir / f'train_multi_target_tsvd{cfg.latent_input_dim}_seed{cfg.seed}_model.pkl'
        with open(compressed_target_model_path, 'rb') as f:
            pca_train_target_model = pickle.load(f)
    n_samples = data_dict['train_target'].shape[0]
    if not cfg.pca_input:
        input_size = data_dict['train_input'].shape[1]
    else:
        input_size = cfg.latent_input_dim
    if not cfg.pca_target:
        output_size = data_dict['train_target'].shape[1]
    else:
        output_size = cfg.latent_target_dim

    fold_list = create_fold(cfg, data_dir, n_samples)

    # foldごとに学習
    for fold in range(cfg.n_folds):
        if fold not in cfg.use_fold:
            continue
        
        seed_everything(cfg.seed)

        if cfg.wandb:
            wandb.config = OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True)
            wandb.config['fold'] = fold
            wandb.config['exp_name'] = exp_name
            wandb.init(project=cfg.wandb_project, entity='luka-magic', name=f'{exp_name}_fold{fold}', config=wandb.config)
            
        save_model_path = save_dir / f'{exp_name}_fold{fold}.pth'

        best_fold_score = {'correlation': -1.}

        train_indices, valid_indices = fold_list[fold]

        train_loader = DataLoader(cfg, data_dict, train_idx=train_indices, batch_size=cfg.train_bs, shuffle=True, drop_last=True)
        valid_loader = DataLoader(cfg, data_dict, train_idx=valid_indices, batch_size=cfg.valid_bs, shuffle=True, drop_last=False)

        earlystopping = EarlyStopping(cfg, save_model_path)

        model = MsciModel(input_size, output_size).to(cfg.device)

        if cfg.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        
        if cfg.loss == 'correlation':
            loss_fn = correlation_loss
        
        if cfg.scheduler == None:
            scheduler = None
        else:
            scheduler = None

        # 学習開始
        for epoch in range(cfg.n_epochs):
            train_result = train_one_epoch(cfg, epoch, train_loader, model, loss_fn, optimizer, scheduler)
            valid_result = valid_one_epoch(cfg, epoch, valid_loader, model, pca_train_target_model)

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
            
            earlystopping(valid_result['correlation'], model)
            if earlystopping.early_stop:
                print(f'Early Stop: epoch{epoch}')
                break

        print(f"BEST CORRELATION: {best_fold_score['correlation']}")
        
        del model, loss_fn, optimizer, scheduler, train_result, valid_result, train_indices, valid_indices, best_fold_score
        wandb.finish()
        gc.collect()
        torch.cuda.empty_cache()
    
    print('ALL FINISHED')

if __name__ == '__main__':
    main()