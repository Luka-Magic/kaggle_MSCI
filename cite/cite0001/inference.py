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


## Dataset
def load_data(data_dir, device):
    # 訓練データの入力の読み込み
    test_input = scipy.sparse.load_npz(data_dir / 'test_cite_inputs_values.sparse.npz')
    test_input = load_csr_data_to_gpu(test_input)
    gc.collect()
    ## 最大値で割って0-1に正規化
    max_input = torch.from_numpy(np.load(data_dir / 'train_cite_inputs_max_values.npz')['max_input'])[0].to(device)
    test_input.data[...] /= max_input[test_input.indices.long()]
    del max_input
    gc.collect()

    return test_input


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


## Test Function
def test_function(cfg, model, test_loader, n_samples, output_size):
    model.eval()

    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    
    preds = torch.zeros((n_samples, output_size), device=cfg.device, dtype=torch.float32)
    
    start = 0
    for step, (input, _) in pbar:
        bs = input.shape[0]
        input = input.to(cfg.device)

        with torch.no_grad():
            pred = model(input)

        pred = (pred - torch.mean(pred, dim=1, keepdim=True)) / (torch.std(pred, dim=1, keepdim=True) + 1e-10)

        preds[start:start+bs] = pred
        start += bs
    
    return preds


## main
@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig):
    # 初期設定    
    exp_name = Path.cwd().parents[2].name
    data_dir = Path.cwd().parents[5] / 'data' / 'data'
    save_dir = Path.cwd().parents[5] / 'output' / 'cite' / exp_name
    save_dir.mkdir(exist_ok=True)
    
    # データのロードと整形
    test_input = load_data(data_dir, cfg.device)
    n_samples = test_input.shape[0]
    input_size = test_input.shape[1]
    output_size = cfg.output_size

    test_loader = DataLoaderCOO(train_inputs=test_input, train_target=None, train_idx=None, batch_size=cfg.test_bs, shuffle=False, drop_last=False)

    preds_all = None

    model_num = 0
    for model_path in save_dir.glob('*.pth'):
        model = MsciModel(input_size, output_size)
        model.to(cfg.device)

        model.load_state_dict(torch.load(model_path))
        preds = test_function(cfg, model, test_loader, n_samples, output_size)

        if preds_all is not None:
            preds_all += preds
        else:
            preds_all = preds
        del preds, model
        torch.cuda.empty_cache()
        gc.collect()
        model_num += 1
    preds_all /= float(model_num)

    del test_loader, test_input
    gc.collect()

    test_pred = preds_all.cpu().detach().numpy()
    sub_df = pd.read_parquet(data_dir / 'sample_submission.parquet')
    sub_df['target'] = None
    sub_df.loc[:len(test_pred.ravel())-1, 'target'] = test_pred.ravel()
    sub_df = sub_df.round(6)
    sub_df.to_csv(str(save_dir / 'submission.csv'), index=False)

    del preds_all, test_pred, sub_df
    gc.collect()
    torch.cuda.empty_cache()
    
    print('ALL FINISHED')

if __name__ == '__main__':
    main()