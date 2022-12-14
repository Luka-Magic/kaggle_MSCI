import os
import warnings
import collections
import torch
import numpy as np
import pandas as pd
import wandb
import pickle
import scipy
import gc
from sklearn.decomposition import PCA, TruncatedSVD
import math

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    warnings.simplefilter('ignore')

### Data
TorchCSR = collections.namedtuple("TrochCSR", "data indices indptr shape")

def load_csr_data_to_gpu(train_inputs, device='cuda:0'):
    th_data = torch.from_numpy(train_inputs.data).to(device)
    th_indices = torch.from_numpy(train_inputs.indices).to(device)
    th_indptr = torch.from_numpy(train_inputs.indptr).to(device)
    th_shape = train_inputs.shape
    return TorchCSR(th_data, th_indices, th_indptr, th_shape)

def make_coo_batch(torch_csr, indx, device='cuda:0'):
    th_data, th_indices, th_indptr, th_shape = torch_csr
    start_pts = th_indptr[indx]
    end_pts = th_indptr[indx+1]
    coo_data = torch.cat([th_data[start_pts[i]: end_pts[i]] for i in range(len(start_pts))], dim=0)
    coo_col = torch.cat([th_indices[start_pts[i]: end_pts[i]] for i in range(len(start_pts))], dim=0)
    coo_row = torch.repeat_interleave(torch.arange(indx.shape[0], device=device), th_indptr[indx+1] - th_indptr[indx])
    coo_batch = torch.sparse_coo_tensor(torch.vstack([coo_row, coo_col]), coo_data, [indx.shape[0], th_shape[1]])
    return coo_batch

def make_coo_batch_slice(torch_csr, start, end, device='cuda:0'):
    th_data, th_indices, th_indptr, th_shape = torch_csr
    if end > th_shape[0]:
        end = th_shape[0]
    start_pts = th_indptr[start]
    end_pts = th_indptr[end]
    coo_data = th_data[start_pts: end_pts]
    coo_col = th_indices[start_pts: end_pts]
    coo_row = torch.repeat_interleave(torch.arange(end-start, device=device), th_indptr[start+1:end+1] - th_indptr[start:end])
    coo_batch = torch.sparse_coo_tensor(torch.vstack([coo_row, coo_col]), coo_data, [end-start, th_shape[1]])
    return coo_batch

## Dataset
def load_data(cfg, data_dir, compressed_data_dir):
    data_dict = {}
    # ???????????????????????????????????????
    if cfg.pca_input is not None:
        compressed_input_train_path = compressed_data_dir / cfg.phase / f'train_{cfg.phase}_input_{cfg.pca_input}{cfg.latent_input_dim}.pkl'
        compressed_input_test_path = compressed_data_dir / cfg.phase / f'test_{cfg.phase}_input_{cfg.pca_input}{cfg.latent_input_dim}.pkl'
        ## ??????????????????pca??????????????
        ##   PCA???????????????????????????????????????????????????????????????????????????????????????
        ##   ????????????????????????????????????????????????????????????????????????????????????PCA?????????????????????????????????????????? (model???infer?????????????????????)
        ##   data_dict?????????????????????????????????
        if compressed_input_train_path.exists() and compressed_input_test_path.exists():
            print('PCA input data already exists, now loading...')
            with open(compressed_input_train_path, 'rb') as f:
                train_input_compressed = pickle.load(f)
        else:
            if cfg.pca_input == 'tsvd':
                concat_input = scipy.sparse.load_npz(data_dir / f'concat_{cfg.phase}_inputs_values.sparse.npz')
                print('PCA input now...')
                pca_input_model = TruncatedSVD(n_components=cfg.latent_input_dim, random_state=cfg.seed)
                concat_input_compressed = pca_input_model.fit_transform(concat_input)
                del concat_input, pca_input_model
                gc.collect()
                train_size = cfg.train_multi_size if cfg.phase == 'multi' else cfg.train_cite_size
                train_input_compressed = concat_input_compressed[:train_size]
                test_input_compressed = concat_input_compressed[train_size:]
                with open(str(compressed_input_train_path), 'wb') as f:
                    pickle.dump(train_input_compressed, f)
                with open(str(compressed_input_test_path), 'wb') as f:
                    pickle.dump(test_input_compressed, f)
                del test_input_compressed
            elif cfg.pca_input == 'umap':
                TODO
        if cfg.eda_input is not None:
            print('eda data concatenate...')
            eda_df = pd.read_csv(data_dir / 'train_eda.csv')
            eda_columns = [column for column in cfg.eda_input]
            eda_arr = eda_df.loc[:, eda_columns].values
            eda_arr = (eda_arr - eda_arr.mean(axis=1, keepdims=True)) / eda_arr.std(axis=1, keepdims=True)
            train_input_compressed = np.concatenate([train_input_compressed, eda_arr], axis=1)
        # row-wise z-score normalization 
        train_input_compressed = (train_input_compressed - np.mean(train_input_compressed, axis=1, keepdims=True)) \
            / np.std(train_input_compressed, axis=1, keepdims=True)
        
        data_dict['input_compressed'] = train_input_compressed
        del train_input_compressed
        print('PCA input complate')
    else:
        ## PCA???????????????
        train_input = scipy.sparse.load_npz(data_dir / f'train_{cfg.phase}_inputs_values.sparse.npz')
        train_input = load_csr_data_to_gpu(train_input)
        gc.collect()
        ## ?????????????????????0-1????????????
        max_input = torch.from_numpy(np.load(data_dir / f'train_{cfg.phase}_inputs_max_values.npz')['max_input'])[0].to(cfg.device)
        train_input.data[...] /= max_input[train_input.indices.long()]
        data_dict['input'] = train_input
        del train_input, max_input
    gc.collect()

    # ????????????????????????????????????????????????
    train_target = scipy.sparse.load_npz(data_dir / f'train_{cfg.phase}_targets_values.sparse.npz')
    if cfg.pca_target is not None:
        compressed_target_train_path = compressed_data_dir / cfg.phase / f'train_{cfg.phase}_target_{cfg.pca_input}{cfg.latent_target_dim}.pkl'
        compressed_target_model_path = compressed_data_dir / cfg.phase / f'train_{cfg.phase}_target_{cfg.pca_input}{cfg.latent_target_dim}_model.pkl'
        ## ??????????????????pca????????????
        ##   PCA???????????????????????????????????????????????????????????????????????????????????????
        ##   ?????????????????????????????????????????????????????????????????????????????????????????????PCA??????????????????????????????????????????
        ##   data_dict?????????????????? "???????????????" ???????????????
        if compressed_target_train_path.exists() and compressed_target_model_path.exists():
            print('PCA target data already exists, now loading...')
            with open(compressed_target_train_path, 'rb') as f:
                train_target_compressed = pickle.load(f)
        else:
            if cfg.pca_target == 'tsvd':
                print('PCA target now...')
                pca_train_target_model = TruncatedSVD(n_components=cfg.latent_target_dim, random_state=cfg.seed)
                train_target_compressed = pca_train_target_model.fit_transform(train_target)
                with open(str(compressed_target_train_path), 'wb') as f:
                    pickle.dump(train_target_compressed, f)
                with open(str(compressed_target_model_path), 'wb') as f:
                    pickle.dump(pca_train_target_model, f)
                del pca_train_target_model
            elif cfg.pca_target == 'umap':
                TODO
        if cfg.eda_input is not None:
            print('eda data concatenate...')
            eda_df = pd.read_csv(data_dir / 'test_eda.csv')
            eda_columns = [column for column in cfg.eda_input]
            eda_arr = eda_df.loc[:, eda_columns].values
            eda_arr = (eda_arr - eda_arr.mean(axis=1, keepdims=True)) / eda_arr.std(axis=1, keepdims=True)
            train_target_compressed = np.concatenate([train_target_compressed, eda_arr], axis=1)
        # row-wise z-score normalization 
        train_target_compressed = (train_target_compressed - np.mean(train_target_compressed, axis=1, keepdims=True)) / np.std(train_target_compressed, axis=1, keepdims=True)
        
        data_dict['target_compressed'] = train_target_compressed
        del train_target_compressed
        print('PCA target complate')
    train_target = load_csr_data_to_gpu(train_target)
    data_dict['target'] = train_target
    del train_target
    gc.collect()
    torch.cuda.empty_cache()
    return data_dict

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def partial_correlation_score_torch_faster(y_true, y_pred):
    y_true_centered = y_true - torch.mean(y_true, dim=1)[:,None]
    y_pred_centered = y_pred - torch.mean(y_pred, dim=1)[:,None]
    cov_tp = torch.sum(y_true_centered*y_pred_centered, dim=1)/(y_true.shape[1]-1)
    var_t = torch.sum(y_true_centered**2, dim=1)/(y_true.shape[1]-1)
    var_p = torch.sum(y_pred_centered**2, dim=1)/(y_true.shape[1]-1)
    return cov_tp/torch.sqrt(var_t*var_p)

def correlation_loss(pred, tgt):
    return -torch.mean(partial_correlation_score_torch_faster(tgt, pred))


class EarlyStopping:
    """earlystopping?????????"""

    def __init__(self, cfg, save_path):
        """??????????????????????????????????????????????????????????????????????????????path"""

        self.patience = cfg.patience    #??????????????????????????????
        if not cfg.earlystopping: # patience???-1?????????earlystopping??????????????????
            self.patience = -1
        self.counter = 0            #????????????????????????
        self.best_score = None      #??????????????????
        self.early_stop = False     #?????????????????????
        self.verbose = True
        self.score_before = -1.0
        self.path = save_path             #????????????????????????path
        self.eps = 1e-4

    def __call__(self, score, model):
        """
        ??????(call)????????????
        ????????????????????????????????????loss????????????????????????????????????????????????
        """
        if math.isnan(score):
            self.early_stop = True
            print('loss is not a number.')
        
        if self.best_score is None:  #1Epoch????????????
            self.best_score = score   #1Epoch?????????????????????????????????????????????????????????
            self.checkpoint(score, model)  #?????????????????????????????????????????????????????????
        elif score < self.best_score + self.eps:  # ???????????????????????????????????????????????????
            self.counter += 1   #???????????????????????????+1
            if self.verbose:  #????????????????????????????????????????????????
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #???????????????????????????????????? 
            if self.counter >= self.patience and self.patience != -1:  #????????????????????????????????????????????????????????????True?????????
                self.early_stop = True
                print(f'BEST SCORE: {self.best_score:.4f}')
        else:  #???????????????????????????????????????
            self.best_score = score  #??????????????????????????????
            self.checkpoint(score, model)  #???????????????????????????????????????
            self.counter = 0  #????????????????????????????????????

    def checkpoint(self, score, model):
        '''???????????????????????????????????????????????????????????????????????????'''
        if self.verbose:  #????????????????????????????????????????????????????????????????????????????????????????????????????????????
            print(f'Validation score increased ({self.score_before:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #?????????????????????????????????path?????????
        self.score_before = score