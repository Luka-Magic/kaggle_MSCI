import os
import warnings
import collections
import torch
import numpy as np
import wandb
import pickle
import scipy
import gc
from sklearn.decomposition import PCA, TruncatedSVD

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
    # 訓練データの入力の読み込み
    if cfg.pca_input:
        compressed_input_train_path = compressed_data_dir / cfg.phase / f'train_{cfg.phase}_input_tsvd{cfg.latent_input_dim}.pkl'
        compressed_input_test_path = compressed_data_dir / cfg.phase / f'test_{cfg.phase}_input_tsvd{cfg.latent_input_dim}.pkl'
        ## 入力データをpcaする場合 
        ##   PCAモデル・圧縮データ共に既に存在する場合は圧縮データをロード
        ##   そうでない場合は元の入力データをロードし次元削減を行い、PCAモデルと圧縮データを共に保存 (modelはinferで使用するため)
        ##   data_dictに圧縮データだけ加える
        if compressed_input_train_path.exists() and compressed_input_test_path.exists():
            print('PCA input data already exists, now loading...')
            with open(compressed_input_train_path, 'rb') as f:
                train_input_compressed = pickle.load(f)
        else:
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
        data_dict['input_compressed'] = train_input_compressed
        del train_input_compressed
        print('PCA input complate')
    else:
        ## PCAしない場合
        train_input = scipy.sparse.load_npz(data_dir / f'train_{cfg.phase}_inputs_values.sparse.npz')
        train_input = load_csr_data_to_gpu(train_input)
        gc.collect()
        ## 最大値で割って0-1に正規化
        max_input = torch.from_numpy(np.load(data_dir / f'train_{cfg.phase}_inputs_max_values.npz')['max_input'])[0].to(cfg.device)
        train_input.data[...] /= max_input[train_input.indices.long()]
        data_dict['input'] = train_input
        del train_input, max_input
    gc.collect()

    # 訓練データのターゲットの読み込み
    train_target = scipy.sparse.load_npz(data_dir / f'train_{cfg.phase}_targets_values.sparse.npz')
    if cfg.pca_target:
        compressed_target_train_path = compressed_data_dir / cfg.phase / f'train_{cfg.phase}_target_tsvd{cfg.latent_target_dim}.pkl'
        compressed_target_model_path = compressed_data_dir / cfg.phase / f'train_{cfg.phase}_target_tsvd{cfg.latent_target_dim}_model.pkl'
        ## ターゲットをpcaする場合
        ##   PCAモデル・圧縮データ共に既に存在する場合は圧縮データをロード
        ##   そうでない場合は元のターゲットデータをロードし次元削減を行い、PCAモデルと圧縮データを共に保存
        ##   data_dictに圧縮データ "と元データ" を加える。
        if compressed_target_train_path.exists() and compressed_target_model_path.exists():
            print('PCA target data already exists, now loading...')
            with open(compressed_target_train_path, 'rb') as f:
                train_target_compressed = pickle.load(f)
        else:
            print('PCA target now...')
            pca_train_target_model = TruncatedSVD(n_components=cfg.latent_target_dim, random_state=cfg.seed)
            train_target_compressed = pca_train_target_model.fit_transform(train_target)
            with open(str(compressed_target_train_path), 'wb') as f:
                pickle.dump(train_target_compressed, f)
            with open(str(compressed_target_model_path), 'wb') as f:
                pickle.dump(pca_train_target_model, f)
            del pca_train_target_model
        data_dict['target_compressed'] = train_target_compressed
        del train_target_compressed
        print('PCA target complate')
    train_target = load_csr_data_to_gpu(train_target)
    data_dict['target'] = train_target
    del train_target
    gc.collect()
    return data_dict


def load_test_data(cfg, data_dir, compressed_data_dir):
    data_dict = {}
    # テストデータの入力の読み込み
    if cfg.pca_input:
        compressed_input_test_path = compressed_data_dir / cfg.phase / f'test_{cfg.phase}_input_tsvd{cfg.latent_input_dim}.pkl'
        ## 入力データをpcaする場合 
        ##   PCAモデル・圧縮データ共に既に存在する場合は圧縮データをロード
        ##   そうでない場合は元の入力データをロードし次元削減を行い、PCAモデルと圧縮データを共に保存 (modelはinferで使用するため)
        ##   data_dictに圧縮データだけ加える
        assert compressed_input_test_path.exists()
        print('PCA input data already exists, now loading...')
        with open(compressed_input_test_path, 'rb') as f:
            test_input_compressed = pickle.load(f)
        data_dict['test_input_compressed'] = test_input_compressed
        del test_input_compressed
        print('PCA input complate')
    else:
        # 訓練データの入力の読み込み
        test_input = scipy.sparse.load_npz(data_dir / 'test_multi_inputs_values.sparse.npz')
        test_input = load_csr_data_to_gpu(test_input)
        gc.collect()
        ## 最大値で割って0-1に正規化
        max_input = torch.from_numpy(np.load(data_dir / 'train_multi_inputs_max_values.npz')['max_input'])[0].to(cfg.device)
        test_input.data[...] /= max_input[test_input.indices.long()]
        data_dict['test_input'] = test_input
        del test_input, max_input
    gc.collect()
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
    """earlystoppingクラス"""

    def __init__(self, cfg, save_path):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = cfg.patience    #設定ストップカウンタ
        if not cfg.earlystopping: # patienceが-1の時はearlystoppingは発動しない
            self.patience = -1
        self.wandb = cfg.wandb
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.verbose = True
        self.score_before = -1.0
        self.path = save_path             #ベストモデル格納path

    def __call__(self, score, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(score, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience and self.patience != -1:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
                print(f'BEST SCORE: {self.best_score:.4f}')
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            if self.wandb:
                wandb.run.summary['best_correlation'] = score
            self.checkpoint(score, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, score, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.score_before:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.score_before = score