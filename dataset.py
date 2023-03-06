#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022.05.17

@author: fan.weiquan
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from scipy.io import loadmat
import torch.nn.functional as F
class MMDataset(Dataset):
    def __init__(self, metadata, cfg, **kwargs):
        super().__init__()
        self.metadata = metadata
        self.cfg = cfg
        self.pool_feat_audio = cfg.dataset.pool_feat_audio
        self.max_len_t = cfg.dataset.max_len_t
        self.max_len_a = cfg.dataset.max_len_a

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        # raw_text: string(L1,)
        # raw_audio: array(L2,)
        id = self.metadata.id[index]
        x0 = loadmat(self.cfg.dataset.path_feat_t + id)['data']
        x1 = loadmat(self.cfg.dataset.path_feat_a + id)['data']
        if x1.shape[0]<=8: print(id, x1.shape)
        x1 = F.avg_pool1d(torch.from_numpy(x1).T, self.pool_feat_audio, self.pool_feat_audio).T.cpu().numpy()
        x0, x0_padding_mask = pad_input(x0, self.max_len_t)
        x1, x1_padding_mask = pad_input(x1, self.max_len_a//self.pool_feat_audio)
        label = torch.tensor(self.metadata.label[index], dtype=torch.long)
        return {'input_ids':x0, 'attention_mask':x0_padding_mask}, {'input_values':x1, 'attention_mask':x1_padding_mask}, label, label, label, id

def pad_input(x: np.ndarray, lenght, pad_value=0):
    t = x.shape[0]
    mask = torch.zeros(lenght)
    if lenght > t:
        x = np.pad(x, ((0,lenght-t), (0,0)), 'constant', constant_values=(pad_value, pad_value))
        mask[-(lenght-t):] = 1
    else:
        x = x[:lenght,:]
    x = torch.from_numpy(x)
    mask = mask.eq(1)
    return x, mask


def balance_data(metadata, num_class, copy_to=2, retain_bt=True):
    print('Before Balance: Train Num: ', np.array([sum(metadata['label']==i) for i in range(num_class)]))
    if copy_to != 0:
        num = np.array([sum(metadata['label']==i) for i in range(num_class)])
        num_copy_to = sorted(num)[-copy_to]
        flag_bal = [num_copy_to-num[i]<0 for i in range(num_class)]
        ind_bal = [np.random.choice(num[i], num[i]-num_copy_to, replace=False) if flag_bal[i] else np.random.randint(0, num[i], num_copy_to-num[i]) for i in range(num_class)]
        for i in range(num_class):
            if not flag_bal[i]:
                metadata = metadata.append(metadata[metadata['label']==i].iloc[ind_bal[i], :])
            else:
                if not retain_bt:
                    tmp = metadata[metadata['label']==i].reset_index(drop=True).drop(index=ind_bal[i])
                    metadata = metadata[metadata['label']!=i]
                    metadata = metadata.append(tmp)
    print('After Balance: Train Num: ', np.array([sum(metadata['label']==i) for i in range(num_class)]))
    return metadata.reset_index(drop=True)


def build_iemocap_dataset(cfg, sess=1, **kwargs):
    metadata = pd.read_csv(cfg.dataset.path)
    print('Before Balance: Num: ', np.array([sum(metadata['label']==i) for i in range(cfg.model.num_classes)]))
    train_metadata = metadata[metadata['session'] != sess].reset_index(drop=True)
    # dev_metadata = train_metadata.sample(frac=0.1, random_state=0, axis=0).reset_index(drop=True)
    # train_metadata = train_metadata[~train_metadata.id.isin(dev_metadata.id)].reset_index(drop=True)
    test_metadata = metadata[metadata['session'] == sess].reset_index(drop=True)
    train_metadata = balance_data(train_metadata, cfg.model.num_classes, copy_to=1, retain_bt=False)

    train_dataset = MMDataset(train_metadata, cfg)
    # dev_dataset = MMDataset(dev_metadata, cfg)
    test_dataset = MMDataset(test_metadata, cfg)
    return train_dataset, test_dataset, test_dataset


def build_meld_dataset(cfg, sess=None, **kwargs):
    metadata = pd.read_csv(cfg.dataset.path)
    metadata = metadata[metadata['label'] < cfg.model.num_classes].reset_index(drop=True)
    print('Before Balance: Num: ', np.array([sum(metadata['label']==i) for i in range(cfg.model.num_classes)]))
    train_metadata = metadata[metadata['mode'] == 'train'].reset_index(drop=True)
    dev_metadata = metadata[metadata['mode'] == 'dev'].reset_index(drop=True)
    test_metadata = metadata[metadata['mode'] == 'test'].reset_index(drop=True)
    # train_metadata = balance_data(train_metadata, cfg.model.num_classes, copy_to=1, retain_bt=False)

    train_dataset = MMDataset(train_metadata, cfg)
    dev_dataset = MMDataset(dev_metadata, cfg)
    test_dataset = MMDataset(test_metadata, cfg)
    return train_dataset, dev_dataset, test_dataset

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf
    import pandas as pd

    cfg = OmegaConf.load('configs/mm-pretraining.yaml')
    train_dataset, _, test_dataset = build_iemocap_dataset(cfg)
    dataloader = DataLoader(train_dataset, batch_size=16)
    data_iter = iter(dataloader)
    batch = next(data_iter)
    print(batch)
