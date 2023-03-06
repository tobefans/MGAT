# -*- coding: utf-8 -*-
"""
Created on 2022.06.06

@author: fan.weiquan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from module import modeling_cross
from multiprocessing import Queue
from module.layers import build_transformer
from module import layers
class MMModel(nn.Module):
    def __init__(self, cfg, pretrained=None, **kwargs):
        super().__init__()
        self.cfg = cfg

        self.text_self_Trans = build_transformer(self_attn=True, num_layers=2, embed_dim=cfg.model.embed_dim, ffn_embed_dim=512, num_heads=8)
        self.audio_self_Trans = build_transformer(self_attn=True, num_layers=2, embed_dim=cfg.model.embed_dim, ffn_embed_dim=512, num_heads=8)
        # self.text_self_Trans = getattr(layers, cfg.model.config_self.model_base)(cfg=cfg.model.config_self.rewrite_param_1, len_q=cfg.dataset.max_len_t, len_k=cfg.dataset.max_len_t)
        # self.audio_self_Trans = getattr(layers, cfg.model.config_self.model_base)(cfg=cfg.model.config_self.rewrite_param_2, len_q=cfg.dataset.max_len_a//cfg.dataset.pool_feat_audio, len_k=cfg.dataset.max_len_a//cfg.dataset.pool_feat_audio)

        self.encoder_cross = getattr(modeling_cross, cfg.model.config_cross.model_name, None)
        if self.encoder_cross != None: self.encoder_cross = self.encoder_cross(cfg)

        fc_dim = cfg.model.embed_dim*2
        self.fc_dim = fc_dim
        self.classifier = nn.Sequential(
            nn.Linear(fc_dim, fc_dim//2),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(fc_dim//2, fc_dim//4),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(fc_dim//4, cfg.model.num_classes),
        )

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.temperature = cfg.model.temperature
        self.num_neg_quene = cfg.model.num_neg_quene
        
        self.quene_1 = torch.tensor([]) # init
        self.quene_2 = torch.tensor([])

        self.__dict__.update(kwargs)
        # print(self)

    def forward(self, token1, token2, mask1, mask2, is_clr=False, **kwargs):
        trg_1,_ = self.text_self_Trans(token1['input_ids'], key_padding_mask=token1['attention_mask'])
        trg_2,_ = self.audio_self_Trans(token2['input_values'], key_padding_mask=token2['attention_mask'])

        CLR_loss = torch.tensor(0, device=trg_2.device)

        if self.encoder_cross != None:
            trg_1, trg_2, attn1 = self.encoder_cross(trg_1, trg_2, token1['attention_mask'], token2['attention_mask'])
            # trg_1_n, trg_2_n = self.encoder_cross(trg_1, trg_2, token1['attention_mask'], token2['attention_mask'])

        out = torch.cat([trg_1.mean(1), trg_2.mean(1)], -1)
        # out = torch.cat([trg_1.mean(1), trg_1_n.mean(1), trg_2.mean(1), trg_2_n.mean(1)], -1)
        out = self.classifier(out)

        return CLR_loss, CLR_loss, CLR_loss, out, attn1

    def flops(self):
        flops = 0
        flops += self.text_self_Trans.flops()
        flops += self.audio_self_Trans.flops()
        flops += self.encoder_cross.flops()

        flops += self.fc_dim * self.fc_dim//2
        flops += self.fc_dim//2 * self.fc_dim//4
        flops += self.fc_dim//4 * self.cfg.model.num_classes
        return flops


if __name__ == '__main__':
    from dataset import build_iemocap_dataset
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    import pandas as pd

    cfg = OmegaConf.load('configs/mm-pretraining.yaml')
    train_dataset, test_dataset = build_iemocap_dataset(cfg)
    dataloader = DataLoader(train_dataset, batch_size=4)#, collate_fn=train_dataset.collate_fn)
    model = MMModel(cfg, pretrained=cfg.model.pretrained)#.to(cfg.device)
    itr = iter(dataloader)
    x1, x2, m1, m2, label, id = next(itr)
    _, _, l_clr, out = model(x1, x2, m1, m2)
    loss_cls = nn.CrossEntropyLoss()
    l_cls = loss_cls(out, label)
    loss = l_cls + l_clr
    loss.backward()
    print(x1['input_ids'].grad)
    print(x1['input_ids'].requires_grad)
    # for name, param in model.named_parameters():
    #     # if param.grad is None:
    #     if not param.requires_grad:
    #         print(name)