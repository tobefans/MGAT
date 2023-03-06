""" PyTorch MM model."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from module import layers
import transformers

class EarlyConcatModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        config = getattr(transformers, cfg.model.config_cross.config_name_cat)(**cfg.model.config_cross.rewrite_param_cat)
        self.cross_cat = getattr(layers, cfg.model.config_cross.model_base)(cfg=cfg.model.config_cross.rewrite_param_cat, config=config, len_q=cfg.dataset.max_len_t+cfg.dataset.max_len_a//cfg.dataset.pool_feat_audio, len_k=cfg.dataset.max_len_t+cfg.dataset.max_len_a//cfg.dataset.pool_feat_audio)

    def forward(self, input_values1, input_values2, attention_mask1=None, attention_mask2=None, 
                output_hidden_states=False, output_attentions=False, **kwargs):
        input = torch.cat([input_values1, input_values2], 1)
        attn = torch.cat([attention_mask1, attention_mask2], 1)
        out = self.cross_cat(input, attention_mask=attn, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        out1, out2 = torch.split(out, [attention_mask1.size(-1), attention_mask2.size(-1)], dim=1)
        return out1, out2


class LateConcatModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        config = getattr(transformers, cfg.model.config_cross.config_name_1)(**cfg.model.config_cross.rewrite_param_1)
        self.cross_1 = getattr(layers, cfg.model.config_cross.model_base)(cfg=cfg.model.config_cross.rewrite_param_1, config=config, len_q=cfg.dataset.max_len_t, len_k=cfg.dataset.max_len_a//cfg.dataset.pool_feat_audio)
        config = getattr(transformers, cfg.model.config_cross.config_name_2)(**cfg.model.config_cross.rewrite_param_2)
        self.cross_2 = getattr(layers, cfg.model.config_cross.model_base)(cfg=cfg.model.config_cross.rewrite_param_2, config=config, len_q=cfg.dataset.max_len_a//cfg.dataset.pool_feat_audio, len_k=cfg.dataset.max_len_t)

    def forward(self, input_values1, input_values2, attention_mask1=None, attention_mask2=None, 
                output_hidden_states=False, output_attentions=False, **kwargs):
        out1 = self.cross_1(input_values1, attention_mask=attention_mask1, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        out2 = self.cross_2(input_values2, attention_mask=attention_mask2, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        return out1, out2


class HierarchicalM2OModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        config = getattr(transformers, cfg.model.config_cross.config_name_1)(**cfg.model.config_cross.rewrite_param_1)
        self.cross_1 = getattr(layers, cfg.model.config_cross.model_base)(cfg=cfg.model.config_cross.rewrite_param_1, config=config, len_q=cfg.dataset.max_len_t, len_k=cfg.dataset.max_len_a//cfg.dataset.pool_feat_audio)
        config = getattr(transformers, cfg.model.config_cross.config_name_2)(**cfg.model.config_cross.rewrite_param_2)
        self.cross_2 = getattr(layers, cfg.model.config_cross.model_base)(cfg=cfg.model.config_cross.rewrite_param_2, config=config, len_q=cfg.dataset.max_len_a//cfg.dataset.pool_feat_audio, len_k=cfg.dataset.max_len_t)
        config = getattr(transformers, cfg.model.config_cross.config_name_cat)(**cfg.model.config_cross.rewrite_param_cat)
        self.cross_cat = getattr(layers, cfg.model.config_cross.model_base)(cfg=cfg.model.config_cross.rewrite_param_cat, config=config, len_q=cfg.dataset.max_len_t+cfg.dataset.max_len_a//cfg.dataset.pool_feat_audio, len_k=cfg.dataset.max_len_t+cfg.dataset.max_len_a//cfg.dataset.pool_feat_audio)

    def forward(self, input_values1, input_values2, attention_mask1=None, attention_mask2=None, 
                output_hidden_states=False, output_attentions=False, **kwargs):
        out1 = self.cross_1(input_values1, attention_mask=attention_mask1, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        out2 = self.cross_2(input_values2, attention_mask=attention_mask2, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        out = torch.cat([out1, out2], 1)
        attn = torch.cat([attention_mask1, attention_mask2], 1)
        out = self.cross_cat(out, attention_mask=attn, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        out1, out2 = torch.split(out, [attention_mask1.size(-1), attention_mask2.size(-1)], dim=1)
        return out1, out2


class HierarchicalO2MModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        config = getattr(transformers, cfg.model.config_cross.config_name_cat)(**cfg.model.config_cross.rewrite_param_cat)
        self.cross_cat = getattr(layers, cfg.model.config_cross.model_base)(cfg=cfg.model.config_cross.rewrite_param_cat, config=config, len_q=cfg.dataset.max_len_t+cfg.dataset.max_len_a//cfg.dataset.pool_feat_audio, len_k=cfg.dataset.max_len_t+cfg.dataset.max_len_a//cfg.dataset.pool_feat_audio)
        config = getattr(transformers, cfg.model.config_cross.config_name_1)(**cfg.model.config_cross.rewrite_param_1)
        self.cross_1 = getattr(layers, cfg.model.config_cross.model_base)(cfg=cfg.model.config_cross.rewrite_param_1, config=config, len_q=cfg.dataset.max_len_t, len_k=cfg.dataset.max_len_a//cfg.dataset.pool_feat_audio)
        config = getattr(transformers, cfg.model.config_cross.config_name_2)(**cfg.model.config_cross.rewrite_param_2)
        self.cross_2 = getattr(layers, cfg.model.config_cross.model_base)(cfg=cfg.model.config_cross.rewrite_param_2, config=config, len_q=cfg.dataset.max_len_a//cfg.dataset.pool_feat_audio, len_k=cfg.dataset.max_len_t)

    def forward(self, input_values1, input_values2, attention_mask1=None, attention_mask2=None, 
                output_hidden_states=False, output_attentions=False, **kwargs):
        input = torch.cat([input_values1, input_values2], 1)
        attn = torch.cat([attention_mask1, attention_mask2], 1)
        out = self.cross_cat(input, attention_mask=attn, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        # out1, out2 = torch.split(out, [attention_mask1.size(-1), attention_mask2.size(-1)], dim=1)
        out1 = self.cross_1(out[:,:attention_mask1.size(-1)], attention_mask=attention_mask1, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        out2 = self.cross_2(out[:,attention_mask1.size(-1):], attention_mask=attention_mask2, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        return out1, out2


class BaseCrossAttentionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        config = getattr(transformers, cfg.model.config_cross.config_name_1)(**cfg.model.config_cross.rewrite_param_1)
        self.cross_1 = getattr(layers, cfg.model.config_cross.model_base)(cfg=cfg.model.config_cross.rewrite_param_1, config=config, len_q=cfg.dataset.max_len_t, len_k=cfg.dataset.max_len_a//cfg.dataset.pool_feat_audio)
        config = getattr(transformers, cfg.model.config_cross.config_name_2)(**cfg.model.config_cross.rewrite_param_2)
        self.cross_2 = getattr(layers, cfg.model.config_cross.model_base)(cfg=cfg.model.config_cross.rewrite_param_2, config=config, len_q=cfg.dataset.max_len_a//cfg.dataset.pool_feat_audio, len_k=cfg.dataset.max_len_t)

    def forward(self, input_values1, input_values2, attention_mask1=None, attention_mask2=None, 
                output_hidden_states=False, output_attentions=False, **kwargs):
        out1 = self.cross_1(input_values1, input_values2, attention_mask=attention_mask1, attention_mask_kv=attention_mask2, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        out2 = self.cross_2(input_values2, input_values1, attention_mask=attention_mask2, attention_mask_kv=attention_mask1, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        return out1, out2

from module.layers import build_transformer
class CrossAttentionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        config = getattr(transformers, cfg.model.config_cross.config_name_1)(**cfg.model.config_cross.rewrite_param_1)
        self.cross_1 = getattr(layers, cfg.model.config_cross.model_base)(cfg=cfg.model.config_cross.rewrite_param_1, config=config, len_q=cfg.dataset.max_len_t, len_k=cfg.dataset.max_len_a//cfg.dataset.pool_feat_audio)
        config = getattr(transformers, cfg.model.config_cross.config_name_2)(**cfg.model.config_cross.rewrite_param_2)
        self.cross_2 = getattr(layers, cfg.model.config_cross.model_base)(cfg=cfg.model.config_cross.rewrite_param_2, config=config, len_q=cfg.dataset.max_len_a//cfg.dataset.pool_feat_audio, len_k=cfg.dataset.max_len_t)
        # config = getattr(transformers, cfg.model.config_cross.config_name_cat)(**cfg.model.config_cross.rewrite_param_cat)
        # self.cross_cat = getattr(layers, cfg.model.config_cross.model_base)(cfg=cfg.model.config_cross.rewrite_param_cat, config=config, len_q=cfg.dataset.max_len_t+cfg.dataset.max_len_a//cfg.dataset.pool_feat_audio, len_k=cfg.dataset.max_len_t+cfg.dataset.max_len_a//cfg.dataset.pool_feat_audio)
        self.cross_cat = build_transformer(self_attn=True, num_layers=3, embed_dim=cfg.model.embed_dim, ffn_embed_dim=512, num_heads=8)

    def forward(self, input_values1, input_values2, attention_mask1=None, attention_mask2=None, 
                output_hidden_states=False, output_attentions=False, **kwargs):
        # out1 = self.cross_1(input_values1, input_values2, attention_mask=attention_mask1, attention_mask_kv=attention_mask2, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        # out2 = self.cross_2(input_values2, input_values1, attention_mask=attention_mask2, attention_mask_kv=attention_mask1, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        # out = torch.cat([out1, out2], 1)
        # attn = torch.cat([attention_mask1, attention_mask2], 1)
        # out = self.cross_cat(out, attention_mask=attn, attention_mask_kv=attn, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        # out1, out2 = torch.split(out, [attention_mask1.size(-1), attention_mask2.size(-1)], dim=1)
        # # out1, out2 = out[:, :attention_mask1.size(-1)], out[:, attention_mask1.size(-1):]
        out1, attn1 = self.cross_1(input_values1, input_values2, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        out2, attn2 = self.cross_2(input_values2, input_values1, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        out = torch.cat([out1, out2], 1)
        attn = torch.cat([attention_mask1, attention_mask2], 1)
        # out, attn = self.cross_cat(out, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        out, attn = self.cross_cat(out)
        out1, out2 = torch.split(out, [attention_mask1.size(-1), attention_mask2.size(-1)], dim=1)
        return out1, out2, attn1

    def flops(self):
        flops = 0
        flops += self.cross_1.flops()
        flops += self.cross_2.flops()
        flops += self.cross_cat.flops()
        return flops
