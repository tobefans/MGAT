# -*- coding: utf-8 -*-
"""
Created on 2022.05.17

@author: fan.weiquan
"""

import numpy as np
import pandas as pd
from scipy import io
from sklearn import preprocessing
from sklearn import metrics
from omegaconf import OmegaConf
import time
import random
import os
import shutil
import argparse
import torch
import torch.utils.data as Data
import torch.nn as nn
from torch.optim import lr_scheduler
from dataset import build_meld_dataset, build_iemocap_dataset
# from model_mmm import PretrainedModel as Model
from model_mmm import MMModel as Model
from loss import FocalLoss
from utils import AverageMeter
import matplotlib.pyplot as plt
from tqdm import tqdm
# torch.multiprocessing.set_start_method('spawn')
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
plt.switch_backend('agg')
# system setting
plt.ion()
np.set_printoptions(suppress=True)

# CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2 --master_port 53453 train.py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-C", default='configs/mm-pretraining.yaml', type=str)
    parser.add_argument("--pretrained_model", "-M", default='', type=str)
    parser.add_argument("--evaluation", "-E", action='store_true')
    parser.add_argument("--gpus", "-G", default='0', type=str)
    parser.add_argument("--ddp", "-D", action='store_true', default=False)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if args.ddp:
        args.world_size = len(args.gpus.split(','))
        port_id = 10000 + np.random.randint(0, 1000)
        args.dist_url = 'tcp://127.0.0.1:' + str(port_id)
        mp.spawn(run,
            args=(args,),
            nprocs=args.world_size,
            join=True)
    else:
        run(int(args.gpus), args)

def saving_code(file_dir, file_dir_dst, ext):
    for root, dirs, files in os.walk(file_dir):  
        for file in files:
            if file.split('.')[-1] in ext:  
                src = os.path.join(root, file)
                dst =  os.path.join(file_dir_dst, file)
                shutil.copyfile(src, dst) 

def run(local_rank, args):
    # local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    cfg = OmegaConf.load(args.config)

    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.ddp:
        dist.init_process_group(backend='nccl', init_method=args.dist_url, rank=local_rank, world_size=args.world_size)

    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.log_dir, 'code'), exist_ok=True)
    # os.makedirs(os.path.join(cfg.log_dir, 'checkpoints'), exist_ok=True)
    # os.makedirs(os.path.join(cfg.log_dir, 'results'), exist_ok=True)
    saving_code('.', os.path.join(cfg.log_dir, 'code'), ['py', 'yaml'])

    num_session = range(1,6,1) if cfg.dataset.corps == 'iemocap' else range(1)

    folds_test_last_ua, folds_test_last_wa = [], []
    folds_test_best_ua, folds_test_best_wa = [], []
    folds_test_best_epoch = []
    for sess in num_session:
        fn_build_dataset = build_meld_dataset if cfg.dataset.corps == 'meld' else build_iemocap_dataset
        train_dataset, dev_dataset, test_dataset = fn_build_dataset(cfg, sess)
        model = Model(cfg, pretrained=cfg.model.pretrained).to(local_rank)
        if args.pretrained_model != '':
            pretrained = torch.load(args.pretrained_model, map_location='cuda:'+str(local_rank))
            state_dict = pretrained['state_dict'] if 'state_dict' in pretrained else pretrained
            model.load_state_dict(state_dict)
            # start_epoch = pretrained['epoch']
        if not args.evaluation:
            if args.ddp:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                train_loader = Data.DataLoader(dataset=train_dataset, batch_size=cfg.train.train_batch_size, num_workers=4, sampler=train_sampler)
            else:
                train_loader = Data.DataLoader(dataset=train_dataset, batch_size=cfg.train.train_batch_size, num_workers=4, shuffle=True)
        # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        # test_loader = Data.DataLoader(dataset=test_dataset, batch_size=cfg.train.val_batch_size, num_workers=4, collate_fn=test_dataset.collate_fn, sampler=test_sampler)
        dev_loader = Data.DataLoader(dataset=dev_dataset, batch_size=cfg.train.val_batch_size, num_workers=4, shuffle=False)
        test_loader = Data.DataLoader(dataset=test_dataset, batch_size=cfg.train.val_batch_size, num_workers=4, shuffle=False)
        
        if not args.ddp or dist.get_rank() == 0:
            num_params_encoder_all = sum(param.numel() for param in model.parameters())
            num_trainable_params_encoder_all = sum(param.numel() for param in model.parameters() if param.requires_grad)
            # print('The number of parameters of the total model: All-{}, Trainable-{}, Freeze-{}'.format(num_params_encoder_all, num_trainable_params_encoder_all, num_params_encoder_all-num_trainable_params_encoder_all))
            print('All parameters: ', num_params_encoder_all)
            print('Trainable:      ', num_trainable_params_encoder_all)
            print('Freeze:         ', num_params_encoder_all-num_trainable_params_encoder_all)

        if args.ddp:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        # num_samples_per_class = [48623,21117,5406,9599]
        # weight = 1e4 / np.array(num_samples_per_class)
        # loss_cls = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1.5,1,1]).to(local_rank)).to(local_rank)
        # loss_mmm = nn.MSELoss(reduction='none').to(local_rank)
        loss_cls = nn.CrossEntropyLoss().to(local_rank)
        # loss_cls = FocalLoss(alpha=[1,1,1,1], gamma=2, num_classes=cfg.model.num_classes).to(local_rank)
        # loss_cls = FocalLoss(alpha=weight, gamma=2, num_classes=cfg.model.num_classes).to(local_rank)
        lambda_clr = [0 for _ in range(cfg.train.num_epochs)]
        lambda_mmm = [0 for _ in range(cfg.train.num_epochs)]
        lambda_cls = [1 for _ in range(cfg.train.num_epochs)]
        # lambda_mmm = [1 if i<cfg.train.num_epochs//2 else 0 for i in range(cfg.train.num_epochs)]
        # lambda_cls = [0 if i<cfg.train.num_epochs//2 else 1 for i in range(cfg.train.num_epochs)]
        # lambda_mmm = [1 if i%2==0 else 0 for i in range(cfg.train.num_epochs)]
        # lambda_cls = [0 if i%2==0 else 1 for i in range(cfg.train.num_epochs)]
        # lambda_mmm = np.linspace(1, 0, cfg.train.num_epochs)
        # lambda_cls = np.linspace(0, 1, cfg.train.num_epochs)

        if args.evaluation:
            if not args.ddp or dist.get_rank() == 0:
                info_test = test(args, cfg, model, test_loader, loss_cls, lambda_mmm, lambda_cls, lambda_clr, local_rank, sess, 0)
                # print(info_test['id'][:20])
                continue
        opt = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=cfg.train.lr, momentum=0.9, weight_decay=cfg.train.weight_decay)
        # opt = torch.optim.SGD(model.parameters(), lr=cfg.train.lr, momentum=0.9, weight_decay=cfg.train.weight_decay)
        # opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=cfg.train.weight_decay)
        # scheduler = lr_scheduler.StepLR(opt, step_size=cfg.train.step_size, gamma=cfg.train.gamma)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=cfg.train.T0, T_mult=2)
        # scheduler = lr_scheduler.CosineAnnealingLR(opt, cfg.train.num_epochs, eta_min=0.1*cfg.train.lr)
        if not args.ddp or dist.get_rank() == 0:
            print('log saved in ', cfg.log_dir)
            print('weights: clr: {}'.format(lambda_clr))
            print('weights: mmm: {}'.format(lambda_mmm))
            print('weights: cls: {}'.format(lambda_cls))
        
        loss_train, loss_test = [], []
        ua_train, ua_test = [], []
        wa_train, wa_test = [], []

        best_result = 0.0
        best_pred = 0
        index_epoch = 0
        for epoch in range(cfg.train.num_epochs):
            start_time = time.time()
            info_train = train(args, cfg, model, train_loader, loss_cls, lambda_mmm, lambda_cls, lambda_clr, local_rank, sess, epoch, opt)
            scheduler.step()
            
            if not args.ddp or dist.get_rank() == 0:
                info_dev = test(args, cfg, model, dev_loader, loss_cls, lambda_mmm, lambda_cls, lambda_clr, local_rank, sess, epoch)
            
                loss_train.append(info_train['loss'])
                ua_train.append(info_train['ua'])
                wa_train.append(info_train['wa'])
                loss_test.append(info_dev['loss'])
                ua_test.append(info_dev['ua'])
                wa_test.append(info_dev['wa'])

                if info_dev['wa'] + info_dev['ua'] > best_result:
                    index_epoch = epoch
                    best_result = info_dev['wa'] + info_dev['ua']
                    best_id = info_dev['id']
                    best_pred = info_dev['pred']
                    best_actu = info_dev['label']
                    if args.ddp:
                        torch.save({'state_dict': model.module.state_dict(), 'epoch': epoch, 'wa': info_dev['wa'], 'ua':info_dev['ua']}, os.path.join(cfg.log_dir, 'session{}.pkl'.format(sess)))
                    else:
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'wa': info_dev['wa'], 'ua':info_dev['ua']}, os.path.join(cfg.log_dir, 'session{}.pkl'.format(sess)))
            
            time_epoch = time.time() - start_time
            if not args.ddp or dist.get_rank() == 0: print('Exp {}: Session {:.0f} Epoch {:.0f} completed with {:.0f}m {:.0f}s'.format(cfg.log_dir.split('/')[-1], sess, epoch, time_epoch // 60, time_epoch % 60))

        if not args.ddp or dist.get_rank() == 0: 
            dataframe = pd.DataFrame({'loss_train':loss_train, 'loss_test':loss_test, 'ua_train':ua_train, 'ua_test':ua_test, 'wa_train':wa_train,'wa_test':wa_test})  
            dataframe.to_csv(os.path.join(cfg.log_dir, 'session{}_loss_metrics.csv'.format(sess)), index=False, sep=',')

            if args.ddp:
                torch.save({'state_dict': model.module.state_dict(), 'epoch': epoch, 'wa': info_dev['wa'], 'ua':info_dev['ua']}, os.path.join(cfg.log_dir, 'session{}_last.pkl'.format(sess)))
            else:
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'wa': info_dev['wa'], 'ua':info_dev['ua']}, os.path.join(cfg.log_dir, 'session{}_last.pkl'.format(sess)))
                
            ## plot
            fig,axes=plt.subplots(2,3)
            axes[0,0].plot(loss_train, color="blue", lw = 2.5, linestyle="-")
            axes[0,1].plot(wa_train, color="black", lw = 2.5, linestyle="-")
            axes[0,2].plot(ua_train, color="black", lw = 2.5, linestyle="-")
            axes[1,0].plot(loss_test, color="blue", lw = 2.5, linestyle="-")
            axes[1,1].plot(wa_test, color="black", lw = 2.5, linestyle="-")
            axes[1,2].plot(ua_test, color="black", lw = 2.5, linestyle="-")
            fig.savefig(os.path.join(cfg.log_dir, 'session{}_curve.png'.format(sess)))

            model_early = Model(cfg, pretrained=cfg.model.pretrained).to(local_rank)
            params = torch.load(os.path.join(cfg.log_dir, 'session{}.pkl'.format(sess)), map_location='cuda:'+str(local_rank))
            model_early.load_state_dict(params['state_dict'])
            info_test = test(args, cfg, model_early, test_loader, loss_cls, lambda_mmm, lambda_cls, lambda_clr, local_rank, sess, epoch)
            info_test_last = test(args, cfg, model, test_loader, loss_cls, lambda_mmm, lambda_cls, lambda_clr, local_rank, sess, epoch)
            
            print(metrics.classification_report(info_test['label'], info_test['pred']))
            print(metrics.confusion_matrix(info_test['label'], info_test['pred']))
            dataframe = pd.DataFrame({'id':info_test['id'], 'prediction':info_test['pred'], 'prediction_last':info_test_last['pred'], 'label':info_test['label']})
            dataframe.to_csv(os.path.join(cfg.log_dir, 'session{}_test_pred.csv'.format(sess)), index=False, sep=',')
            
            folds_test_best_epoch.append(index_epoch)
            folds_test_best_wa.append(info_test['wa'])
            folds_test_best_ua.append(info_test['ua'])
            folds_test_last_wa.append(info_test_last['wa'])
            folds_test_last_ua.append(info_test_last['ua'])
            
            dataframe = pd.DataFrame({'folds_test_last_ua':folds_test_last_ua, 'folds_test_last_wa':folds_test_last_wa, 'folds_test_best_ua':folds_test_best_ua, 'folds_test_best_wa':folds_test_best_wa, 'folds_test_best_epoch':folds_test_best_epoch})
            dataframe.to_csv(os.path.join(cfg.log_dir, 'all_session_test_metrics.csv'), index=False, sep=',')
        
            # print(model.encoder_cross.cross_1.layers[0].blocks[0])

    if (not args.evaluation) and (not args.ddp or dist.get_rank() == 0): 
        print('TEST:: Last: WA: ', np.around(np.mean(folds_test_last_wa), 4), 'UA: ', np.around(np.mean(folds_test_last_ua), 4))
        print('TEST:: Best: WA: ', np.around(np.mean(folds_test_best_wa), 4), 'UA: ', np.around(np.mean(folds_test_best_ua), 4))
        print(np.around(folds_test_best_wa, 4))
        print(np.around(folds_test_best_ua, 4))
        print(folds_test_best_epoch)
        print('saved in {}.'.format(cfg.log_dir))

    
def train(args, cfg, model, dataloader, loss_cls, lambda_mmm, lambda_cls, lambda_clr, local_rank, sess, epoch, opt):
    if args.ddp: dataloader.sampler.set_epoch(epoch)
    model.train()

    # Trackers
    loss_tracker = AverageMeter('loss')
    loss_cls_tracker = AverageMeter('loss')
    loss_clr_tracker = AverageMeter('loss')
    loss_mmm1_tracker = AverageMeter('loss')
    loss_mmm2_tracker = AverageMeter('loss')

    loss_tr = 0.0
    pred_all, actu_all = [], []
    id_all = []
    with tqdm(dataloader, unit="batch", desc=f'TRAIN:: Epoch: {epoch}/{cfg.train.num_epochs} ',
        bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}', ascii=" #") as iterator:
        for step, (x1, x2, m1, m2, label, id) in enumerate(iterator, 0):
            label = label.to(local_rank)
            m1, m2 = m1.to(local_rank), m2.to(local_rank)
            x1['input_ids'], x1['attention_mask'] = x1['input_ids'].to(local_rank), x1['attention_mask'].to(local_rank)
            x2['input_values'], x2['attention_mask'] = x2['input_values'].to(local_rank), x2['attention_mask'].to(local_rank)
            is_clr = True if lambda_clr[epoch]!= 0 else False
            l_mmm1, l_mmm2, l_clr, out, _ = model(x1, x2, m1, m2, is_clr=is_clr)
            l_mmm = l_mmm1 + l_mmm2
            l_cls = loss_cls(out, label)
            loss = lambda_mmm[epoch]*l_mmm + lambda_cls[epoch]*l_cls + lambda_clr[epoch]*l_clr
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(C2.parameters(), 0.5)
            opt.step()
            opt.zero_grad()
            # model.ema_step()

            # if step % 100 == 0:
            #     if not args.ddp or dist.get_rank() == 0: print('Epoch {} Step {}: \t Loss: {:.3f} \t Loss_mmm: {:.3f}-{}-{} \t Loss_cls: {:.3f}'.format(epoch, step, loss, l_mmm, int(l_mmm1), int(l_mmm2), l_cls))

            pred = torch.max(out.cpu().data,1)[1].numpy()
            actu = label.cpu().data.numpy()
            # pred = actu
            pred_all = pred_all + list(pred)
            actu_all = actu_all + list(actu)
            id_all = id_all + list(id)

            loss_tr += loss.cpu().item()

            loss_tracker.update(loss.cpu().item())
            loss_cls_tracker.update(l_cls.cpu().item())
            loss_clr_tracker.update(l_clr.cpu().item())
            if len(x1)!=0: loss_mmm1_tracker.update(l_mmm1.cpu().item())
            if len(x2)!=0: loss_mmm2_tracker.update(l_mmm2.cpu().item())
            iterator.set_postfix(loss=loss_tracker.avg, l_clr=loss_clr_tracker.avg, l_m1=loss_mmm1_tracker.avg, l_m2=loss_mmm2_tracker.avg, l_c=loss_cls_tracker.avg)

    loss_tr = loss_tr / len(dataloader.dataset)

    pred_all, actu_all = np.array(pred_all), np.array(actu_all)
    ua_tr = metrics.recall_score(actu_all, pred_all, average='macro')
    wa_tr = metrics.accuracy_score(actu_all, pred_all)
    # print(actu_all)
    # print(pred_all)
    if not args.ddp or dist.get_rank() == 0:
        print('TRAIN:: Epoch: ', epoch, '| Loss: %.4f' % loss_tr, '| ua: %.4f' % ua_tr, '| wa: %.4f' % wa_tr, '| lr: %.5f' % opt.param_groups[0]['lr'])
        # torch.save(model.module.state_dict(), os.path.join(cfg.log_dir, 'checkpoints', 'session{}_{}.pkl'.format(sess, epoch)))
    
    return {'loss': loss_tr, 'ua': ua_tr, 'wa': wa_tr, 'pred': pred_all, 'label': actu_all, 'id': id_all}

def test(args, cfg, model, dataloader, loss_cls, lambda_mmm, lambda_cls, lambda_clr, local_rank, sess, epoch):
    with torch.no_grad():
        # dataloader.sampler.set_epoch(epoch)
        model.eval()
        # Trackers
        loss_tracker = AverageMeter('loss')
        loss_cls_tracker = AverageMeter('loss')
        loss_clr_tracker = AverageMeter('loss')
        loss_mmm1_tracker = AverageMeter('loss')
        loss_mmm2_tracker = AverageMeter('loss')

        loss_te = 0.0
        pred_all, actu_all = [], []
        id_all = []
        with tqdm(dataloader, unit="batch", desc=f'TEST:: Epoch: {epoch}/{cfg.train.num_epochs} ',
                    bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}', ascii=" #") as iterator:
            for step, (x1, x2, m1, m2, label, id) in enumerate(iterator, 0):
                label = label.to(local_rank)
                m1, m2 = m1.to(local_rank), m2.to(local_rank)
                x1['input_ids'], x1['attention_mask'] = x1['input_ids'].to(local_rank), x1['attention_mask'].to(local_rank)
                x2['input_values'], x2['attention_mask'] = x2['input_values'].to(local_rank), x2['attention_mask'].to(local_rank)
                is_clr = True if lambda_clr[epoch]!= 0 else False
                l_mmm1, l_mmm2, l_clr, out, _ = model(x1, x2, m1, m2, is_clr=is_clr)
                l_mmm = l_mmm1 + l_mmm2
                # l_cls = l_mmm
                # x, y, x1, y1, x2, y2, out = model(x1m, x1, m1, x2, m2)
                # if len(x)==0: continue
                # l_mmm1 = loss_mmm(x1.float(), y1.float()).sum() if len(x1)!=0 else torch.tensor(0., device=x1.local_rank)
                # l_mmm2 = loss_mmm(x2.float(), y2.float()).sum() if len(x2)!=0 else torch.tensor(0., device=x2.local_rank)
                # l_mmm = (l_mmm1 + l_mmm2).div(x.size(0))
                l_cls = loss_cls(out, label)
                loss = lambda_mmm[epoch]*l_mmm + lambda_cls[epoch]*l_cls + lambda_clr[epoch]*l_clr

                pred = torch.max(out.cpu().data,1)[1].numpy()
                actu = label.cpu().data.numpy()
                # pred = actu
                pred_all = pred_all + list(pred)
                actu_all = actu_all + list(actu)
                id_all = id_all + list(id)

                loss_te += loss.cpu().item()

                loss_tracker.update(loss.cpu().item())
                loss_cls_tracker.update(l_cls.cpu().item())
                loss_clr_tracker.update(l_clr.cpu().item())
                if len(x1)!=0: loss_mmm1_tracker.update(l_mmm1.cpu().item())
                if len(x2)!=0: loss_mmm2_tracker.update(l_mmm2.cpu().item())
                iterator.set_postfix(loss=loss_tracker.avg, l_clr=loss_clr_tracker.avg, l_m1=loss_mmm1_tracker.avg, l_m2=loss_mmm2_tracker.avg, l_c=loss_cls_tracker.avg)

    loss_te = loss_te / len(dataloader.dataset)

    pred_all, actu_all = np.array(pred_all), np.array(actu_all)
    ua_te = metrics.recall_score(actu_all, pred_all, average='macro')
    wa_te = metrics.accuracy_score(actu_all, pred_all)

    if not args.ddp or dist.get_rank() == 0:
        print('TEST :: Epoch: ', epoch, '| Loss: %.4f' % loss_te, '| ua: %.4f' % ua_te, '| wa: %.4f' % wa_te)

    return {'loss': loss_te, 'ua': ua_te, 'wa': wa_te, 'pred': pred_all, 'label': actu_all, 'id': id_all}

if __name__=="__main__":
    main()