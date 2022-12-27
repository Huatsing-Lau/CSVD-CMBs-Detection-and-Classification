# -*- coding: utf-8 -*-
# 影像特征+临床信息

# +
import os

import numpy as np
import pdb
import math
from itertools import islice
import collections

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler

from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.preprocessing import label_binarize

from scipy.stats import spearmanr

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -

def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [N/len(dataset.patient_cls_ids[c]) for c in range(len(dataset.patient_cls_ids))]           
    weight = [0] * int(N)
    for idx in range(len(dataset)):   
        y = dataset.getclasslabel(idx)  
        weight[idx] = weight_per_class[y]      
        
    return torch.DoubleTensor(weight)


class SubsetSequentialSampler(Sampler):
    """
    Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_split_loader(split_dataset, training = False, testing = False, weighted = False):
    """
    return either the validation loader or training loader 
    input arguments:
        testing: 是否代码调试模式。
    """
#     kwargs = {'num_workers': 0} if device.type == "cuda" else {}
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_MIL_mtl_concat, **kwargs)	
            else:
                loader = DataLoader(split_dataset, batch_size=1, sampler = RandomSampler(split_dataset), collate_fn = collate_MIL_mtl_concat, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL_mtl_concat, **kwargs)
    else:
        ids = np.random.choice(np.arange(len(split_dataset)), int(len(split_dataset)*0.1), replace = False)
        loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate_MIL_mtl_concat, **kwargs )

    return loader


def collate_MIL_mtl_concat(batch):
    """
    一个batch里边，每个病例是一个tuple: (img, label) 或者 (img, label, clinical).
    本函数将不同病例的tuple整理为一个图像变量和一个标签变量。
    """
    # 图像
    # 多模态(tuple)
    if isinstance(batch[0][0],tuple):
        img = tuple([torch.cat([item[0][k] for item in batch], dim=0) for k in range(len(batch[0][0]))])
    # 多模态(dict)
    # 此处默认每个字典的词条都一样.实际上只有一个字典(batch size==1)
    elif isinstance(batch[0][0],dict):
        img = dict()
        for modal_name in batch[0][0].keys():
            img[modal_name] = torch.cat([item[0][modal_name] for item in batch], dim=0)
    # 单模态
    elif isinstance(batch[0][0],torch.Tensor):
        img = torch.cat([item[0] for item in batch], dim=0)
    
    # label
    if isinstance(batch[0][1],dict):
        # dual task
        label = torch.Tensor([list(item[1].values()) for item in batch]).float()
    else:
        # single task
        label = torch.Tensor([item[1] for item in batch]).float()
        
    # 临床信息
    if len(batch[0])==3:
        if isinstance(batch[0][2],dict):
            #clin = [item for item in batch[0][2].values()]
            clin = torch.cat( [torch.tensor([v for v in item[2].values()]).unsqueeze(0) for item in batch],dim=0)
        elif isinstance(batch[0][2],tuple) or isinstance(batch[0][2],list):
            #clin = batch[0][2]
            clin = torch.cat( [torch.tensor(item[2]).unsqueeze(0) for item in batch],dim=0)
        return [img, label, clin]
    else:
        return [img, label]



class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, count, correct, c):
        self.data[c]["count"] += count
        self.data[c]["correct"] += correct
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            tpr = None
        else:
            tpr = float(correct) / count
        
        return tpr, correct, count


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=100, verbose=False):#stop_epoch=50
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, ckpt_name)
        self.val_loss_min = val_loss


# +
def train(model, datasets, cur, args):
    """ 
    train for a single fold
    """
    print('Training Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('Init train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

#     loss_fn = nn.CrossEntropyLoss()#weight=torch.tensor(args.weight).cuda()
#     model.relocate()

    print('Init optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('Init Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split)
    test_loader = get_split_loader(test_split)
    print('Done!')

    print('Setup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = args.patience, stop_epoch=args.stop_epoch, verbose = True)#, stop_epoch=100

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epoch):
        train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, 
                   loss_fn=None, multi_modal=args.multi_modal, dual_task=args.dual_task, mix=args.mix)
        stop = validate(cur, epoch, model, val_loader, args.n_classes, early_stopping, writer, loss_fn=None, 
                        results_dir=args.results_dir, multi_modal=args.multi_modal, dual_task=args.dual_task, mix=args.mix)
        if stop: 
            break

    if args.early_stopping:
#         model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
        model = torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))).cuda()
    else:
#         torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
        torch.save(model, os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    
    if args.dual_task:
        _, cls_val_acc, cls_val_auc, score_val_coor, _= summary(model, val_loader, args.n_classes, multi_modal=args.multi_modal, dual_task=True, mix=args.mix)
        print( 'Validation Set, Cls acc: {:.4f}, Cls ROC AUC: {:.4f}, Score Corr: {:.4f}'.format(cls_val_acc, cls_val_auc, score_val_coor) )
        results_dict, cls_test_acc, cls_test_auc, score_test_coor, acc_loggers = summary(model, test_loader, args.n_classes, multi_modal=True, dual_task=True, mix=args.mix)
        print( 'Test Set, Cls acc: {:.4f}, Cls ROC AUC: {:.4f}, Score Corr: {:.4f}'.format(cls_test_acc, cls_test_auc, score_test_coor) )
    else:
        _, cls_val_acc, cls_val_auc, _= summary(model, val_loader, args.n_classes, multi_modal=args.multi_modal, dual_task=False, mix=args.mix)
        print( 'Validation Set, Cls acc: {:.4f}, Cls ROC AUC: {:.4f}'.format(cls_val_acc, cls_val_auc) )
        results_dict, cls_test_acc, cls_test_auc, acc_loggers = summary(model, test_loader, args.n_classes, multi_modal=args.multi_modal, dual_task=False, mix=args.mix)
        print( 'Test Set, Cls acc: {:.4f}, Cls ROC AUC: {:.4f}'.format(cls_test_acc, cls_test_auc,) )

        
    for i in range(args.n_classes):
        acc, correct, count = acc_loggers[0].get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_tpr'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/cls_val_acc', cls_val_acc, 0)
        writer.add_scalar('final/cls_val_auc', cls_val_auc, 0)
        writer.add_scalar('final/cls_test_acc', cls_test_acc, 0)
        writer.add_scalar('final/cls_test_auc', cls_test_auc, 0)
        if args.dual_task:
            writer.add_scalar('final/cls_val_score_coor', score_val_coor, 0)
            writer.add_scalar('final/cls_test_score_coor', score_test_coor, 0)
        writer.close()
    
    if args.dual_task:
        return results_dict, cls_test_acc, cls_val_acc, cls_test_auc, cls_val_auc, score_test_coor, score_val_coor
    else:
        return results_dict, cls_test_acc, cls_val_acc, cls_test_auc, cls_val_auc


# +
def train_loop(
    epoch, model, loader, optimizer, n_classes, 
    writer = None, loss_fn = None, 
    multi_modal=False, dual_task=False, mix=False):  
    """
    仅支持单任务。支持单模态和多模态。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    cls_logger = Accuracy_Logger(n_classes=n_classes)
    cls_train_loss = 0.
    score_train_loss = 0.
    print('\n')
    
    cls_probs = np.zeros((len(loader), n_classes))
    cls_labels = np.zeros(len(loader))
    scores = []
    score_labels = []
    
    for batch_idx, sample in enumerate(loader):
        data, label = sample[0], sample[1]
        if mix:
            assert len(sample)==3, 'sample must contains clinical information'
            clin = sample[2]
            clin = clin.float().to(device)
        if multi_modal:# 多个MR序列模型
            if isinstance(data,tuple):
                data = tuple([img.to(device) for img in data])
            elif isinstance(data,dict):
                for modal_name in data.keys():
                    data[modal_name] = data[modal_name].to(device)
        else:
            data =  data.to(device)

        if not dual_task:
            label = label[:,0]
        label = label.long().to(device)
        
#         import pdb
#         pdb.set_trace()
        if mix:
            output = model( (data,clin) )# ResTOADMix模型的输入是一个tuple
        else:
            output = model(data)

        if dual_task:
            logits = output[0]
            score = output[1].squeeze(1)
        else:
            logits = output
        Y_prob = torch.softmax(logits,-1)
        Y_hat = torch.argmax(Y_prob)
        
        
        if loss_fn is None:
            loss_fn = model.loss_fn
        if dual_task:
            cls_loss = loss_fn[0](logits, label[:,0].long())
            # 若score的label缺失(即label[:,1]是nan),则计算的得到的score_loss也是nan
            # 经过后向传播之后,网络各层的参数也变成了nan,之后的网路输出值将一直是nan.
            if not torch.isnan(label[:,1]):
                score_loss = loss_fn[1](score, label[:,1].float())
                loss = cls_loss+score_loss
                score_labels += [label[:,1].detach().item()]
                scores += [score.detach().cpu().numpy()]
            else:
                loss = cls_loss
            cls_labels[batch_idx] = label[:,0].detach().item()
            cls_logger.log(Y_hat, label[:,0])
        else:
            if isinstance(loss_fn,tuple):
                cls_loss = loss_fn[0](logits, label.long())
            else:
                cls_loss = loss_fn(logits, label.long())
            loss = cls_loss
            cls_labels[batch_idx] = label.detach().item()
            cls_logger.log(Y_hat, label)    
        cls_probs[batch_idx] = Y_prob.detach().cpu().numpy()
        
        
        cls_loss_value = cls_loss.item()  
        cls_train_loss += cls_loss_value
        if dual_task and not torch.isnan(label[:,1]):
            score_loss_value = score_loss.item()
            score_train_loss += score_loss_value
        
        
        if (batch_idx + 1) % 5 == 0:
            if dual_task:
                if multi_modal:
                    print('batch {}, cls loss: {:.4f}, cls label: {}, score label: {}, bag_size: {}'.format(batch_idx+1, cls_loss_value, label[0,0], label[0,1], [v.size(0) for _,v in data.items()]) )
                else:
                    print('batch {}, cls loss: {:.4f}, cls label: {}, score label: {}, bag_size: {}'.format(batch_idx+1, cls_loss_value, label[0,0], label[0,1], data.size(0)) )
            else:
                if multi_modal:
                    print('batch {}, cls loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx+1, cls_loss_value, label.item(), [v.size(0) for _,v in data.items()]) )
                else:
                    print('batch {}, cls loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx+1, cls_loss_value, label.item(), data.size(0)) )
            
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    score_labels = np.array(score_labels)
    scores = np.array(scores)
    
    # calculate loss for epoch
    cls_train_loss /= len(loader)
    if dual_task:
        score_train_loss /= score_labels.size 
        train_loss = cls_train_loss+score_train_loss
    else:
        train_loss = cls_train_loss
    
    #######################################################################################
    ## 计算auc 和 acc (和Corr)
    #######################################################################################
    if n_classes == 2:
        cls_auc = roc_auc_score(cls_labels, cls_probs[:,1])
        cls_aucs = []
    else:
        cls_aucs = []
        binary_labels = label_binarize(cls_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in cls_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], cls_probs[:, class_idx])
                cls_aucs.append(calc_auc(fpr, tpr))
            else:
                cls_aucs.append(float('nan'))

        cls_auc = np.nanmean(np.array(cls_aucs))
    cls_acc = metrics.accuracy_score(cls_labels, np.argmax(cls_probs,-1)) 
    if dual_task:
        score_coor = spearmanr(score_labels,scores)[0]
    #######################################################################################   
    if dual_task:
        print('Epoch: {}, cls train_loss: {:.4f}, cls acc: {:.4f}, cls auc: {:.4f}, score_train_loss: {:.4f}, score Corr: {:.4f}'.format(epoch, cls_train_loss, cls_acc, cls_auc, score_train_loss, score_coor))
    else:
        print('Epoch: {}, cls train_loss: {:.4f}, cls acc: {:.4f}, cls auc: {:.4f}'.format(epoch, cls_train_loss, cls_acc, cls_auc))
    
    for i in range(n_classes):
        tpr, correct, count = cls_logger.get_summary(i)
        print('class {}: accuracy {}, correct {}/{}'.format(i, tpr, correct, count))
        if writer:
            if tpr is None:
                acc = 0.0
            writer.add_scalar('train/class_{}_tpr'.format(i), tpr, epoch)

    if writer:
        writer.add_scalar('train/loss', loss, epoch)
        writer.add_scalar('train/cls_train_loss', cls_train_loss, epoch)
        writer.add_scalar('train/cls_acc', cls_acc, epoch)
        writer.add_scalar('train/cls_auc', cls_auc, epoch)
        if dual_task:
            writer.add_scalar('train/score_train_loss', score_train_loss, epoch)
            writer.add_scalar('train/score_coor', score_coor, epoch)

# +
# def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None, 
#                multi_modal=False, dual_task=False, mix=False):  
#     """
#     仅支持单任务。支持单模态和多模态。
#     魔改: 累计多个样本的loss之后，再返向传播，以便达到模拟batch_size>1的效果。
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
#     model.train()
#     cls_logger = Accuracy_Logger(n_classes=n_classes)
#     cls_train_loss = 0.
#     score_train_loss = 0.
#     print('\n')
    
#     cls_probs = np.zeros((len(loader), n_classes))
#     cls_labels = np.zeros(len(loader))
#     scores = []
#     score_labels = []
    
#     batch_size = 5
#     loss = None
#     for batch_idx, sample in enumerate(loader):
#         data, label = sample[0], sample[1]
#         if mix:
#             assert len(sample)==3, 'sample must contains clinical information'
#             clin = sample[2]
#             clin = clin.float().to(device)
#         # initial loss
#         if (batch_idx + 1) % batch_size == 1:
#             loss=None
#         else:
#             pass
        
#         if (batch_idx + 1) % batch_size != 0:
#             if multi_modal:
#                 if isinstance(data,tuple):
#                     data = tuple([img.to(device) for img in data])
#                 elif isinstance(data,dict):
#                     for modal_name in data.keys():
#                         data[modal_name] = data[modal_name].to(device)
#             else:
#                 data =  data.to(device)

#             if not dual_task:
#                 label = label[:,0].long()
#             label = label.to(device)
                

#             if mix:
#                 output = model(data,clin)
#             else:
#                 output = model(data)

#             if dual_task:
#                 logits = output[0]
#                 score = output[1].squeeze(1)
#             else:
#                 logits = output
#             Y_prob = torch.softmax(logits,-1)
#             Y_hat = torch.argmax(Y_prob)


#             if loss_fn is None:
#                 loss_fn = model.loss_fn
#             # 双任务
#             if dual_task:
#                 cls_loss = loss_fn[0](logits, label[:,0].long())
#                 # 若score的label缺失(即label[:,1]是nan),则计算的得到的score_loss也是nan
#                 # 经过后向传播之后,网络各层的参数也变成了nan,之后的网路输出值将一直是nan.
#                 if not torch.isnan(label[:,1]):
#                     score_loss = loss_fn[1](score, label[:,1].float())
#                     if loss is None:
#                         loss = cls_loss+score_loss
#                     else:
#                         loss = loss+cls_loss+score_loss
#                     score_labels += [label[:,1].detach().item()]
#                     scores += [score.detach().cpu().numpy()]
#                 else:
#                     if loss is None:
#                         loss = cls_loss
#                     else:
#                         loss = loss+cls_loss
#                 cls_labels[batch_idx] = label[:,0].detach().item()
#                 cls_logger.log(Y_hat, label[:,0])
#             # 单任务
#             else:
#                 if isinstance(loss_fn,tuple):
#                     cls_loss = loss_fn[0](logits, label.long())
#                 else:
#                     cls_loss = loss_fn(logits, label.long())
#                 if loss is None:
#                     loss = cls_loss
#                 else:
#                     loss = loss+cls_loss
#                 cls_labels[batch_idx] = label.detach().item()
#                 cls_logger.log(Y_hat, label)    
#             cls_probs[batch_idx] = Y_prob.detach().cpu().numpy()
        
        
#             cls_loss_value = cls_loss.item()  
#             cls_train_loss += cls_loss_value
#             if dual_task and not torch.isnan(label[:,1]):
#                 score_loss_value = score_loss.item()
#                 score_train_loss += score_loss_value
                
#         else:
            
#             # backward pass
#             loss.backward()
#             # step
#             optimizer.step()
#             optimizer.zero_grad()
            
#             if dual_task:
#                 if multi_modal:
#                     print('batch {}, cls loss: {:.4f}, cls label: {}, score label: {}, bag_size: {}'.format(batch_idx+1, cls_loss_value, label[0,0], label[0,1], [v.size(0) for _,v in data.items()]) )
#                 else:
#                     print('batch {}, cls loss: {:.4f}, cls label: {}, score label: {}, bag_size: {}'.format(batch_idx+1, cls_loss_value, label[0,0], label[0,1], data.size(0)) )
#             else:
#                 if multi_modal:
#                     print('batch {}, cls loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx+1, cls_loss_value, label.item(), [v.size(0) for _,v in data.items()]) )
#                 else:
#                     print('batch {}, cls loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx+1, cls_loss_value, label.item(), data.size(0)) )

#     score_labels = np.array(score_labels)
#     scores = np.array(scores)
    
#     # calculate loss for epoch
#     cls_train_loss /= len(loader)
#     if dual_task:
#         score_train_loss /= score_labels.size 
#         train_loss = cls_train_loss+score_train_loss
#     else:
#         train_loss = cls_train_loss
    
#     #######################################################################################
#     ## 计算auc 和 acc (和Corr)
#     #######################################################################################
#     if n_classes == 2:
#         cls_auc = roc_auc_score(cls_labels, cls_probs[:,1])
#         cls_aucs = []
#     else:
#         cls_aucs = []
#         binary_labels = label_binarize(cls_labels, classes=[i for i in range(n_classes)])
#         for class_idx in range(n_classes):
#             if class_idx in cls_labels:
#                 fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], cls_probs[:, class_idx])
#                 cls_aucs.append(calc_auc(fpr, tpr))
#             else:
#                 cls_aucs.append(float('nan'))

#         cls_auc = np.nanmean(np.array(cls_aucs))
#     cls_acc = metrics.accuracy_score(cls_labels, np.argmax(cls_probs,-1)) 
#     if dual_task:
#         score_coor = spearmanr(score_labels,scores)[0]
#     #######################################################################################   
#     if dual_task:
#         print('Epoch: {}, cls train_loss: {:.4f}, cls acc: {:.4f}, cls auc: {:.4f}, score_train_loss: {:.4f}, score Corr: {:.4f}'.format(epoch, cls_train_loss, cls_acc, cls_auc, score_train_loss, score_coor))
#     else:
#         print('Epoch: {}, cls train_loss: {:.4f}, cls acc: {:.4f}, cls auc: {:.4f}'.format(epoch, cls_train_loss, cls_acc, cls_auc))
    
#     for i in range(n_classes):
#         tpr, correct, count = cls_logger.get_summary(i)
#         print('class {}: accuracy {}, correct {}/{}'.format(i, tpr, correct, count))
#         if writer:
#             if tpr is None:
#                 acc = 0.0
#             writer.add_scalar('train/class_{}_tpr'.format(i), tpr, epoch)

#     if writer:
#         writer.add_scalar('train/loss', loss, epoch)
#         writer.add_scalar('train/cls_train_loss', cls_train_loss, epoch)
#         writer.add_scalar('train/cls_acc', cls_acc, epoch)
#         writer.add_scalar('train/cls_auc', cls_auc, epoch)
#         if dual_task:
#             writer.add_scalar('train/score_train_loss', score_train_loss, epoch)
#             writer.add_scalar('train/score_coor', score_coor, epoch)

# +
def validate(
    cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, 
    loss_fn = None, results_dir=None, multi_modal=False, dual_task=False, mix=False):
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    cls_logger = Accuracy_Logger(n_classes=n_classes)
    cls_val_loss = 0.
    score_val_loss = 0.
    
    cls_probs = np.zeros((len(loader), n_classes))
    cls_labels = np.zeros(len(loader))
    scores = []
    score_labels = []

    with torch.no_grad():
        for batch_idx, sample in enumerate(loader):
            data, label = sample[0], sample[1]
            if mix:
                assert len(sample)==3, 'sample must contains clinical information'
                clin = sample[2]
                clin = clin.long().to(device)
            if multi_modal:
                if isinstance(data,tuple):
                    data = tuple([img.to(device) for img in data])
                elif isinstance(data,dict):
                    for modal_name in data.keys():
                        data[modal_name] = data[modal_name].to(device)
            else:
                data =  data.to(device)
            
            if not dual_task:
                label = label[:,0].long()
            label = label.to(device)

            if mix:
                output = model( (data,clin) )
            else:
                output = model(data)
            
            if dual_task:
                logits = output[0]
                score = output[1].squeeze(1)
            else:
                logits = output

            Y_prob = torch.softmax(logits,-1)
            Y_hat = torch.argmax(Y_prob)
            del output

#             ##################################################################
#             cls_logger.log(Y_hat, label)
            
#             cls_loss =  loss_fn(logits, label) 
#             loss = cls_loss
#             cls_loss_value = cls_loss.item()

#             cls_probs[batch_idx] = Y_prob.cpu().numpy()
#             cls_labels[batch_idx] = label.item()
            
#             cls_val_loss += cls_loss_value
#             ##################################################################
            
            ##################################################################
            if loss_fn is None:
                loss_fn = model.loss_fn
            if dual_task:
                cls_loss = loss_fn[0](logits, label[:,0].long())
                # 若score的label缺失(即label[:,1]是nan),则计算的得到的score_loss也是nan
                # 经过后向传播之后,网络各层的参数也变成了nan,之后的网路输出值将一直是nan.
                if not torch.isnan(label[:,1]):
                    score_loss = loss_fn[1](score, label[:,1].float())
                    loss = cls_loss+score_loss
                    score_labels += [label[:,1].detach().item()]
                    scores += [score.detach().cpu().numpy()]
                else:
                    loss = cls_loss
                cls_labels[batch_idx] = label[:,0].detach().item()
                cls_logger.log(Y_hat, label[:,0])
            else:
                if isinstance(loss_fn,tuple):
                    cls_loss = loss_fn[0](logits, label.long())
                else:
                    cls_loss = loss_fn(logits, label.long())
                loss = cls_loss
                cls_labels[batch_idx] = label.detach().item()
                cls_logger.log(Y_hat, label)    
            cls_probs[batch_idx] = Y_prob.detach().cpu().numpy()


            cls_loss_value = cls_loss.item()  
            cls_val_loss += cls_loss_value
            if dual_task and not torch.isnan(label[:,1]):
                score_loss_value = score_loss.item()
                score_val_loss += score_loss_value
            ##################################################################
            
    score_labels = np.array(score_labels)
    scores = np.array(scores)
    
    # calculate loss for epoch
    cls_val_loss /= len(loader)
    if dual_task:
        score_val_loss /= score_labels.size
        val_loss = cls_val_loss+score_val_loss
    else:
        val_loss = cls_val_loss
        

    # 计算auc
    if n_classes == 2:
        cls_auc = roc_auc_score(cls_labels, cls_probs[:, 1])
        cls_aucs = []
    else:
        cls_aucs = []
        binary_labels = label_binarize(cls_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in cls_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], cls_probs[:, class_idx])
                cls_aucs.append(calc_auc(fpr, tpr))
            else:
                cls_aucs.append(float('nan'))
        cls_auc = np.nanmean(np.array(cls_aucs))
        
    cls_acc = metrics.accuracy_score(cls_labels, np.argmax(cls_probs,-1)) 
    if dual_task:
        score_coor = spearmanr(score_labels,scores)[0]
    if writer:
        writer.add_scalar('val/val_loss', val_loss, epoch)
        writer.add_scalar('val/cls_val_loss', cls_val_loss, epoch)
        writer.add_scalar('val/cls_auc', cls_auc, epoch)
        writer.add_scalar('val/cls_acc', cls_acc, epoch)
        if dual_task:
            writer.add_scalar('val/score_loss', score_val_loss, epoch)
            writer.add_scalar('val/score_coor', score_coor, epoch)

    if dual_task:
        print( '\nVal Set, cls val_loss: {:.4f}, cls acc: {:.4f}, cls auc: {:.4f}, score_val_loss: {:.4f}, score coor: {:.4f}'.format(cls_val_loss, cls_acc, cls_auc, score_val_loss, score_coor))
    else:
        print( '\nVal Set, cls val_loss: {:.4f}, cls acc: {:.4f}, cls auc: {:.4f}'.format(cls_val_loss, cls_acc, cls_auc))
    
    for i in range(n_classes):
        tpr, correct, count = cls_logger.get_summary(i)
        print('class {}: tpr {}, correct {}/{}'.format(i, tpr, correct, count))
        if writer:
            if tpr is None:
                tpr = 0.0
            writer.add_scalar('val/class_{}_tpr'.format(i), tpr, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


# -

def summary(model, loader, n_classes, multi_modal=False, dual_task=False, mix=False):
    """
    仅适用于单任务.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    cls_test_error = 0.
    cls_test_loss = 0.

    cls_probs = np.zeros((len(loader), n_classes))
    cls_labels = np.zeros(len(loader))
    
    score_labels = []
    scores = []

    patient_IDs = loader.dataset.patient_IDs
    patient_results = {}

    for batch_idx, sample in enumerate(loader):
        data, label = sample[0], sample[1]
        if mix:
            assert len(sample)==3, 'sample must contain clinical information'
            clin = sample[2]
            clin = clin.long().to(device)
        if multi_modal:
            if isinstance(data,tuple):
                data = tuple([img.to(device) for img in data])
            elif isinstance(data,dict):
                for modal_name in data.keys():
                    data[modal_name] = data[modal_name].to(device)
        else:
            data =  data.to(device)
        if not dual_task:
            label = label[:,0].long()

        patient_id = patient_IDs[batch_idx]
        with torch.no_grad():
            if mix:
                output = model( (data,clin) )
            else:
                output = model(data)
                
        if dual_task:
            logits = output[0]
            score = output[1].squeeze(1)
        else:
            logits = output
            
                
        Y_prob = torch.softmax(logits,-1)
        Y_hat = torch.argmax(Y_prob)
        del output

        
        cls_prob = Y_prob.cpu().numpy()
        cls_probs[batch_idx] = cls_prob
        if dual_task:
            cls_label = label[:,0]
            score_label = label[:,1]
            if not torch.isnan(score_label):
                score_labels += [score_label.detach().item()]
                scores += [score.detach().cpu().numpy()]
        else:
            cls_label = label
        cls_logger.log(Y_hat, cls_label)
        cls_labels[batch_idx] = cls_label.item()
        
        if dual_task:
            patient_results.update({patient_id: {'patient_id': np.array(patient_id), 'cls_label': cls_label.detach().item(), 'cls_prob': cls_prob, 'score label':score_label.detach().item(), 'score':score.detach().item()}})
        else:
            patient_results.update({patient_id: {'patient_id': np.array(patient_id), 'cls_label': cls_label.detach().item(), 'cls_prob': cls_prob}})

    if dual_task:
        score_labels = np.array(score_labels)
        scores = np.array(scores)
        score_coor = spearmanr(score_labels,scores)[0]
    
    cls_acc = metrics.accuracy_score(cls_labels, np.argmax(cls_probs,-1)) 
    if n_classes == 2:
        cls_auc = roc_auc_score(cls_labels, cls_probs[:, 1])
    else:
        cls_auc = roc_auc_score(cls_labels, cls_probs, multi_class='ovr')
    
    if dual_task:
        return patient_results, cls_acc, cls_auc, score_coor, (cls_logger,)
    else:
        return patient_results, cls_acc, cls_auc, (cls_logger,)
