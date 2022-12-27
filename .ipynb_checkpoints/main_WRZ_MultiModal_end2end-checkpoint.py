#!/usr/bin/env python
# coding: utf-8
# %%

# %%


# 端到端的多示例学习,从MRI图像特征提取,到分类预测结果.


# %%


import argparse
import pdb
import os
import math
import json

import pandas as pd
import numpy as np
from glob import glob
import copy

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F


# %%

from models.model_toad import *
from datasets.WRZ_DataSet import *
from utils.core_utils import *
from utils.file_utils import save_pkl, load_pkl


# %%

def seed_torch(seed=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# %%


def main(args):
    args.multi_modal = True
    
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    all_cls_test_auc = []
    all_cls_val_auc = []
    all_cls_test_acc = []
    all_cls_val_acc = []
    
    all_site_test_auc = []
    all_site_val_auc = []
    all_site_test_acc = []
    all_site_val_acc = []

    # 获取真各个数据集的文件全名列表
    args.pats = json.loads(args.pats)
    filepath_df = pd.DataFrame(columns=args.modal_names)
    for modal,pat in args.pats.items():
        filepaths = glob(pat)
        patient_IDs = [filepath.split(os.path.sep)[-2] for filepath in filepaths]
        for patient_ID,filepath in zip(patient_IDs,filepaths):
            filepath_df.loc[patient_ID,modal] = filepath
    patient_IDs = list(filepath_df.index)# 为dataset做准备
    
    #######################################################################
    # 检查三个序列的文件是否齐全，若不齐全，则删除该病例
    #filepath_df = filepath_df[ filepath_df.isna().transpose().apply(lambda xs: np.sum(xs)==0 ) ]
    #######################################################################
#     #######################################################################
#     filepath_df.drop(columns=['T1','T2'],inplace=True)
#     #######################################################################

    # 转为列表,列表的的元素为字典.
    # 对于任意一个病例,其字典包含且仅包含该病例所具有的序列, 且一定得包含SWAN序列
    # 检查是否有SWAN序列的文件，若无，则删除该病例
    filepath_df = filepath_df[ filepath_df['SWAN'].isna()==False ]
    filepaths_list = [dict(zip(filepath_df.columns,filepaths)) for filepaths in filepath_df.values.tolist()]
    for i in range(len(filepaths_list)):
        for modal_name in filepath_df.columns:
            if isinstance(filepaths_list[i][modal_name],str):
                continue
            if np.isnan(filepaths_list[i][modal_name]):
                filepaths_list[i].pop(modal_name)
    
    # 获取对应的标签,从而可以分层k折交叉验证
    label_list = []
    for filepaths in filepaths_list:
        for cls_name, cls in args.clses_name.items():
            if cls_name in list(filepaths.values())[0]:
                label_list += [cls]
                break
    
    # 分层K折交叉验证
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=args.fold, shuffle=True, random_state=args.seed)
    for i,(train_index, test_index) in enumerate(skf.split(X=filepaths_list, y=label_list)):
        seed_torch(args.seed)
        # 数据集划分
        # X:
        train_filepaths_list = [ filepaths_list[idx] for idx in train_index ]
        test_filepaths_list = [ filepaths_list[idx] for idx in test_index ]
        # y:
        train_label_list = [ label_list[idx] for idx in train_index ]
        test_label_list = [ label_list[idx] for idx in test_index ]
        # ID
        train_patient_IDs = [ patient_IDs[idx] for idx in train_index ]
        test_patient_IDs = [ patient_IDs[idx] for idx in test_index ]
        # dataset object
        train_dataset = WRZ_MultiModal_Dataset( # WRZ_MultiModal_Dataset # WRZ_Single_Dataset
            data_dir='/raid/huaqing/tyler/WRZ/data/data_V2_nii_gz', 
            filepaths_list=train_filepaths_list, 
            label_list=train_label_list,
            patient_IDs=train_patient_IDs,
            clses_name=args.clses_name,
            transform=transforms.Compose([
                ToTensor(),
                CropOuterFrame(),
                RandomDropInstance(p=0.2),
                ScaleVoxelValue(),
                AddChannelDim(),
                RepeatChannelDim(3),
            ]),
            modal_dropout = {'T1':0.2, 'T2':0.2}
        )
        
        test_dataset = WRZ_MultiModal_Dataset( # WRZ_MultiModal_Dataset # WRZ_Single_Dataset
            data_dir='/raid/huaqing/tyler/WRZ/data/data_V2_nii_gz', 
            filepaths_list=test_filepaths_list,
            patient_IDs=test_patient_IDs,
            label_list=test_label_list,
            clses_name=args.clses_name,
            transform=transforms.Compose([
                ToTensor(),
                CropOuterFrame(),
                ScaleVoxelValue(),
                AddChannelDim(),
                RepeatChannelDim(3),
            ]),
            modal_dropout = None,
        )  
        val_dataset = copy.deepcopy(test_dataset)
        
        print('training: {}, validation: {}, testing: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))
        datasets = (train_dataset, val_dataset, test_dataset)
        
        # define network
        model = MultimodalResTOAD(modal_names=args.modal_names, task_classes=args.n_classes, dropout=True, requires_grad=False).to(device)
#         #################################################
#         # 调试用
#         model = ResTOAD(args.n_classes,dropout=True,requires_grad=False).to(device)
#         #################################################
        
        # 训练
        results, cls_test_acc, cls_val_acc, cls_test_auc, cls_val_auc = train(model, datasets, i, args)
        all_cls_test_auc.append(cls_test_auc)
        all_cls_val_auc.append(cls_val_auc)
        all_cls_test_acc.append(cls_test_acc)
        all_cls_val_acc.append(cls_val_acc)
        
        # write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({
        'folds': np.arange(args.fold), 
        'cls_test_acc': all_cls_test_acc,
        'cls_val_acc' : all_cls_val_acc,
        'cls_test_auc': all_cls_test_auc, 
        'cls_val_auc': all_cls_val_auc, })


    save_name = 'summary.csv'
    save_filepath = os.path.join(args.results_dir, save_name)
    final_df.to_csv( save_filepath )
    print( 'Results of {}-fold CV successfully saved to {}.'.format(args.fold, save_filepath) )


# %%


# Training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', default=None, type=str, help='data directory')

# data_V2:
parser.add_argument('--pats', type=str, 
                    default='{"SWAN": "../../data/data_V2_nii_gz/*/*/*swan.nii.gz",\
                             "T1": "../../data/data_V2_nii_gz/*/*/*t1 flair transverse.nii.gz",\
                             "T2": "../../data/data_V2_nii_gz/*/*/*t2 flair transverse.nii.gz"}',
                    help='pattern to glob all sample files of the dataset')
parser.add_argument('--modal_names', type=str, default=["SWAN", "T1", "T2"],help='MRI series names')

parser.add_argument('--clses_name', type=str, default='{"ACSVD":0, "CADASIL":1, "CAA":2}',help='patient class-name and its label number')
parser.add_argument('--max_epoch', type=int, default=200, help='maximum number of epochs to train (default: 200)')
parser.add_argument('--patience', type=int, default=20, help='maximum number of epochs to train (default: 20)')
parser.add_argument('--stop_epoch', type=int, default=100, help='maximum number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
parser.add_argument('--reg', type=float, default=1e-5, help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--fold', type=int, default=10, help='number of folds (default: 1)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')

parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--exp_code', type=str, default='multi_modal_3category_data_V2', help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--gpu', type=str,  default='6,7', help='GPU to use')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%


if __name__ == "__main__":
    seed_torch(seed=args.seed)
    
    args.clses_name = json.loads(args.clses_name)
    args.n_classes = len(set(args.clses_name.values()))
    args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)
    
    with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
        print(args, file=f)
    f.close()

    results = main(args)
    print("finished!")
    print("end script")


# %%
