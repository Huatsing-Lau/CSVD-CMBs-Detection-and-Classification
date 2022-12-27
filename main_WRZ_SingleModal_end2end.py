#!/usr/bin/env python
# coding: utf-8
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

import SimpleITK as sitk

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
    print(args)
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

#     # 获取整个数据集的文件全名列表
#     filepath_list = []
#     for pat in args.pat:
#         filepath_list += glob(pat)

#     # 获取对应的标签,从而可以分层k折交叉验证
#     label_list = []
#     for filepath in filepath_list:
#         for cls_name, cls in args.clses_name.items():
#             if cls_name in filepath:
#                 label_list += [cls]
#                 break
#     patient_IDs = list(filepath_df.index)# 为dataset做准备
    
    
    # 获取整个数据集的文件全名列表
    filepath_df = pd.DataFrame(columns=['filepath'])
    for pat in args.pats:
        filepaths = glob(pat)
        patient_IDs = [os.path.basename(filepath)+'--'+filepath.split(os.path.sep)[-4].split('_')[0]+'_'+filepath.split(os.path.sep)[-2] for filepath in filepaths]
        # [filepath.split(os.path.sep)[-2] for filepath in filepaths]
        for patient_ID,filepath in zip(patient_IDs,filepaths):
            filepath_df.loc[patient_ID,'filepath'] = filepath
    patient_IDs = list(filepath_df.index)# 为dataset做准备
    filepath_list = filepath_df.loc[:,'filepath']
    
    # 临床信息(clin_df的index和patient_ID对应)
    clin_df = get_clinical_df(args.clinical_filepath)
    clin_list = []
    for patient_ID in patient_IDs:
        gender = clin_df.loc[patient_ID.split('--')[1],'gender']
        age = clin_df.loc[patient_ID.split('--')[1],'age']
        clin_list += [{'gender':gender,'age':age}]
        
    # 获取对应的标签,从而可以分层k折交叉验证
    # 这是疾病的3分类标签
    label_list = []
    for patient_ID in patient_IDs:
        for cls_name, cls in args.clses_name.items():
            if cls_name in patient_ID:
                label_list += [{'class':cls}]
                break
                
    
    cls_list = [item['class'] for item in label_list]
    for x,y in zip(filepath_list, cls_list):
        print(x,y)
    # 分层K折交叉验证
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=args.fold, shuffle=True, random_state=args.seed)
    for i,(train_index, test_index) in enumerate(skf.split(X=filepath_list, y=cls_list)):
        # 数据集划分
        # X:
        train_filepath_list = [ filepath_list[idx] for idx in train_index ]
        test_filepath_list = [ filepath_list[idx] for idx in test_index ]
        # clinical
        train_clin_list = [ filepath_list[idx] for idx in train_index ]
        test_clin_list = [ filepath_list[idx] for idx in test_index ]
        # y:
        train_label_list = [ label_list[idx] for idx in train_index ]
        test_label_list = [ label_list[idx] for idx in test_index ]
        
        # fixed image filepath
        fixedFilepath_list = copy.deepcopy(train_filepath_list)
        
#         ####################################################################
#         fixedFilepath_list = []
#         for pat in [ '../../data/external_center_data/SDFY_data/*/*/swan.nii.gz']:
#             fixedFilepath_list += glob(pat)
#         ####################################################################
        
        # dataset object
        train_dataset = WRZ_Single_Dataset(
            data_dir=None,#'/raid/huaqing/tyler/WRZ/data/data_V2_nii_gz', 
            filepath_list=train_filepath_list, 
            label_list=train_label_list,
            clin_list=train_clin_list,
            fixedFilepaths=fixedFilepath_list,
            clses_name=args.clses_name,
            transform=transforms.Compose([
                # SITKCropOuterFrame( 5 ),
                SITKNormalize(),
                SITKRandomCrop( 0.75,1.0 ),
                SITKRandomResample( (0.8,0.8,3.0),(1.2,1.2,4.0) ),#(1.0,1.0,2.0),(1.0,1.0,3.0) #(0.4,0.4,2.0),(0.6,0.6,3.0)# ( (0.4,0.4,3.0),(0.6,0.6,6.0) )
                SITKAdaptiveHistEqual(),
                SITKtoNumpy(),
                ToTensor(),
                RandomFilp(prob=0.5,dims=[1,2]),
                #ScaleVoxelValue(quantile=0.995),
                RandomDropInstance(p=0.2,sort='ascending'),
                AddChannelDim(),
#                 RepeatChannelDim(3),
#                 Normalize(mean=[125.3, 123.0, 113.9], std=[63.0, 62.1, 66.7]),
            ])
        )
        
        test_dataset = WRZ_Single_Dataset(
            data_dir=None,#'/raid/huaqing/tyler/WRZ/data/data_V2_nii_gz', 
            filepath_list=test_filepath_list, 
            label_list=test_label_list,
            clin_list=test_clin_list,
#             fixedFilepaths=fixedFilepath_list,
            clses_name=args.clses_name,
            transform=transforms.Compose([
                # SITKCropOuterFrame( 5 ),
                SITKNormalize(),
                SITKResample( (1.0,1.0,3.5) ),# (1.0,1.0,2.5) #(0.5,0.5,2.5)#SITKResample( (0.5,0.5,4.0) ),
                SITKAdaptiveHistEqual(),
                SITKtoNumpy(),
                ToTensor(),
                #ScaleVoxelValue(quantile=0.995),
                AddChannelDim(),
#                 RepeatChannelDim(3),
#                 Normalize(mean=[125.3, 123.0, 113.9], std=[63.0, 62.1, 66.7]),
            ])
        )
            
        val_dataset = copy.deepcopy(test_dataset)
        
        print('training: {}, validation: {}, testing: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))
        datasets = (train_dataset, val_dataset, test_dataset)
        
        # define network
        if args.trained_model_filepath is not None:
            # 加载模型
            model = torch.load(args.trained_model_filepath)
            from torch.nn.parallel.data_parallel import DataParallel
            if isinstance(model,DataParallel):
                model = net.module
            for child in model.children():
                for param in child.parameters():
                    param.requires_grad = True      
            # 冻结backbone
            # 检验方法： [item.requires_grad for item in model.resnet_baseline.parameters()]
            if not args.backbone_requires_grad:
                for param in model.resnet_baseline.parameters():
                    param.requires_grad = False
            # 修改最后一层
            model.classifier = nn.Linear(in_features=512, out_features=args.n_classes, bias=True)
            model.to(args.device)
        else:
            model = ResTOAD(
                args.n_classes,
                dropout=True,
                requires_grad=args.backbone_requires_grad,#False,
                loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(args.class_weights).to(args.device)),# 损失函数
                input_channel=args.input_channel,
            ).to(args.device)
        
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
def get_clinical_df(filepaths):
    """
    获取临床信息，返回dataframe。
    输入：
        filepaths：dict, key是站点名称，value是excel文件名称。
    """
        
    def get_one_sit_cinical_df(site,filepath):
        """
        获取一个站点（医院）的临床信息。返回dataframe。
        备注：一个站点对应一个excel文件。
        """
        df = pd.ExcelFile(filepath)
        df_list = []
        for sheet in df.sheet_names:
            tmp_df = pd.read_excel(df, sheet)
            tmp_df.insert(loc=0,column='cls',value=[sheet]*len(tmp_df))
            tmp_df['id'] = tmp_df['id'].apply(lambda x: '{}_{}_{:03d}'.format(site,sheet,x))
            df_list += [tmp_df]
        clin_df = pd.concat(df_list,axis=0)
        # 将所有sheet中数据合并到一个df中
        clin_df.reset_index(drop=True,inplace=True)
        # 添加站点列
        clin_df.insert(loc=0,column='site',value=[site]*len(clin_df))
        return clin_df

    clin_df = pd.concat([get_one_sit_cinical_df(site,filepath) for site, filepath in filepaths.items()],axis=0)
    clin_df.set_index(keys='id',inplace=True)
    return clin_df

# clin_df = get_clinical_df(args.clinical_filepath)
# display(clin_df)
# clin_df['age'].hist(bins=20)

# %%


# Training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', default=None, type=str, help='data directory')

# data_V2:
parser.add_argument('--pats', type=list, 
                    default=[ '../../data/internal_center_data/ZSSY_data_after/*/*/*swan.nii.gz', '../../data/external_center_data/MMSY_data_after/*/*/*swan.nii.gz' ],
                    # [ '../../data/internal_center_data/ZSSY_data_after/*/*/*swan.nii.gz' ],
                    # [ '../../data/internal_center_data/ZSSY_data_after/*/*/*swan.nii.gz', '../../data/external_center_data/MMSY_data_after/*/*/*swan.nii.gz' ],
                    # [  '../../data/data_V2_nii_gz_before/*/*/*swan.nii.gz', '../../data/external_center_data/MMSY_data_before/*/*/swan.nii.gz' ],
                    #[ '../../data/external_center_data/MMSY_data/*/*/swan.nii.gz' ],
                    #[  '../../data/data_V2_nii_gz/*/*/*swan.nii.gz', '../../data/external_center_data/MMSY_data/*/*/swan.nii.gz' ],
                    #[  '../../data/data_V2_nii_gz/*/*/*swan.nii.gz', '../../data/external_center_data/SDFY_data/*/*/swan.nii.gz' ],
                    help='pattern to glob all sample files of the dataset')

parser.add_argument('--clinical_filepath',type=str,
                   default='{"ZSSY":"../../data/ZSSY_clinical_data.xlsx", "MMSY":"../../data/MMSY_clinical_data.xlsx", "SDFY":"../../data/SDFY_clinical_data.xlsx"}',
                    help='filepath of the clinical dataset')

parser.add_argument('--clses_name', type=str, default='{"ACSVD":0, "CADASIL":1, "CAA":2}', help='patient class-name and its label number')

parser.add_argument('--trained_model_filepath', type=str, default=None, help='filepath of the trained model for transfer-learning.')

parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--max_epoch', type=int, default=100, help='maximum number of epochs to train (default: 200)')
parser.add_argument('--patience', type=int, default=10, help='maximum number of epochs to train (default: 20)')
parser.add_argument('--stop_epoch', type=int, default=40, help='minimum number of epochs to train before early_stopping (default: 50)')

parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
parser.add_argument('--reg', type=float, default=1e-5, help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--fold', type=int, default=10, help='number of folds (default: 1)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')

parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')

parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--exp_code', type=str, default='single_serie_swan_3category_data_V2', help='experiment code for saving results')
parser.add_argument('--class_weights', type=list, default=[1.0,5.0,1.0], help='a manual rescaling weight given to each class.')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--backbone_requires_grad', action='store_true', default=False, help='whether to train the backbone')
parser.add_argument('--input_channel', type=int, default=3, help='number of input image channels')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
args = parser.parse_args()#args=[]


# %%


if __name__ == "__main__":
    seed_torch(seed=args.seed)
    
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.multi_modal = False
    args.dual_task = False
    
    args.clinical_filepath = json.loads(args.clinical_filepath)
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

