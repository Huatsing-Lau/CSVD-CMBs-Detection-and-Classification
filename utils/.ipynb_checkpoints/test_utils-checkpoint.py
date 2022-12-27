import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
# for font in sorted( font_manager.fontManager.ttflist, key=lambda font: font.name ):
#     # 查看字体名以及对应的字体文件名
#     print(font.name, '-', font.fname) 
plt.rcParams['font.sans-serif'] = 'Times New Roman'

from PIL import Image

import SimpleITK as sitk

import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.model_toad import *
from datasets.WRZ_DataSet_withMask import *

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import classification_report

from copy import deepcopy
import itertools as it


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
            tmp_df['ID'] = tmp_df['ID'].apply(lambda x: '{}_{}_{:03d}'.format(site,sheet,x))
            df_list += [tmp_df]
        clin_df = pd.concat(df_list,axis=0)
        # 将所有sheet中数据合并到一个df中
        clin_df.reset_index(drop=True,inplace=True)
        # 添加站点列
        clin_df.insert(loc=0,column='site',value=[site]*len(clin_df))
        return clin_df

    clin_df = pd.concat([get_one_sit_cinical_df(site,filepath) for site, filepath in filepaths.items()],axis=0)
    clin_df.set_index(keys='ID',inplace=True)
    return clin_df


def predict(model,test_loader,mix=False,device='cpu'):
    multi_modal, dual_task = False, False
    
    cls_probs = None
    cls_labels = np.zeros(test_loader.dataset.__len__())

    with torch.no_grad():
        for batch_idx, sample in tqdm(enumerate(test_loader)):
            data, label = sample[0], sample[1]
            if mix:
                clin = sample[2]
                clin = clin.float().to(device)
            if multi_modal:
                if isinstance(data,tuple):
                    data = tuple([img.to(device) for img in data])
                elif isinstance(data,dict):
                    for modal_name in data.keys():
                        data[modal_name] = data[modal_name].to(device)
            else:
                data =  data.to(device)
            label = label.to(device)

            # 预测
            if mix:
                input_tensor = ( data, clin ) # a tuple
                output = model(input_tensor) # 作为一个tuple或者自定义一个输入类（类似字典）输入是最合适的，
            else:
                output = model(data)
            del data
            if dual_task:
                logits = output[0]
                score = output[1].squeeze(1)
            else:
                logits = output

            y_prob = torch.softmax(logits,-1)
            y_hat = torch.argmax(y_prob)

            if cls_probs is None:
                cls_probs = np.zeros((test_loader.dataset.__len__(), y_prob.shape[1]))
            cls_probs[batch_idx] = y_prob.detach().cpu().numpy()
            cls_labels[batch_idx] = label[:,0].detach().cpu().numpy()
    return cls_labels, cls_probs

def get_roc(y_true,y_score,n_classes):
    """
    y_true: onehot
    y_score proba
    """
    from numpy import interp
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = np.round( auc(fpr[i], tpr[i]), 4 )

    # micro
    # micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = np.round( auc(fpr["micro"], tpr["micro"]), 4 )

    # macro
    # macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = np.round( auc(fpr["macro"], tpr["macro"]), 4 )
    
    # weighted
    cls_weights = y_true.sum(axis=0)/y_true.sum()
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i, weight in enumerate(cls_weights):
        mean_tpr += weight * interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    fpr["weighted"] = all_fpr
    tpr["weighted"] = mean_tpr
    roc_auc["weighted"] = np.round( auc(fpr["weighted"], tpr["weighted"]), 4 )
    return fpr, tpr, roc_auc


def bootstrap_score(func, y_true, y_pred, classes=None, bootstraps = 100, fold_size = 100):
    '''
    boottrap的方式计算指标，返回一个数组。
    输入：
        func: callable score function
        y_true不是onehot
        bootstraps: 统计多少次
        fold_size 每次随机采样多少个样本
    '''
    

    if classes is None:
        # 例如accuracy这类与类别无关的指标
        statistics = np.zeros((1,bootstraps))
        for i in range(bootstraps):
            df = pd.DataFrame(columns=['y_true', 'y_pred'])
            df.loc[:, 'y_true'] = y_true
            df.loc[:, 'y_pred'] = y_pred
            # 直接随机采样            
            sample_df = df.sample(n = fold_size, replace=True)
            y_true_sample = sample_df.y_true.values
            y_pred_sample = sample_df.y_pred.values
            score = func(y_true_sample, y_pred_sample)
            statistics[0][i] = score
    else:
        # 例如特异度\敏感度这类与类别相关的指标
        statistics = np.zeros((len(classes), bootstraps))
        for k,c in enumerate(classes):
            if c in ['micro','macro']:
                df = pd.DataFrame(columns=['y_true', 'y_pred'])
                df.loc[:, 'y_true'] = y_true
                df.loc[:, 'y_pred'] = y_pred
                for i in range(bootstraps):
                    # 直接随机采样            
                    sample_df = df.sample(n = fold_size, replace=True)
                    y_true_sample = sample_df.y_true.values
                    y_pred_sample = sample_df.y_pred.values
                    score = func(y_true_sample, y_pred_sample, average=c)
                    statistics[k][i] = score
            else:
                df = pd.DataFrame(columns=['y_true', 'y_pred'])
                df.loc[:, 'y_true'] = y_true==c
                df.loc[:, 'y_pred'] = y_pred==c
                df_pos = df[df.y_true==True]
                df_neg = df[df.y_true==False]
                prevalence = len(df_pos) / len(df)
                for i in range(bootstraps):
                    # 类别等比随机采样
                    pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
                    neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)
                    y_true_sample = np.concatenate([pos_sample.y_true.values, neg_sample.y_true.values])
                    y_pred_sample = np.concatenate([pos_sample.y_pred.values, neg_sample.y_pred.values])
        #             # 直接随机采样            
        #             sample_df = df.sample(n = fold_size, replace=True)
        #             y_true_sample = sample_df.y_true.values
        #             y_pred_sample = sample_df.y_pred.values

                    score = func(y_true_sample, y_pred_sample)
                    statistics[k][i] = score
    return statistics


# +
def micro_roc_auc_score(y_true, y_score, n_classes=None):
    """
    y_true:
    y_proba: 
    n_classes: int
    """
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder()
    if n_classes is None:
        n_classes = y_score.shape[1]
    # 计算每一类的ROC
    y_true_onehot = enc.fit_transform(y_true[:,np.newaxis]).toarray()
    fpr, tpr, _ = roc_curve(y_true_onehot.ravel(), y_score.ravel())
    micro_roc_auc = np.round( auc(fpr, tpr), 4 )
    return micro_roc_auc


def bootstrap_AUC_score(y_true, y_prob, bootstraps = 100, fold_size = 100):
    '''
    boottrap的方式计算指标，返回一个数组。
    输入：
        func: callable score function
        y_true: np.array, 不是onehot
        y_prob: np.array, 预测概率
        bootstraps: 统计多少次
        fold_size 每次随机采样多少个样本
    '''
    from sklearn.metrics import accuracy_score
    n_classes = y_prob.shape[1]
    auc_df = pd.DataFrame( columns=list(range(n_classes))+['micro','macro','weighted'], index=range(bootstraps), data=0 )
    for i in range(bootstraps):
        
        # 类别等比随机采样
        df = pd.DataFrame(data=y_true, columns=['y_true']).groupby('y_true').apply(lambda x: x.sample(n=int(fold_size*len(x)/y_true.shape[0]), replace=True))
        index = [ index[1] for index in df.index]
        y_true_sample = y_true[index]
        y_prob_sample = y_prob[index]

        
        for cls in range(n_classes):
            auc_df.loc[i,cls] = roc_auc_score(y_true=y_true_sample==cls, y_score=y_prob_sample[:,cls])
        auc_df.loc[i,'micro'] = micro_roc_auc_score(y_true=y_true_sample, y_score=y_prob_sample)
        auc_df.loc[i,'macro'] = roc_auc_score(y_true=y_true_sample, y_score=y_prob_sample, average='macro', multi_class='ovr')
        auc_df.loc[i,'weighted'] = roc_auc_score(y_true=y_true_sample, y_score=y_prob_sample, average='weighted', multi_class='ovr')
    return auc_df


# -

def bootstrap_classification_report(y_true, y_pred, bootstraps = 100, fold_size = 100):
    '''
    boottrap的方式计算指标，返回一个数组。
    输入：
        func: callable score function
        y_true: np.array, 不是onehot
        y_pred: np.array, 预测类别
        bootstraps: 统计多少次
        fold_size 每次随机采样多少个样本
    '''
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder()
    y_true_onehot = enc.fit_transform(y_true[:,np.newaxis])
    statistics = []
    for i in range(bootstraps):
        # 类别等比随机采样
        df = pd.DataFrame(data=y_true, columns=['y_true']).groupby('y_true').apply(lambda x: x.sample(n=int(fold_size*len(x)/y_true.shape[0]), replace=True))
        index = [ index[1] for index in df.index]
        y_true_sample = y_true[index]
        y_pred_sample = y_pred[index]

        # classification_report
        y_true_onehot = enc.fit_transform(y_true_sample[:,np.newaxis]).toarray()
        report = classification_report(
            y_true=y_true_sample,
            y_pred=y_pred_sample,
            digits=4,
            output_dict=True
        )
        report_df = pd.DataFrame(report)
        report_df.drop( index=['support'], inplace=True )
        statistics.append(report_df)
    return statistics


def ci_t(X, ci:float=0.95):
    from scipy import stats
    X = np.array(X)
    Xmean = X.mean(axis=1)
    Xstd = X.std(axis=1,ddof=1)
    Xinterval = stats.t.interval(ci,X.shape[1]-1,Xmean,Xstd)
    return Xmean, Xinterval


def classification_CI_report(y_true: np.ndarray, y_pred: np.ndarray, ci: int):
    """
    返回带有置信区间的classification_report(DataFrame格式)
    """
    # 统计指标: precision/recall/f1score_accuracy/micro/macro
    statistics = bootstrap_classification_report(
        y_true = y_true.astype(int),   
        y_pred = y_pred,
        bootstraps = 100, 
        fold_size = len(y_true)*10,#1000, #len(y_true)*10 
    )
    
    report_CI_df = deepcopy(statistics[0])
    
    for idx, col in it.product(report_CI_df.index, report_CI_df.columns):
        if idx=="support":
            continue
        Xmean, Xinterval = ci_t( np.array([[df.loc[idx,col] for df in statistics]]), ci=ci/100)
        mean_value = Xmean[0]
        CI_lower, CI_upper = Xinterval[0][0], Xinterval[1][0]
        report_CI_df.loc[idx,col] = "{:.3f}, {:d}%CI: {:.3f} ~ {:.3f}".format( mean_value, ci, CI_lower, CI_upper)
    report_CI_df.rename(columns={'0': 'ACSVD', '1': 'CADASIL', '2': 'CAA'},inplace=True)
    return report_CI_df


def AUC_CI_score(y_true:np.ndarray, y_prob:np.ndarray, ci:int=95):
    """
    返回一个测试组的AUC（含置信区间CI），DataFrame格式。
    """
    bootstrap_df = bootstrap_AUC_score(y_true, y_prob, bootstraps = 100, fold_size = 1000)
    Xmean, Xinterval = ci_t(bootstrap_df.T.values, ci=ci/100)
    # 整理到表格中 
    AUC_CI_score_df = pd.DataFrame([])
    for cls_name,mean_value, CI_lower, CI_upper in zip(bootstrap_df.columns, Xmean, Xinterval[0], Xinterval[1]):
        AUC_CI_score_df.loc[cls_name,'AUC'] = "{:.3f}, {:d}%CI: {:.3f} ~ {:.3f}".format( mean_value, ci, CI_lower, CI_upper)
    return AUC_CI_score_df
