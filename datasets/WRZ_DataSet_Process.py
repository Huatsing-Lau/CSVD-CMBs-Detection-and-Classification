# -*- coding: utf-8 -*-
# +
# 简介：
# 有些预处理环节比耗时,所以采用该代码实现离线预处理,这些预处理包括:

# 删除多余timepoint(实际上数据集人工梳理之后,不存在在多个timepoint的nii.gz文件了)
# 原始nii.gz文件中黑框较多，事先除去用本代码去除黑框。
# LPS方向调整
# 对病灶进行dilation

# +
import random
import numpy as np
import cupy as cp
import h5py
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk
# 必须在导入torch之前先导入SimpleITK，顺序不能颠倒，否则内核挂掉。
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
# os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
# -
class WRZ_MaskRCNN_Dataset_Processor(Dataset):
    """
    WRZ项目MaskRCNN的数据集预处理类.
    读取nii.gz格式的MRI图像和mask标注,处理后另存为指定格式.
    1种模式:
        用户指定数据集的样本文件路径名列表.
    """
    def __init__(
        self,
        lower_bound=5,
        outspacing=None,#(0.5,0.5,1.5)
    ):
        """
        """
        self.lower_bound = lower_bound
        self.outspacing = outspacing
        

    def __call__(self, filepath_img, filepath_mask):
        # 读图
        img_sitk = sitk.ReadImage(filepath_img)
        mask_sitk = sitk.ReadImage(filepath_mask)
        
        # delete nuisance channel
        if img_sitk.GetDimension()==4:
            img_sitk = img_sitk[:,:,:,0]# 对于多个timepoint的MRI,仅对第一个timepoint做病灶标注
        
        # orient
        img_sitk = sitk.DICOMOrient(img_sitk,desiredCoordinateOrientation='LPS')
        mask_sitk = sitk.DICOMOrient(mask_sitk,desiredCoordinateOrientation='LPS')
        
        # crop outer frame
        img_sitk, bbox = SITKCropOuterFrame( self.lower_bound )(img_sitk)
        mask_sitk = mask_sitk[bbox[0]:bbox[1],bbox[2]:bbox[3],bbox[4]:bbox[5]]
        
        # dilation
        mask_sitk = sitk.DilateObjectMorphology(mask_sitk, kernelRadius=(5, 5, 0), kernelType=sitk.sitkBall)
        
        # resample
        if self.outspacing:
            img_sitk = resampleimg_sitk(outspacing=self.outspacing, img_sitk=img_sitk, method=sitk.sitkLinear)
            mask_sitk = resampleimg_sitk(outspacing=self.outspacing, img_sitk=mask_sitk, method=sitk.sitkNearestNeighbor)
        
        return img_sitk, mask_sitk



def resampleimg_sitk(outspacing, img_sitk, method=sitk.sitkLinear):
    """
    将体数据重采样到指定的spacing大小\n
    paras：
    outpacing：指定的spacing，例如[1,1,1]
    img_sitk：sitk读取的image信息，这里是体数据\n
    return：重采样后的数据
    """
    outsize = [0, 0, 0]
    # 读取文件的size和spacing信息
    inputsize = img_sitk.GetSize()
    inputspacing = img_sitk.GetSpacing()
 
    transform = sitk.Transform()
    transform.SetIdentity()
    # 计算改变spacing后的size，用物理尺寸/体素的大小
    outsize[0] = round(inputsize[0] * inputspacing[0] / outspacing[0])
    outsize[1] = round(inputsize[1] * inputspacing[1] / outspacing[1])
    outsize[2] = round(inputsize[2] * inputspacing[2] / outspacing[2])
 
    # 设定重采样的一些参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(method)
    resampler.SetOutputOrigin(img_sitk.GetOrigin())
    resampler.SetOutputSpacing(outspacing)
    resampler.SetOutputDirection(img_sitk.GetDirection())
    resampler.SetSize(outsize)
    newimg_sitk = resampler.Execute(img_sitk)
    return newimg_sitk


class SITKResample(object):
    """
    将SimpleITK图像重采样到指定的spacing
    """
    def __init__(self,target_spacing):
        super().__init__()
        self.target_spacing = target_spacing
        
    def __call__(self,img_sitk):
        target_spacing = list(self.target_spacing)
        if (self.target_spacing[2] == -1):
            target_spacing[2] = max(1.0,img_sitk.GetSpacing()[2])    
        return resampleimg_sitk(target_spacing, img_sitk)


class SITKRandomResample(object):
    """
    将SimpleITK图像重采样到指定的spacing
    """
    def __init__(self,target_spacing_lower,target_spacing_upper):
        super().__init__()
        self.target_spacing_lower = target_spacing_lower
        self.target_spacing_upper = target_spacing_upper
        
    def __call__(self,img_sitk):
        if (self.target_spacing_lower[2] == -1) and (self.target_spacing_lower[2] == -1):
            target_spacing = ( 
                np.round( np.random.uniform(self.target_spacing_lower[0],self.target_spacing_upper[0]), 2),
                np.round( np.random.uniform(self.target_spacing_lower[1],self.target_spacing_upper[1]), 2),
                max(1.0,img_sitk.GetSpacing()[2]),
            )
        else:
            target_spacing = ( 
                np.round( np.random.uniform(self.target_spacing_lower[0],self.target_spacing_upper[0]), 2),
                np.round( np.random.uniform(self.target_spacing_lower[1],self.target_spacing_upper[1]), 2),
                np.round( np.random.uniform(self.target_spacing_lower[2],self.target_spacing_upper[2]), 2),
            )
        return resampleimg_sitk(target_spacing, img_sitk)


# +
class SITKDICOMOrient(object):
    """
    SipleITK.Normalize
    """
    def __init__(self,desiredCoordinateOrientation='LPS'):
        super().__init__()
        self.desiredCoordinateOrientation = desiredCoordinateOrientation
        
    def __call__(self,img_sitk):
        return sitk.DICOMOrient(img_sitk,self.desiredCoordinateOrientation)
    
    
class SITKNormalize(object):
    """
    SipleITK.Normalize
    """
    def __init__(self):
        super().__init__()
        
    def __call__(self,img_sitk):
        return sitk.Normalize(img_sitk)


# -

class SITKCropOuterFrame(object):
    """
    裁剪掉外部的黑框
    refenrence: https://simpleitk.readthedocs.io/en/v1.2.4/Examples/ImageGridManipulation/Documentation.html?highlight=crop#code
    """
    def __init__(self,lower_bound=0):
        super().__init__()
        self.lower_bound = lower_bound
        
    def __call__(self,img_sitk):
        tmp_arr = sitk.GetArrayFromImage( img_sitk ).transpose((2,1,0))
        tmp_arr = tmp_arr[::3,::3,::2]
        idx = np.where(tmp_arr>self.lower_bound)
#         idx = cp.where(cp.array(tmp_arr)>self.lower_bound)
        idx0_min = int(idx[0].min()*3)
        idx0_max = int(idx[0].max()*3+1)
        idx1_min = int(idx[1].min()*3)
        idx1_max = int(idx[1].max()*3+1)
        idx2_min = int(idx[2].min()*2)
        idx2_max = int(idx[2].max()*2+1)
        bbox = (idx0_min, idx0_max, idx1_min, idx1_max, idx2_min, idx2_max)
        img_sitk = img_sitk[
            bbox[0]:bbox[1],
            bbox[2]:bbox[3],
            bbox[4]:bbox[5]]
        
        return img_sitk,bbox


class SITKRandomCrop(object):
    """
    裁剪掉外部的黑框
    refenrence: https://simpleitk.readthedocs.io/en/v1.2.4/Examples/ImageGridManipulation/Documentation.html?highlight=crop#code
    """
    def __init__(self,size_lower=0.0,size_upper=1.0):
        super().__init__()
        self.size_lower = size_lower
        self.size_upper = size_upper
        
    def __call__(self,img_sitk):
        size = img_sitk.GetSize()
        # 最小最大尺寸
        min_size, max_size = np.ceil(self.size_lower*np.array(size)), np.floor(self.size_upper*np.array(size))
        # 实际切割出来的尺寸
        crop_size = [random.randint(a,b) for a,b in zip(min_size,max_size)]
        # 实际切割起始位置
        start_loc = [random.randint(0,b-a) for a,b in zip(crop_size,size)]
        # 切割
        img_sitk = img_sitk[
            start_loc[0]:start_loc[0]+crop_size[0], 
            start_loc[1]:start_loc[1]+crop_size[1], 
            start_loc[2]:start_loc[2]+crop_size[2]]

        return img_sitk


class SITKAdaptiveHistEqual(object):
    """
    自适应直方图均衡.
    refenrence: https://blog.csdn.net/qq_39071739/article/details/107492462
    """
    def __init__(self,alpha=0.9,beta=0.9,radius=3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.radius = radius
        
    def __call__(self,img_sitk):
        # 4.Histogram equalization
        sitk_hisequal = sitk.AdaptiveHistogramEqualizationImageFilter()
        sitk_hisequal.SetAlpha(self.alpha)
        sitk_hisequal.SetBeta(self.beta)
        sitk_hisequal.SetRadius(self.radius)
        sitk_hisequal = sitk_hisequal.Execute(img_sitk)
        return sitk_hisequal


# +
if __name__ == "__main__":
    """
    # 旧版数据集
    # ZSSY数据集
    params = {
        'pat_img': '/raid/huaqing/tyler/WRZ/data/internal_center_data/ZSSY_before_ROI/*/*SWS.nii.gz',
        'pat_mask': '/raid/huaqing/tyler/WRZ/data/internal_center_data/ZSSY_before_ROI/*/*SWS_ROI.nii.gz',
        'save_dir': '/raid/huaqing/tyler/WRZ/data/internal_center_data/ZSSY_before_ROI_processed',
        'clses_name': {"ACSVD":0, "CADASIL":1, "CAA":2},
        #'outspacing': [0.3,0.3,1.0],#[0.5,0.5,1.5],
    }
    # MMSY数据集
    params = {
        'pat_img': '/raid/huaqing/tyler/WRZ/data/external_center_data/MMSY_before_ROI/*/*SWS.nii.gz',
        'pat_mask': '/raid/huaqing/tyler/WRZ/data/external_center_data/MMSY_before_ROI/*/*SWS_ROI.nii.gz',
        'save_dir': '/raid/huaqing/tyler/WRZ/data/external_center_data/MMSY_before_ROI_processed',
        'clses_name': {"ACSVD":0, "CADASIL":1, "CAA":2},
        #'outspacing': [0.3,0.3,1.0],#[0.5,0.5,1.5],
    }

    # SDFY数据集
    params = {
        'pat_img': '/raid/huaqing/tyler/MedAI/WRZ/data/external_center_data/SDFY_before_ROI/*/*SWS.nii.gz',
        'pat_mask': '/raid/huaqing/tyler/MedAI/WRZ/data/external_center_data/SDFY_before_ROI/*/*SWS_ROI.nii.gz',
        'save_dir': '/raid/huaqing/tyler/MedAI/WRZ/data/external_center_data/SDFY_before_ROI_processed',
        'clses_name': {"ACSVD":0, "CADASIL":1, "CAA":2},
        #'outspacing': [0.3,0.3,1.0],#[0.5,0.5,1.5],
    }
    """
    
#     # 新版数据集
#     # ZSSY数据集
#     params = {
#         'pat_img': '/local_data_ssd/huaqing/tyler/MedAI/WRZ/data/AI_Final_Data/ZSSY/*/*SWS.nii.gz',
#         'pat_mask': '/local_data_ssd/huaqing/tyler/MedAI/WRZ/data/AI_Final_Data/ZSSY/*/*SWS_ROI.nii.gz',
#         'save_dir': '/local_data_ssd/huaqing/tyler/MedAI/WRZ/data/AI_Final_Data/ZSSY_processed',
#         'clses_name': {"ACSVD":0, "CADASIL":1, "CAA":2},
#         #'outspacing': [0.3,0.3,1.0],#[0.5,0.5,1.5],
#     }
    
#     # MMSY数据集
#     params = {
#         'pat_img': '/local_data_ssd/huaqing/tyler/MedAI/WRZ/data/AI_Final_Data/MMSY/*/*SWS.nii.gz',
#         'pat_mask': '/local_data_ssd/huaqing/tyler/MedAI/WRZ/data/AI_Final_Data/MMSY/*/*SWS_ROI.nii.gz',
#         'save_dir': '/local_data_ssd/huaqing/tyler/MedAI/WRZ/data/AI_Final_Data/MMSY_processed',
#         'clses_name': {"ACSVD":0, "CADASIL":1, "CAA":2},
#         #'outspacing': [0.3,0.3,1.0],#[0.5,0.5,1.5],
#     }
    
    # SDFY数据集
    params = {
        'pat_img': '/local_data_ssd/huaqing/tyler/MedAI/WRZ/data/AI_Final_Data/SDFY/*/*SWS.nii.gz',
        'pat_mask': '/local_data_ssd/huaqing/tyler/MedAI/WRZ/data/AI_Final_Data/SDFY/*/*SWS_ROI.nii.gz',
        'save_dir': '/local_data_ssd/huaqing/tyler/MedAI/WRZ/data/AI_Final_Data/SDFY_processed',
        'clses_name': {"ACSVD":0, "CADASIL":1, "CAA":2},
        #'outspacing': [0.3,0.3,1.0],#[0.5,0.5,1.5],
    }

    
    
    filepath_list_img = glob(params['pat_img'])
    filepath_list_mask = [filepath.replace('SWS.nii.gz','SWS_ROI.nii.gz') for filepath in filepath_list_img]
    
    preprocessor = WRZ_MaskRCNN_Dataset_Processor(lower_bound=5)

    for filepath_img,filepath_mask in tqdm(zip(filepath_list_img,filepath_list_mask)):
        img_sitk, mask_sitk = preprocessor(filepath_img,filepath_mask)
        # 保存处理后的图像
        save_path = os.path.join(params['save_dir'],filepath_img.split(os.path.sep)[-2])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_filepath_img = os.path.join(save_path,filepath_img.split(os.path.sep)[-1])
        sitk.WriteImage( img_sitk,save_filepath_img )
        # 保存处理后的mask
        save_filepath_mask = os.path.join(save_path,filepath_mask.split(os.path.sep)[-1])
        sitk.WriteImage( mask_sitk,save_filepath_mask )
# -


