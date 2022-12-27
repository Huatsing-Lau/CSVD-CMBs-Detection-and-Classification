# -*- coding: utf-8 -*-
# +
import random
import numpy as np
import cupy as cp
import h5py
from glob import glob
import SimpleITK as sitk
# 必须在导入torch之前先导入SimpleITK，顺序不能颠倒，否则内核挂掉。
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
# os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

# +
class WRZ_Single_withMask_Dataset(Dataset):
    """
    WRZ项目的单模态Dataset对象.
    读取nii.gz格式的MRI图像,并输出.
    1种模式:
        用户指定数据集的样本文件路径名列表.
    """
    def __init__(
        self,
        data_dir=None, 
        filepath_list=None, 
        label_list=None,
        clin_list=None,
        patient_IDs=None,
        clses_name=None,
        transform=None,
        fixedFilepaths=None,
        probHistMatch=0.5,
    ):
        """
        filepath_list: list, 每个元素是一个tuple
        clses_name: eg. {'CADASIL':0, 'aCSVD':1}
        tranfrom：至少得包含ResampleSITK和SITKtoNumpy
        fixedFilepaths: 直方图匹配的参考MRI图像文件名列表
        """
        self._data_dir = data_dir
        self._filepath_list = filepath_list
        self._label_list = label_list
        self._clin_list = clin_list
        self._clses_name = clses_name
        self.transform = transform
        self.fixedFilepaths = fixedFilepaths
        self.probHistMatch = probHistMatch
        
        if self._label_list:
            pass
        else:
            self._label_list = []
            for filepath in self._filepath_list:
                for cls_name, cls in self._clses_name.items():
                    if cls_name in filepath[0]:
                        self._label_list += [{'class':cls}]
                        break
        
        if patient_IDs:
            self.patient_IDs = patient_IDs
        else:
            self.patient_IDs = filepath_list
            
        # set self.patient_cls_ids
        self.cls_ids_prep()
            
    def cls_ids_prep(self):
        # store ids corresponding each class at the patient or case level
        unique_label = np.unique([item['class'] for item in self._label_list])
        self.patient_cls_ids = [[] for i in range(len(unique_label))]
        for i in range(len(unique_label)):
            self.patient_cls_ids[i] = np.where(np.array([item['class'] for item in self._label_list]) == i)[0]
            
    
    def getlabel(self,idx):
        return self._label_list[idx]
    
    def getclasslabel(self,idx):
        return self._label_list[idx]['class']
    
    def __len__(self):
        return len(self._filepath_list)
        
    def __getitem__(self, idx):
        filepath = self._filepath_list[idx]
        clin = self._clin_list[idx]
        label = self._label_list[idx]
        # 读MRI影像
        img_sitk = sitk.ReadImage(filepath[0])
        # delete nuisance channel
        if img_sitk.GetDimension()==4:
            img_sitk = img_sitk[:,:,:,random.sample(range(img_sitk.GetSize()[-1]),1)[0]]
        # 读mask
        mask_sitk = sitk.ReadImage(filepath[1])
        
        # orient
        img_sitk = SITKDICOMOrient('LPS')(img_sitk)
        mask_sitk = SITKDICOMOrient('LPS')(mask_sitk)
        
        # crop outer frame
#         img_sitk, mask_sitk = SITKCropOuterFrame( 5 )(img_sitk, mask_sitk)
        
#         # 随机直方图匹配
#         if self.fixedFilepaths is not None and np.random.uniform(0.0,1.0)<self.probHistMatch:
#             fixed_filepath = random.choice( self.fixedFilepaths )
#             fixed_img_sitk = sitk.ReadImage(fixed_filepath[0])
#             fixed_mask_sitk = sitk.ReadImage(fixed_filepath[1])
#             # orient
#             fixed_img_sitk = SITKDICOMOrient('LPS')(fixed_img_sitk)
#             fixed_mask_sitk = SITKDICOMOrient('LPS')(fixed_mask_sitk)
#             fixed_img_sitk, fixed_mask_sitk = SITKCropOuterFrame( 5 )(fixed_img_sitk, fixed_mask_sitk)
            
#             matcher = sitk.HistogramMatchingImageFilter()
#             if ( fixed_img_sitk.GetPixelID() in ( sitk.sitkUInt8, sitk.sitkInt8 ) ):
#                 matcher.SetNumberOfHistogramLevels(128)
#             else:
#                 matcher.SetNumberOfHistogramLevels(1024)
#             matcher.SetNumberOfMatchPoints(7)
#             matcher.ThresholdAtMeanIntensityOn()
#             img_sitk = matcher.Execute(img_sitk,fixed_img_sitk)
        
        # trasnform
        sample = (img_sitk,mask_sitk)
        sample = self.transform(sample)
#         print(filepath)
        return sample, label, clin

# +
#     target_Size = target_img.GetSize()      # 目标图像大小  [x,y,z]
#     target_Spacing = target_img.GetSpacing()   # 目标的体素块尺寸    [x,y,z]
#     target_origin = target_img.GetOrigin()      # 目标的起点 [x,y,z]
#     target_direction = target_img.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]


# +
import SimpleITK as sitk

# 初始版本,from 网上,适用于只有spacing不同的两个图
def resize_img_sitk(img_sitk, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = img_sitk.GetSize()  # 原来的体素块尺寸
    originSpacing = img_sitk.GetSpacing()

    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(int)    # spacing肯定不能是整数

    resampler.SetReferenceImage(img_sitk)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itk_img_res = resampler.Execute(img_sitk)  # 得到重新采样后的图像
    return itk_img_res


# -

def resample_img_sitk(img_sitk, newspacing=None, out_size=None): 
    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(img_sitk.GetDirection())
    resample.SetOutputOrigin(img_sitk.GetOrigin())
    resample.SetOutputSpacing(newspacing)
    out_size = tuple(np.round(np.array(img_sitk.GetSize())*np.abs(img_sitk.GetSpacing())/np.array(newspacing)).astype('int').tolist())
    resample.SetSize(out_size)
    
    # resample.SetDefaultPixelValue(0)
    
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputPixelType(sitk.sitkFloat32)
    out_image = resample.Execute(img_sitk)

    return out_image



def resampleimg_sitk(img_sitk, outspacing, outsize=None ):
    """
    将体数据重采样到指定的spacing大小\n
    paras：
    outpacing：指定的spacing，例如[1,1,1]
    img_sitk：sitk读取的image信息，这里是体数据\n
    return：重采样后的数据
    """
    
    # 读取文件的size和spacing信息
    insize = img_sitk.GetSize()
    inspacing = img_sitk.GetSpacing()
 
    transform = sitk.Transform()
    transform.SetIdentity()
    
    # 计算改变spacing后的size，用物理尺寸/体素的大小
    if outsize is None:
        outsize = [0, 0, 0]
        outsize[0] = round(insize[0] * inspacing[0] / outspacing[0])
        outsize[1] = round(insize[1] * inspacing[1] / outspacing[1])
        outsize[2] = round(insize[2] * inspacing[2] / outspacing[2])
 
    # 设定重采样的一些参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
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
    def __init__(self,outspacing):
        super().__init__()
        self.outspacing = outspacing
        
    def __call__(self,sample):
        img_sitk, mask_sitk = sample[0], sample[1]
        outspacing = list(self.outspacing)
        if (self.outspacing[2] == -1):
            outspacing[2] = max(1.0,img_sitk.GetSpacing()[2])  
            
        # 根据target_spacing计算outsize
        insize = img_sitk.GetSize()
        inspacing = img_sitk.GetSpacing()
        outsize = [0.0, 0.0, 0.0]
        outsize[0] = round(insize[0] * inspacing[0] / outspacing[0])
        outsize[1] = round(insize[1] * inspacing[1] / outspacing[1])
        outsize[2] = round(insize[2] * inspacing[2] / outspacing[2])
        
        img_sitk = resampleimg_sitk(img_sitk, outspacing=outspacing, outsize=outsize )
        mask_sitk = resampleimg_sitk(mask_sitk, outspacing=outspacing, outsize=outsize )
        return (img_sitk,mask_sitk)


class SITKRandomResample(object):
    """
    将SimpleITK图像重采样到指定的spacing
    """
    def __init__(self,target_spacing_lower,target_spacing_upper):
        super().__init__()
        self.target_spacing_lower = target_spacing_lower
        self.target_spacing_upper = target_spacing_upper
        
    def __call__(self,sample):
        img_sitk, mask_sitk = sample[0], sample[1]
        
        # for debug
        if not img_sitk.GetSize()[0]==mask_sitk.GetSize()[0]:
#             import pdb
#             pdb.set_trace()
            print(img3D.shape[0],mask.shape[0])
        
        if (self.target_spacing_lower[2] == -1) and (self.target_spacing_lower[2] == -1):
            outspacing = ( 
                np.round( np.random.uniform(self.target_spacing_lower[0],self.target_spacing_upper[0]), 2),
                np.round( np.random.uniform(self.target_spacing_lower[1],self.target_spacing_upper[1]), 2),
                max(1.0,img_sitk.GetSpacing()[2]),
            )
        else:
            outspacing = ( 
                np.round( np.random.uniform(self.target_spacing_lower[0],self.target_spacing_upper[0]), 2),
                np.round( np.random.uniform(self.target_spacing_lower[1],self.target_spacing_upper[1]), 2),
                np.round( np.random.uniform(self.target_spacing_lower[2],self.target_spacing_upper[2]), 2),
            )
        # 根据target_spacing计算outsize
        insize = img_sitk.GetSize()
        inspacing = img_sitk.GetSpacing()
        outsize = [0.0, 0.0, 0.0]
        outsize[0] = round(insize[0] * inspacing[0] / outspacing[0])
        outsize[1] = round(insize[1] * inspacing[1] / outspacing[1])
        outsize[2] = round(insize[2] * inspacing[2] / outspacing[2])
        
        img_sitk = resampleimg_sitk(img_sitk, outspacing=outspacing, outsize=outsize )
        mask_sitk = resampleimg_sitk(mask_sitk, outspacing=outspacing, outsize=outsize )
        
        if not img_sitk.GetSize()==mask_sitk.GetSize():
#             import pdb
#             pdb.set_trace()
            print('WTF')
        return (img_sitk,mask_sitk)


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
        
    def __call__(self,sample):
        img_sitk, mask_sitk = sample[0], sample[1]
        img_sitk = sitk.Normalize(img_sitk)
        return (img_sitk,mask_sitk)


# -

class SITKCropOuterFrame(object):
    """
    裁剪掉外部的黑框
    refenrence: https://simpleitk.readthedocs.io/en/v1.2.4/Examples/ImageGridManipulation/Documentation.html?highlight=crop#code
    """
    def __init__(self,lower_bound=0):
        super().__init__()
        self.lower_bound = lower_bound
        
    def __call__(self,sample):
        img_sitk, mask_sitk = sample[0], sample[1]
        tmp_arr = sitk.GetArrayFromImage( img_sitk ).transpose((2,1,0))
        tmp_arr = tmp_arr[::2,::2,::2]
#         idx = np.where(tmp_arr>self.lower_bound)
        idx = cp.where(cp.array(tmp_arr)>self.lower_bound)
        idx0_min = int(idx[0].min()*2)
        idx0_max = int(idx[0].max()*2+1)
        idx1_min = int(idx[1].min()*2)
        idx1_max = int(idx[1].max()*2+1)
        idx2_min = int(idx[2].min()*2)
        idx2_max = int(idx[2].max()*2+1)
        bbox = (idx0_min, idx0_max, idx1_min, idx1_max, idx2_min, idx2_max)
        img_sitk = img_sitk[
            bbox[0]:bbox[1],
            bbox[2]:bbox[3],
            bbox[4]:bbox[5]]
        mask_sitk = mask_sitk[
            bbox[0]:bbox[1],
            bbox[2]:bbox[3],
            bbox[4]:bbox[5]]
        
        return (img_sitk,mask_sitk)


class SITKRandomCrop(object):
    """
    裁剪掉外部的黑框
    refenrence: https://simpleitk.readthedocs.io/en/v1.2.4/Examples/ImageGridManipulation/Documentation.html?highlight=crop#code
    """
    def __init__(self,size_lower=0.0,size_upper=1.0):
        super().__init__()
        self.size_lower = size_lower
        self.size_upper = size_upper
        
    def __call__(self,sample):
        img_sitk, mask_sitk = sample[0], sample[1]
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
        mask_sitk = mask_sitk[
            start_loc[0]:start_loc[0]+crop_size[0], 
            start_loc[1]:start_loc[1]+crop_size[1], 
            start_loc[2]:start_loc[2]+crop_size[2]]

        return (img_sitk,mask_sitk)


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
        
    def __call__(self,sample):
        img_sitk, mask_sitk = sample[0], sample[1]
        # 4.Histogram equalization
        sitk_hisequal = sitk.AdaptiveHistogramEqualizationImageFilter()
        sitk_hisequal.SetAlpha(self.alpha)
        sitk_hisequal.SetBeta(self.beta)
        sitk_hisequal.SetRadius(self.radius)
        img_sitk = sitk_hisequal.Execute(img_sitk)
        return (img_sitk,mask_sitk)



class SITKtoNumpy(object):
    """
    将SimpleITK对象转为Numpy数组
    """
    def __init__(self):
        super().__init__()
    def __call__(self,sample):
        img_sitk, mask_sitk = sample[0], sample[1]
        return (sitk.GetArrayFromImage(img_sitk),sitk.GetArrayFromImage(mask_sitk))


# +
class ToTensor(object):
    """
    array转为tensor
    """
    def __init__(self,):
        super().__init__()
    def __call__(self,sample):
        img_arr,mask_arr = sample[0], sample[1]
        img = torch.tensor(img_arr.astype(float),dtype=torch.float)
        mask = torch.tensor(mask_arr.astype(float),dtype=torch.float)
        return (img,mask)
    

    
    
class ScaleVoxelValue(object):
    """
    调整像素取值范围到0~255
    """
    def __init__(self,quantile=0.995):
        super().__init__()
        self.quantile = quantile
    def __call__(self,sample):
        img,mask = sample[0], sample[1]
        upper_bound = torch.quantile(img.flatten().float()[::5],self.quantile)
        img = torch.clip(img, 0, upper_bound)/upper_bound*255.0
        mask = (mask>0)*255.0
        return (img,mask)

    
class Normalize(object):
    """
    归一化。
    要求颜色通道是第1维。
    """
    def __init__(self,mean=[125.3, 123.0, 113.9], std=[63.0, 62.1, 66.7]):
        self.mean = mean
        self.std = std
    def __call__(self,sample):
        img3D,mask = sample[0], sample[1]
        for i,(mean,std) in enumerate(zip(self.mean,self.std)):
            img3D[:,i,...] = (img3D[:,i,...]-mean)/std
        return (img3D,mask)
    
class RandomDropInstance:
    """
    随机丢弃一定比例的实例.
    """
    def __init__(self,p=0.25,sort='ascending'):
        self.p = p
        self.sort = sort
    def __call__(self,sample):
        img3D,mask = sample[0], sample[1]
        if not img3D.shape[0]==mask.shape[0]:
            print(img3D.shape[0],mask.shape[0])
        n = img3D.shape[0]
        n_pick = int(n*(1-self.p))
        
        idx_pick = random.sample( range(n),n_pick )
        if self.sort == 'ascending':
            idx_pick.sort(reverse=False)
        elif self.sort == 'descending':
            idx_pick.sort(reverse=True)
        else:
            pass
        try:
            img3D_rt, mask_rt = img3D[idx_pick,:], mask[idx_pick,:]
        except:
#             import pdb
#             pdb.set_trace()
            # ('../../data/AI_Final_Data/ZSSY_processed/ACSVD/061_SWS.nii.gz', '../../data/AI_Final_Data/ZSSY_processed/ACSVD/061_SWS_ROI.nii.gz')
            print('WTF')
        return (img3D[idx_pick,:],mask[idx_pick,:])
    
class RandomSampleInstance:
    """
    随机采样一定数量的实例.
    """
    def __init__(self,max_instance=25,sort='ascending'):
        self.n_pick = max_instance
        self.sort = sort
    def __call__(self,sample):
        img3D, mask = sample[0], sample[1]
        if not img3D.shape[0]==mask.shape[0]:
            print(img3D.shape[0],mask.shape[0])
        n = img3D.shape[0]
        n_pick = min(n,self.n_pick)
        
        idx_pick = random.sample( range(n),n_pick )
        if self.sort == 'ascending':
            idx_pick.sort(reverse=False)
        elif self.sort == 'descending':
            idx_pick.sort(reverse=True)
        else:
            pass
        try:
            img3D_rt, mask_rt = img3D[idx_pick,:], mask[idx_pick,:]
        except:
#             import pdb
#             pdb.set_trace()
            print('WTF')
        return (img3D[idx_pick,:],mask[idx_pick,:])

    
class RandomFilp(object):
    """
    以制定(矢状面位)对称轴，随机翻转。
    """
    def __init__(self,prob,dims):
        super().__init__()
        self.prob = prob
        self.dims = dims
    def __call__(self,sample):
        img3D,mask = sample[0], sample[1]
        if np.random.uniform(0.0,1.0)<self.prob:
            img3D = torch.flip(img3D,self.dims)
            mask = torch.flip(mask,self.dims)
        return (img3D,mask)
  

class AddChannelDim(object):
    """
    添加颜色通道
    """
    def __init__(self,):
        super().__init__()
    def __call__(self,sample):
        img3D, mask = sample[0], sample[1]
        return (img3D.unsqueeze(1),mask.unsqueeze(1))
    
# class RepeatChannelDim(object):
#     """
#     重复颜色通道
#     """
#     def __init__(self,n=3):
#         super().__init__()
#         self.n = n
#     def __call__(self,img3D,mask):
#         return img3D.repeat(1,self.n,1,1)

class ConcatChannelDim(object):
    """
    颜色通道拼接
    """
    def __init__(self,dim=1):
        super().__init__()
        self.dim = dim
    def __call__(self,sample):
        return torch.cat(sample,dim=self.dim)


# -

if __name__ == "__main__":
#     data/internal_center_data/ZSSY_before_ROI/ACSVD_SWS_before/
    pat = "/local_data_ssd/huaqing/tyler/MedAI/WRZ/data/AI_Final_Data/MMSY_processed/*/*SWS.nii.gz"
    # "/local_data_ssd/huaqing/tyler/MedAI/WRZ/data/AI_Final_Data/ZSSY_processed/*/*SWS.nii.gz"
    # '../../../data/internal_center_data/ZSSY_before_ROI/*/*SWS.nii.gz'
    pat_fixed = '../../../data/external_center_data/SDFY_before_ROI/*/*SWS.nii.gz'
    clses_name = {"ACSVD":0, "CADASIL":1, "CAA":2}
    filepath_list = glob(pat)
    
    fixedFilepaths = glob(pat_fixed)
    # 添加(预测)mask文件路径名
    filepath_list = [(filepath,filepath.replace('.nii.gz','_ROI.nii.gz')) for filepath in filepath_list]
    
    dataset = WRZ_Single_withMask_Dataset(
        data_dir='../../../data/internal_center_data/ZSSY_data',
        #'/raid/huaqing/tyler/WRZ/data/data_V2_nii_gz', 
        filepath_list=filepath_list, 
        label_list=None,
        clses_name=clses_name,
        clin_list=[None]*len(filepath_list),
        transform=transforms.Compose([
            SITKRandomCrop( 0.75,1.0 ),
            SITKRandomResample( (0.8,0.8,3.0),(1.2,1.2,4.0) ),#(1.0,1.0,2.0),(1.0,1.0,3.0) #(0.4,0.4,2.0),(0.6,0.6,3.0)# ( (0.4,0.4,3.0),(0.6,0.6,6.0) )
            SITKAdaptiveHistEqual(),
            SITKtoNumpy(),
            ToTensor(),
            RandomFilp(prob=0.5,dims=[1,2]),
            ScaleVoxelValue(quantile=0.995),
            RandomDropInstance(p=0.2,sort='ascending'),
            RandomSampleInstance(max_instance=30,sort='ascending'),
            AddChannelDim(),
            ConcatChannelDim(),
        ]),
    )
    for i in range(len(filepath_list)):
        img,label,clin = dataset.__getitem__(idx=i)#215
        print( img.shape, img.max(), img.min() )

# +
# # ('/local_data_ssd/huaqing/tyler/MedAI/WRZ/data/AI_Final_Data/ZSSY_processed/ACSVD/061_SWS.nii.gz', '/local_data_ssd/huaqing/tyler/MedAI/WRZ/data/AI_Final_Data/ZSSY_processed/ACSVD/061_SWS_ROI.nii.gz')
# if __name__ == "__main__":
#     filepath = ('../../../data/AI_Final_Data/ZSSY_processed/ACSVD/067_SWS.nii.gz', '../../../data/AI_Final_Data/ZSSY_processed/ACSVD/067_SWS_ROI.nii.gz')
#     import SimpleITK as sitk
#     import numpy as np
#     mask_sitk = sitk.ReadImage(filepath[1])
# np.unique( sitk.GetArrayFromImage(mask_sitk) )
# -



# +
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     plt.subplots(1,3,dpi=100)
#     plt.subplot(1,3,1)
#     plt.imshow(img[30,0,:,:],cmap='gray')
#     plt.subplot(1,3,2)
#     plt.imshow(img[:,0,160,:],cmap='gray')
#     plt.subplot(1,3,3)
#     plt.imshow(img[:,0,:,135],cmap='gray')
#     plt.show()
    
#     plt.subplots(1,3,dpi=100)
#     plt.subplot(1,3,1)
#     plt.imshow(img[30,1,:,:],cmap='gray')
#     plt.subplot(1,3,2)
#     plt.imshow(img[:,1,160,:],cmap='gray')
#     plt.subplot(1,3,3)
#     plt.imshow(img[:,1,:,135],cmap='gray')
#     plt.show()

# +
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     plt.subplots(1,3,dpi=100)
#     plt.subplot(1,3,1)
#     plt.imshow(img[8,0,:,:],cmap='gray')
#     plt.subplot(1,3,2)
#     plt.imshow(img[:,0,160,:],cmap='gray')
#     plt.subplot(1,3,3)
#     plt.imshow(img[:,0,:,135],cmap='gray')
#     plt.show()

# +
# if __name__ == "__main__":
#     filepath_img = '/local_data_ssd/huaqing/tyler/MedAI/WRZ/data/AI_Final_Data/ZSSY_processed/ACSVD/061_SWS.nii.gz'
#     filepath_mask = '/local_data_ssd/huaqing/tyler/MedAI/WRZ/data/AI_Final_Data/ZSSY_processed/ACSVD/061_SWS_ROI.nii.gz'
#     img_sitk = sitk.ReadImage(filepath_img)
#     mask_sitk = sitk.ReadImage(filepath_mask)
#     print( img_sitk.GetSize(), mask_sitk.GetSize() )
    
#     transform=transforms.Compose([
#         SITKCropOuterFrame( 0 ), 
#         SITKRandomCrop( 0.75,1.0 ),
#         SITKtoNumpy(),
#         ToTensor(),
#         RandomFilp(prob=0.5,dims=[2]),
#         ScaleVoxelValue(),
#         RandomDropInstance(p=0.2),
#         AddChannelDim(),
#         ConcatChannelDim(),
#     ]),
#     sample = (img_sitk,mask_sitk)
#     sample = transform(sample)
# -



