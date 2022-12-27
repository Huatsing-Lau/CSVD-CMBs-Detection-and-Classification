# CSVD-CMBs-Detection-and-Classification
Code for the paper: 


# Data Preprocess
    cd datasets
    python WRZ_DataSet_Process.py


# Training
    cd TOAD_end2end

## Multi-series and Single Task:
    CUDA_VISIBLE_DEVICES=0 python main_WRZ_MultiModal_end2end.py --log_data --weighted_sample --drop_out --early_stopping
## Multi-series and Dual-Task:
    CUDA_VISIBLE_DEVICES=0 python main_WRZ_MultiModal_DualTask_end2end.py --log_data --weighted_sample --drop_out --early_stopping

## Single Series, with CMBs-Mask,  No Clinical informations,  Single Task:
    CUDA_VISIBLE_DEVICES=0 python main_WRZ_SingleModal_end2end.py --log_data --weighted_sample --drop_out --early_stopping --backbone_requires_grad

## Single Series, with CMBs-Mask,  No Clinical informations, Single Task:
    CUDA_VISIBLE_DEVICES=0 python main_WRZ_SingleModal_end2end_withMask.py --log_data --weighted_sample --drop_out --early_stopping --backbone_requires_grad  --exp_code SWS_withMask_3category_ZSSYandMMSY_20221124 > results/SWS_withMask_3category_ZSSYandMMSY_20221124_s1.log &

## Single Series, with CMBs-Mask and Clinical informations, Single Task:
    CUDA_VISIBLE_DEVICES=0 python main_WRZ_SingleModal_end2end_withMask.py --log_data --weighted_sample --drop_out --early_stopping --backbone_requires_grad --mix  --exp_code SWS_withMask_mix_3category_ZSSYandMMSY_20221124 > results/SWS_withMask_mix_3category_ZSSYandMMSY_20221124_s1.log &


# Test
## Single Series, with CMBs-Mask and Clinical informations, Single Task:
    main_test_WRZ_SingleModal_end2end_withMask.ipynb

```python

```
