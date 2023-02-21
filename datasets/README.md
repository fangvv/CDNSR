# dataset directory
```sh
datasets
├── DIV2K800_scale
│   ├── Bic
│   │   └── x4
│   ├── GT
│   ├── HR
│   │   └── x4
│   └── LR
│       └── x4
├── DIV2K_scale_sub
│   ├── GT
│   └── LR
├── DIV2K_scale_sub_psnr_GT_class1
├── DIV2K_scale_sub_psnr_GT_class2
├── DIV2K_scale_sub_psnr_GT_class3
├── DIV2K_scale_sub_psnr_LR_class1
├── DIV2K_scale_sub_psnr_LR_class2
├── DIV2K_scale_sub_psnr_LR_class3
├── DIV2K_valid_HR_sub
│   ├── GT
│   └── LR
├── DIV2K_valid_HR_sub_psnr_GT_class1
├── DIV2K_valid_HR_sub_psnr_GT_class2
├── DIV2K_valid_HR_sub_psnr_GT_class3
├── DIV2K_valid_HR_sub_psnr_LR_class1
├── DIV2K_valid_HR_sub_psnr_LR_class2
├── DIV2K_valid_HR_sub_psnr_LR_class3
├── test2k
│   ├── HR
│   │   └── X4
│   └── LR
│       └── X4
├── test4k
│   ├── HR
│   │   └── X4
│   └── LR
│       └── X4
├── test8k
│   ├── HR
│   │   └── X4
│   └── LR
│       └── X4
├── val_10
│   ├── HR
│   │   └── X4
│   └── LR
│       └── X4
└── visual
    ├── HR
    │   └── X4
    └── LR
        └── X4
```

# dataset prepare
We use a similar approach to get data set provided by the "classSR" repo.  
The data set is about 100 gigabytes.  
Generate simple, medium, hard (class1, class2, class3) validation data
```python
cd codes/data_scripts
python data_augmentation.py
python generate_mod_LR_bic.py
python extract_subimages_train.py
python divide_subimages_train.py
```

Generate simple, medium, hard (class1, class2, class3) training data
```python
cd codes/data_scripts
python data_augmentation.py
python generate_mod_LR_bic.py
python extract_subimages_train.py
```