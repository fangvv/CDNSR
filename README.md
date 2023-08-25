## CDNSR

This is the source code for our paper: Classification-based Dynamic Network for Efficient Super-Resolution. A brief introduction of this work is as follows:

> Deep neural networks (DNNs) based approaches have achieved superior performance in single image super-resolution (SR). To obtain better visual quality, DNNs for SR are generally designed with massive computation overhead. To accelerate network inference under resource constraints, we propose a classification-based dynamic network for efficient super-resolution (CDNSR), which combines the classification and SR networks in a unified framework. Specifically, CDNSR decomposes a large image into a number of image-patches, and uses a classification network to categorize them into different classes based on the restoration difficulty. Each class of image-patches will be handled by the SR network that corresponds to the difficulty of this class. In particular, we design a new loss to trade off between the computational overhead and the reconstruction quality. Besides, we apply contrastive learning based knowledge distillation to guarantee the performance of SR networks and the quality of reconstructed images. Extensive experiments show that CDNSR significantly outperforms the other SR networks and backbones on image quality and computational overhead.

This paper has been published by ICASSP 2023, and can be accessed from [IEEExplore](https://ieeexplore.ieee.org/document/10096521). Due to the 5-page requirement of this conference, we provide a full version of technique report in this repo.

## Citation

    @inproceedings{wang2023classification,
      title={Classification-Based Dynamic Network for Efficient Super-Resolution},
      author={Wang, Qi and Fang, Weiwei and Wang, Meng and Cheng, Yusong},
      booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      pages={1--5},
      year={2023},
      organization={IEEE}
    }

## Required software

PyTorch

## Pre-train & test SR-Nets
`train`
```python
cd codes
python train_SR_Net.py -opt options/train/train_CARN_branch1.yml
python train_SR_Net.py -opt options/train/train_CARN_branch2.yml
python train_SR_Net.py -opt options/train/train_CARN_branch3.yml
```
`test`
```
cd codes
python test_SR_Net.py -opt options/test/test_CARN.yml
```

## Train & test CDNSR
`train`
```
cd codes
python train_CDNSR.py -opt options/train/train_CDNSR_CARN.yml

```
`distill`
```
cd codes
python train_CDNSR.py -opt options/train/train_CDNSR_CARN_KD.yml
```

`test`
```
cd codes
python test_CDNSR.py -opt options/test/test_CDNSR_CARN.yml
```

## Contact

Qi Wang (20120417@bjtu.edu.cn)

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.

