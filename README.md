## CDNSR

This is the source code for our paper: Classification-based Dynamic Network for Efficient Super-Resolution. A brief introduction of this work is as follows:

> Deep neural networks (DNNs) based approaches have achieved superior performance in single image super-resolution (SR). To obtain better visual quality, DNNs for SR are generally designed with massive computation overhead. To accelerate network inference under resource constraints, we propose a classification-based dynamic network for efficient super-resolution (CDNSR), which combines the classification and SR networks in a unified framework. Specifically, CDNSR decomposes a large image into a number of image-patches, and uses a classification network to categorize them into different classes based on the restoration difficulty. Each class of image-patches will be handled by the SR network that corresponds to the difficulty of this class. In particular, we design a new loss to trade off between the computational overhead and the reconstruction quality. Besides, we apply contrastive learning based knowledge distillation to guarantee the performance of SR networks and the quality of reconstructed images. Extensive experiments show that CDNSR significantly outperforms the other SR networks and backbones on image quality and computational overhead.

This paper has been accepted by ICASSP 2023. Due to the 5-page requirement of this conference, we provide a full version of technique report in this repo.

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
