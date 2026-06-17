# CDNSR

This is the source code for our paper: **Classification-based Dynamic Network for Efficient Super-Resolution**. A brief introduction of this work is as follows:

> Deep neural networks (DNNs) based approaches have achieved superior performance in single image super-resolution (SR). To obtain better visual quality, DNNs for SR are generally designed with massive computation overhead. To accelerate network inference under resource constraints, we propose a classification-based dynamic network for efficient super-resolution (CDNSR), which combines the classification and SR networks in a unified framework. Specifically, CDNSR decomposes a large image into a number of image-patches, and uses a classification network to categorize them into different classes based on the restoration difficulty. Each class of image-patches will be handled by the SR network that corresponds to the difficulty of this class. In particular, we design a new loss to trade off between the computational overhead and the reconstruction quality. Besides, we apply contrastive learning based knowledge distillation to guarantee the performance of SR networks and the quality of reconstructed images. Extensive experiments show that CDNSR significantly outperforms the other SR networks and backbones on image quality and computational overhead.

> 基于深度神经网络（DNN）的方法在单图像超分辨率（SR）任务中已取得卓越性能。为获得更优的视觉质量，超分辨率神经网络通常被设计为具有巨大计算开销的结构。为在资源约束下加速网络推理，我们提出一种基于分类的动态超分辨率高效网络（CDNSR），将分类网络与超分辨率网络整合至统一框架。具体而言，CDNSR将大幅图像分解为若干图像块，并采用分类网络根据复原难度将其划分为不同类别。每类图像块将由与其难度相对应的超分辨率网络进行处理。我们特别设计了一种新型损失函数，用于权衡计算开销与重建质量。此外，应用基于对比学习的知识蒸馏技术来保证超分辨率网络的性能与重建图像的质量。大量实验表明，CDNSR在图像质量和计算开销方面显著优于其他超分辨率网络及骨干模型。

This paper has been published by ICASSP 2023, and can be accessed from [IEEExplore](https://ieeexplore.ieee.org/document/10096521). Due to the 5-page requirement of this conference, we provide a full version of technique report in this repo.

## Required software

- PyTorch
- NumPy
- OpenCV (`opencv-python`)
- tqdm

## Project Structure

```
CDNSR/
├── codes/
│   ├── data/                          # Datasets and data loaders
│   │   ├── LQ_dataset.py              # LR-only dataset for SR-Net pre-training
│   │   ├── LQGT_dataset.py            # LR/GT paired dataset
│   │   ├── LQ_label_dataset.py        # Patch dataset with classification labels
│   │   ├── LQGT_classify_test.py      # Test loader for classification branch
│   │   ├── LQGT_rcan_dataset.py       # RCAN-style dataset wrapper
│   │   ├── data_sampler.py            # Distributed sampler
│   │   └── util.py                    # Data utilities (degradation, etc.)
│   ├── data_scripts/                  # Pre-processing scripts
│   │   ├── generate_mod_LR_bic.py     # Generate bicubic-downsampled LR images
│   │   ├── extract_subimages_train.py # Crop training sub-images
│   │   ├── extract_subimages_test.py  # Prepare test sub-images
│   │   ├── divide_sub_images_train.py # Divide large images into patches for training
│   │   ├── divide_sub_images_test.py  # Divide large images into patches for testing
│   │   └── data_augmentation.py       # Random flip / rotation augmentation
│   ├── metrics/
│   │   └── calculate_PSNR_SSIM.py     # PSNR / SSIM evaluation
│   ├── models/
│   │   ├── archs/                     # Network architectures
│   │   │   ├── CARN_arch.py           # CARN baseline SR-Net
│   │   │   ├── FSRCNN_arch.py         # FSRCNN baseline SR-Net
│   │   │   ├── RCAN_arch.py           # RCAN baseline SR-Net
│   │   │   ├── SRResNet_arch.py       # SRResNet baseline SR-Net
│   │   │   ├── CDNSR_carn_arch.py     # CDNSR built on CARN
│   │   │   ├── CDNSR_fsrcnn_arch.py   # CDNSR built on FSRCNN
│   │   │   ├── CDNSR_rcan_arch.py     # CDNSR built on RCAN
│   │   │   ├── CDNSR_srresnet_arch.py # CDNSR built on SRResNet
│   │   │   └── arch_util.py           # gumbel_softmax, FLOPs counter, etc.
│   │   ├── CDNSR_model.py             # CDNSR training/eval logic with KD loss
│   │   ├── SR_model.py                # Plain SR-Net training/eval logic
│   │   ├── networks.py                # Network builder
│   │   ├── loss.py                    # Charbonnier, classification, FLOPs, contrastive, KD losses
│   │   ├── lr_scheduler.py            # LR schedulers
│   │   └── base_model.py              # Base model wrapper
│   ├── options/
│   │   ├── train/                     # Training YAMLs (one per branch / backbone)
│   │   └── test/                      # Testing YAMLs
│   ├── utils/                         # Logging, FLOPs, misc helpers
│   ├── train_CDNSR.py                 # Entry: train / distill CDNSR
│   ├── train_SR_Net.py                # Entry: pre-train each SR branch
│   ├── test_CDNSR.py                  # Entry: test CDNSR pipeline
│   └── test_SR_Net.py                 # Entry: test a single SR branch
├── datasets/                          # Place training/testing datasets here (see datasets/README.md)
├── Tech Report CDNSR.pdf              # Full technical report
└── README.md
```

## Core Modules

### Network Architecture — `models/archs/CDNSR_carn_arch.py`

The CDNSR model is a unified network that combines a `Classifier` with multiple SR sub-networks. By default it stacks three CARN branches of different capacities.

**Three SR branches (using CARN as backbone example):**

| Branch | Channels (`nf`) | Target Difficulty | Approx. FLOPs |
|---|---|---|---|
| `net1` | 36 | Easy | Low |
| `net2` | 52 | Medium | Medium |
| `net3` | 64 | Hard | High |

The `Classifier` outputs a 3-way logits per patch; a `gumbel_softmax` (temperature `tau`) is used during training to make a soft, differentiable routing decision. The final output is a weighted sum of the three branch outputs. At inference, the argmax of the classifier is taken and only one branch is executed, achieving dynamic computation.

### Loss Design — `models/loss.py`

CDNSR trains with a compound loss:

| Loss | Weight | Purpose |
|---|---|---|
| Pixel loss (`l1` / `cb`) | `pixel_criterion` | Reconstruction fidelity |
| `class_loss_3class` | `class_loss_w` | Push classifier to be confident |
| `average_loss_3class` | `average_loss_w` | Balance the proportion of branches used |
| `EE_flops_loss` | `flops_loss_w` | Penalize FLOPs above `target_flops` |
| `ContrastLoss` (CS) | `cs_loss_w` | Patch-level contrastive supervision |
| `CSDLoss` (CSD) | `csd_loss_w` | Distillation contrastive loss (branch ↔ teacher) |
| `PerceptualLoss` | `perceptual_loss_w` | VGG perceptual loss |
| `KDL1Loss` | `kd_l1_loss_w` | L1 knowledge distillation from teacher |

### Training / Evaluation Pipeline — `models/CDNSR_model.py`

`CDNSR_Model` (subclass of `BaseModel`) orchestrates:

- Building the network via `networks.define_G(opt)` and wrapping it with `DataParallel` / `DistributedDataParallel`.
- Loading pre-trained SR branches as initialization.
- Building all the losses listed above and combining them in `calculate_loss()`.
- Patch-level iteration: each image is decomposed into patches (`patch_size`, `step`), processed by the dynamic network, and stitched back.
- Logging and checkpointing (`save_network`, `load_network`).

### Data Pipeline

- **Pre-training SR branches:** use `LQ_dataset.py` / `LQGT_dataset.py` with full-image LR/GT pairs.
- **Training CDNSR:** use `LQ_label_dataset.py`, where each LR patch is paired with a pseudo-label (easy / medium / hard) generated by branch L1 errors. See `data_scripts/divide_sub_images_*.py` for the patch extraction workflow.
- **Testing:** full images are divided into patches via `divide_sub_images_test.py`, processed branch by branch according to the classifier, then merged back. PSNR / SSIM are computed on the Y channel (see `metrics/calculate_PSNR_SSIM.py`).

### Configuration Files

All hyperparameters are managed by YAML files in `codes/options/`:

| Config | Purpose |
|---|---|
| `train_CARN_branch1/2/3.yml` | Pre-train the three CARN branches |
| `train_CDNSR_CARN.yml` | Train CDNSR (CARN backbone) |
| `train_CDNSR_CARN_KD.yml` | Train CDNSR with contrastive knowledge distillation |
| `train_CDNSR_FSRCNN/RCAN/SRResNet.yml` | Train CDNSR on other backbones |
| `test_CARN.yml`, `test_CDNSR_CARN.yml`, ... | Evaluation configs |

## Usage

### 1. Pre-train each SR branch

```bash
cd codes
python train_SR_Net.py -opt options/train/train_CARN_branch1.yml
python train_SR_Net.py -opt options/train/train_CARN_branch2.yml
python train_SR_Net.py -opt options/train/train_CARN_branch3.yml
```

### 2. Train CDNSR (joint classification + dynamic inference)

```bash
cd codes
python train_CDNSR.py -opt options/train/train_CDNSR_CARN.yml
```

### 3. Train CDNSR with contrastive knowledge distillation

```bash
cd codes
python train_CDNSR.py -opt options/train/train_CDNSR_CARN_KD.yml
```

### 4. Test

```bash
cd codes
# Test a single pre-trained SR branch
python test_SR_Net.py -opt options/test/test_CARN.yml

# Test the full CDNSR pipeline (dynamic)
python test_CDNSR.py -opt options/test/test_CDNSR_CARN.yml
```

> Please make sure datasets are placed under `datasets/` and the `dataroot` fields in the YAML configs are updated accordingly. See [datasets/README.md](datasets/README.md) for the expected directory layout.

## Citation

If you find CDNSR useful or relevant to your project and research, please kindly cite our paper:

```
@inproceedings{wang2023classification,
  title={Classification-Based Dynamic Network for Efficient Super-Resolution},
  author={Wang, Qi and Fang, Weiwei and Wang, Meng and Cheng, Yusong},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

## For more

We have another work on [UAV-DDPG](https://github.com/fangvv/UAV-DDPG) and related deep reinforcement learning / efficient inference research by the same group. Feel free to check them out for reference.

A full technical report describing CDNSR in detail is provided in this repository as `Tech Report CDNSR.pdf`.

## Contact

Qi Wang ([20120417@bjtu.edu.cn](mailto:20120417@bjtu.edu.cn))

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.
