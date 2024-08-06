# Density-invariant Features for Distant Point Cloud Registration (ICCV 2023)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/density-invariant-features-for-distant-point/point-cloud-registration-on-kitti-distant-pcr)](https://paperswithcode.com/sota/point-cloud-registration-on-kitti-distant-pcr?p=density-invariant-features-for-distant-point)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/density-invariant-features-for-distant-point/point-cloud-registration-on-nuscenes-distant)](https://paperswithcode.com/sota/point-cloud-registration-on-nuscenes-distant?p=density-invariant-features-for-distant-point)

Registration of distant outdoor LiDAR point clouds is crucial to extending the 3D vision of collaborative autonomous vehicles, and yet is challenging due to small overlapping area and a huge disparity between observed point densities. In this paper, we propose Group-wise Contrastive Learning (GCL) scheme to extract density-invariant geometric features to register distant outdoor LiDAR point clouds. We mark through theoretical analysis and experiments that, contrastive positives should be independent and identically distributed (i.i.d.), in order to train density-invariant feature extractors. We propose upon the conclusion a simple yet effective training scheme to force the feature of multiple point clouds in the same spatial location (referred to as positive groups) to be similar, which naturally avoids the sampling bias introduced by a pair of point clouds to conform with the i.i.d. principle. The resulting fully-convolutional feature extractor is more powerful and density-invariant than state-of-the-art methods, improving the registration recall of distant scenarios on KITTI and nuScenes benchmarks by 40.9% and 26.9%, respectively.

Paper links: [Camera-ready](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Density-invariant_Features_for_Distant_Point_Cloud_Registration_ICCV_2023_paper.html),  [Arxiv preprint](https://arxiv.org/abs/2307.09788)

This repository is the implementation of FCGF+GCL upon [FCGF](https://github.com/chrischoy/FCGF). KPConv+GCL is available in a separate repository [here](https://github.com/liuQuan98/GCL-KPConv).

## News

20231113 - Pretrained weights are released [here](https://drive.google.com/file/d/17rt_eNBiLdOr5WxxYz8rOuUDwGsnDTXZ/view?usp=sharing).

20230808 - Source code is released.

20230713 - Our paper has been accepted by ICCV'23!

## Overview of Group-wise Contrastive Learning (GCL)

<div align="center">
<img src=assets\overview.png>
</div>

## Results

All results below are tested at a registration criterion of TE<2m, RE<5°.
The metrics of RR(%), RRE(°) and RTE(cm) are reported.

KITTI:
| Method | RR | RRE | RTE |
| :- | :-: | - | - |
| FCGF         |97.8    |0.35   | 12.6|
| FCGF+APR     |98.2    |0.34   | 9.6 |
| Predator     |100.0   |0.31   | 7.4 | 
| Predator+APR |100.0   |0.30   | 7.3 | 
| GCL+Conv     |98.6 	|0.26   |6.62 |
| GCL+KPConv   |99.2 	|0.25   |7.50 |

LoKITTI:
| Method | RR | RRE | RTE |
| :- | :-: | - | - |
| FCGF         |22.2    |2.02   |55.2 |
| FCGF+APR     |32.7    |1.74   |51.9 |
| Predator     |42.4    |1.75   |43.4 |
| Predator+APR |50.8    |1.64   |39.5 |
| GCL+Conv     |72.3 	|1.03   |25.9 |
| GCL+KPConv   |55.4 	|1.28   |27.8 |

nuScenes:
| Method | RR | RRE | RTE |
| :- | :-: | - | - |
| FCGF         |93.6   |0.46 	|50.0 |  
| FCGF+APR     |94.5   |0.45 	|37.0 |  
| Predator     |97.8   |0.58 	|20.2 |  
| Predator+APR |99.5   |0.47 	|19.1 |  
| GCL+Conv     |99.2   |0.30    |16.7 |
| GCL+KPConv   | 99.7  |0.35    |15.9 |

LoNuScenes:
| Method | RR | RRE | RTE |
| :- | :-: | - | - |
| FCGF        | 49.1  | 1.30 	|60.9 |
| FCGF+APR    | 51.8  | 1.40 	|62.0 |
| Predator    | 50.4  | 1.47 	|54.5 |
| Predator+APR| 62.7  | 1.30 	|51.8 |
| GCL+Conv    | 82.4  | 0.70    |46.8 |
| GCL+KPConv  | 86.5  | 0.84    | 46.5|

## Requirements

- Ubuntu 14.04 or higher
- CUDA 11.1 or higher
- Python v3.7 or higher
- Pytorch v1.6 or higher
- [MinkowskiEngine](https://github.com/stanfordvl/MinkowskiEngine) v0.5 or higher

## Dataset Preparation

### KITTI

For KITTI dataset preparation, please first follow the [KITTI official instructions](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) to download the 'velodyne laser data', 'calibration files', and (optionally) 'ground truth poses'.

Since the GT poses provided in KITTI drift a lot, we recommend using the pose labels provided by [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download) instead, as they are more accurate. Please follow the official instruction to download the split called 'SemanticKITTI label data'.

Extract all compressed files in the same folder and we are done. We denote KITTI_ROOT as the directory that have the following structure: `{$KITTI_ROOT}/dataset/poses` and `{$KITTI_ROOT}/dataset/sequences/XX`.

The option to use KITTI original pose is still preserved which can be enabled by setting `use_old_pose` to True in the scripts, although we highly discourage doing so due to performance degredation. Please note that all of the methods reported in our paper are retrained on the label of SemanticKITTI instead of OdometryKITTI.

### nuScenes

The vanilla nuScenes dataset structure is not friendly to the registration task, so we propose to convert the lidar part into KITTI format for ease of development and extension. Thanks to the code provided by nuscenes-devkit, the conversion requires only minimal modification.

To download nuScenes, please follow the [nuscenes official page](https://www.nuscenes.org/nuscenes#download) to obtain the 'lidar blobs' (inside 'file blobs') and 'Metadata' of the 'trainval' and 'test' split in the 'Full dataset (v1.0)' section. Only LiDAR scans and pose annotations are used.

Next, execute the following commands to deploy [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) and our conversion script:

```
git clone https://github.com/nutonomy/nuscenes-devkit.git
conda create -n nuscenes-devkit python=3.8
conda activate nuscenes-devkit
pip install nuscenes-devkit
cp ./assets/export_kitti_minimal.py ./nuscenes-devkit/python-sdk/nuscenes/scripts/export_kitti_minimal.py
```

Cater the `nusc_dir` and `nusc_kitti_dir` parameter in `./nuscenes-devkit/python-sdk/nuscenes/scripts/export_kitti_minimal.py` line 51 & 52 to your preferred path. Parameter `nusc_dir` specifies the path to the nuScenes dataset, and `nusc_kitti_dir` will be the path to store the converted nuScenes LiDAR data. Start conversion by executing the following instructions:

```
cd ./nuscenes-devkit/python-sdk
python nuscenes/scripts/export_kitti_minimal.py
```

The process may be slow (can take hours).

## Installation

We recommend conda for installation. First, we need to create a basic environment to setup MinkowskiEngine:

```
conda create -n gcl python=3.7 pip=21.1
conda activate gcl
conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia
pip install numpy
```

Then install [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine) along with other dependencies:

```
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
pip install -r requirements.txt
```

### Setting the distance between two LiDARs (registration difficulty)

As the major focus of this paper, we divide the PCL registration datasets  into different slices according to the distance $d$ between two LiDARs, both during testing and PCL training. Greater $d$ leads to a smaller overlap and more divergent point density, resulting in a higher registration difficulty. We denote range of $d$ with the parameter `--pair_min_dist` and `--pair_max_dist`, which can be found in `./scripts/train_{$method}_{$dataset}.sh`. For example, setting

```
--pair_min_dist 5 \
--pair_max_dist 20 \
```

will set $d\in [5m,20m]$. In other words, for every pair of point clouds, the ground-truth euclidean distance betwen two corresponding LiDAR positions (i.e., the origins of the two specified point clouds) obeys a uniform distribution between 5m and 20m.

### Training suggestions for the baseline: FCGF

Note: This strategy is NOT NEEDED for GCL to converge, but is still necessary for PCL methods like FCGF.

Specifically, we recommend following the two-stage training paradigm for FCGF as pointed out in our paper:

1. Pretrain a model with the following distance parameters: `--pair_min_dist 5  --pair_max_dist 20`. Record the pretrained model path that is printed at the beginning of the training. It shoud be some path like this: `./outputs/Experiments/PairComplementKittiDataset-v0.3/HardestContrastiveLossTrainer//SGD-lr1e-1-e200-b4i1-modelnout32/YYYY-MM-DD_HH-MM-SS`
2. Finetune a new model on `--pair_min_dist 5  --pair_max_dist {$YOUR_SPECIFIC_DISTANCE}`, and paste the pretrained model path to  `--resume "{$PRETRAINED_PATH}/chechpoint.pth"` and `--resume_dir "{$PRETRAINED_PATH}"`. Do not forget to set `--finetune_restart true`.

Emperically, the pretraining strategy helps a lot in FCGF model convergence especially when the distance is large; Otherwise the model just diverges.

### Launch the training

Notes:

1. Remember to set `--use_old_pose` to true when using the nuScenes dataset.
2. When dealing with GCL training, there is no need to alter the `pair_min_dist, pair_max_dist, min_dist, max_dist` parameters. The former two parameters specifies the dataset split used to assess model performance during validation, which will not affect the model itself; The latter two are used to specify the minimum and maximum range to select neighborhood point clouds in GCL, which is our selected optimal parameter.

To train FCGF+GCL on either dataset, run either of the following command inside conda environment `gcl`:

```
./scripts/train_gcl_kitti.sh
./scripts/train_gcl_nuscenes.sh
```

The baseline method FCGF can be trained similarly with our dataset:

```
./scripts/train_fcgf_kitti.sh
./scripts/train_fcgf_nuscenes.sh
```

### Testing

To test FCGF+GCL on either dataset, you can choose to use SC2-PCR to speedup the result with a slight performance increase, by setting `use_RANSAC` to `false`. Do not forget to set  `OUT_DIR` to the specific model path before running the corresponding script inside conda environment `gcl`:

```
./scripts/test_gcl_kitti.sh
./scripts/test_gcl_nuscenes.sh
```

The baseline method FCGF can be tested similarly:

```
./scripts/test_fcgf_kitti.sh
./scripts/test_fcgf_nuscenes.sh
```

Our simple integration of GCL and SC2-PCR achieves 7 FPS inference speed on an RTX 3090 GPU.

### Generalization to ETH

Install pytorch3d by the following command:

```
pip install pytorch3d
```

Then download ETH dataset from the official [website](https://projects.asl.ethz.ch/datasets/doku.php?id=laserregistration:laserregistration), and organize the gazebo_summer, gazebo_winter, wood_autmn, and wood_summer splits in the following structure:

```
--ETH--gazebo_summer
    |--gazebo_winter
    |--wood_autmn
    |--wood_summer
```

Change the `checkpoint` in `generalization_ETH/evaluate.py`, line 235 to your specified model checkpoint path, then run the following commands:

```
cd generalization_eth
python evaluate.py
```

## Acknowlegdements

We thank [FCGF](https://github.com/chrischoy/FCGF) for the wonderful baseline, [SC2-PCR](https://github.com/ZhiChen902/SC2-PCR) for a powerful and fast alternative registration algorithm, and [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) for the convenient dataset conversion code.
