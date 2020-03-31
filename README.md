# MaskFlownet: Asymmetric Feature Matching with Learnable Occlusion Mask, CVPR 2020 (Oral)

By Shengyu Zhao, Yilun Sheng, Yue Dong, Eric I-Chao Chang, Yan Xu.

Paper links: [[arXiv]]() [[ResearchGate]]()

```
@inproceedings{zhao2020maskflownet,
  author = {Zhao, Shengyu and Sheng, Yilun and Dong, Yue and Chang, Eric I-Chao and Xu, Yan},
  title = {MaskFlownet: Asymmetric Feature Matching with Learnable Occlusion Mask},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}
```

## Introduction

Feature warping is a core technique in optical flow estimation; however, the ambiguity caused by occluded areas during warping is a major problem that remains unsolved. We propose an asymmetric occlusion-aware feature matching module, which can learn a rough occlusion mask that filters useless (occluded) areas immediately after feature warping without any explicit supervision. The proposed module can be easily integrated into end-to-end network architectures and enjoys performance gains while introducing negligible computational cost. The learned occlusion mask can be further fed into a subsequent network cascade with dual feature pyramids with which we achieve state-of-the-art performance. For more details, please refer to our [paper]().

![mask_visualization](./images/mask_visualization-image.jpg)

This repository includes training and inferring scripts using Python and MXNet. Code has been tested with Python 3.6 and MXNet 1.5.

## Datasets

We follow the common training schedule for optical flow using the following datasets:

- [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html)
- [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [MPI Sintel](http://sintel.is.tue.mpg.de/downloads)
- [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow) & [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
- [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/)

Please modify the paths specified in `main.py` (for FlyingChairs), `reader/things3d.py` (for FlyingThings3D), `reader/sintel.py` (for Sintel), `reader/kitti.py` (for KITTI 2012 & KITTI 2015), and `reader/hd1k.py` (for HD1K) according to where you store the corresponding datasets. Please be aware that the FlyingThings3D dataset (subset) is still very large, so you might want to load only a relatively small proportion of it (see `main.py`).

## Training

The following script is for training:

`python main.py CONFIG [-dataset_cfg DATASET_CONFIG] [-g GPU_DEVICES] [-c CHECKPOINT, --clear_steps] [--debug]`

where `CONFIG` specifies the network and training configuration; `DATASET_CONFIG` specifies the dataset configuration (default to `chairs.yaml`); `GPU_DEVICES` specifies the GPU IDs to use (default to cpu only), split by commas with multi-GPU support. Please make sure that the number of GPUs evenly divides the `BATCH_SIZE`, which depends on the `DATASET_CONFIG` (`BATCH_SIZE` are `8` or `4` in the given configurations, so `4`, `2`, or `1` GPU(s) will be fine); `CHECKPOINT` specifies the previous checkpoint to start with; use `--clear_steps` to clear the step history and start from step 0; use `--debug` to enter the DEBUG mode, where only a small fragment of the data is read.

to test whether your environment has been set up properly, run: `python main.py MaskFlownet.yaml -g 0 --debug`

Here, we present the procedure to train a complete *MaskFlownet* model for validation on Sintel dataset. About 20% sequences (ambush_2, ambush_6, bamboo_2, cave_4, market_6, temple_2) are split as Sintel *val*, while the remaining are left as Sintel *train* (see `Sintel_train_val_maskflownet.txt`). The `CHECKPOINT` in each command line should correspond to the one generated in the previous step.

| # | Network         | Training         | Validation     | Command Line |
|---|---|---|---|---|
| 1 | *MaskFlownet-S* | Flying Chairs    | Sintel *train* + *val* | `python main.py MaskFlownet_S.yaml -g 0,1,2,3` |
| 2 | *MaskFlownet-S* | Flying Things3D  | Sintel *train* + *val* | `python main.py MaskFlownet_S_ft.yaml --dataset_cfg things3d.yaml -g 0,1,2,3 -c [CHECKPOINT] --clear_steps` |
| 3 | *MaskFlownet-S* | Sintel *train* + KITTI 2015 + HD1K | Sintel *val* | `python main.py MaskFlownet_S_sintel.yaml --dataset_cfg sintel_kitti2015_hd1k.yaml -g 0,1,2,3 -c [CHECKPOINT] --clear_steps` |
| 4 | *MaskFlownet*   | Flying Chairs    | Sintel *val* | `python main.py MaskFlownet.yaml -g 0,1,2,3 -c [CHECKPOINT] --clear_steps` |
| 5 | *MaskFlownet*   | Flying Things3D  | Sintel *val* | `python main.py MaskFlownet_ft.yaml --dataset_cfg things3d.yaml -g 0,1,2,3 -c [CHECKPOINT] --clear_steps` |
| 6 | *MaskFlownet*   | Sintel *train* + KITTI 2015 + HD1K | Sintel *val* | `python main.py MaskFlownet_sintel.yaml --dataset_cfg sintel_kitti2015_hd1k.yaml -g 0,1,2,3 -c [CHECKPOINT] --clear_steps` |

## Inferring

The following script is for inferring:

`python main.py CONFIG [-g GPU_DEVICES] [-c CHECKPOINT] [--valid or --predict] [--resize INFERENCE_RESIZE]`

where `CONFIG` specifies the network configuration (`MaskFlownet_S.yaml` or `MaskFlownet.yaml`); `GPU_DEVICES` specifies the GPU IDs to use, split by commas with multi-GPU support; `CHECKPOINT` specifies the checkpoint to do inference on; use `--valid` to do validation; use `--predict` to do prediction; `INFERENCE_RESIZE` specifies the resize used to do inference.

For example,

- to do validation for *MaskFlownet-S* on checkpoint `fffMar16`, run `python main.py MaskFlownet_S.yaml -g 0 -c fffMar16 --valid` (the output will be under `./logs/val/`).

- to do prediction for *MaskFlownet* on checkpoint `000Mar17`, run `python main.py MaskFlownet.yaml -g 0 -c 000Mar17 --predict` (the output will be under `./flows/`).

## Acknowledgement

We thank Tingfung Lau for the initial implementation of the FlyingChairs pipeline.
