## Pre-trained Models

Here, we provide three pre-trained models,

- `771Sep25-0735_500000.params`, the pre-trained *MaskFlownet-S* model, which has been trained on Flying Chairs and Flying Things3D, and it should be evaluated on Sintel *train* + *val* (corresponding to one in step 2).

- `dbbSep30-1206_1000000.params`, the pre-trained *MaskFlownet-S* model, which has been trained on Flying Chairs, Flying Things3D, and Sintel *train*, and it should be evaluated on Sintel *val* (corresponding to one in step 3).

- `5adNov03-0005_1000000.params`, the pre-trained *MaskFlownet* model, which has been trained on Flying Chairs, Flying Things3D, and Sintel *train*, and it should be evaluated on Sintel *val* (corresponding to one in step 6).

## Evaluation

<center>

| Network | Checkpoint | Sintel *train* + *val* | Sintel *val* | KITTI 2012 | KITTI 2015 |
|---|---|---|---|---|---|
| *MaskFlownet-S* | `abbSep15-1037_500000`  | 2.33, 3.72 | 2.93, 5.35 | 4.69, 0.20 | 11.88, 0.29 |
| *MaskFlownet-S* | `dbbSep30-1206_1000000` | - | 2.70, 4.07 | 3.25, 0.11 | 9.14, 0.18 |
| *MaskFlownet*   | `5adNov03-0005_1000000` | - | 2.52, 3.83 | 2.85, 0.10 | 8.15, 0.17 |

</center>

for Sintel, the values are `AEPE (clean), AEPE (final)`; for KITTI, the values are `AEPE, FI-all`.

## Inferring

For example,

- to do validation for *MaskFlownet-S* on checkpoint `771Sep25-0735_500000.params`, run `python main.py MaskFlownet_S.yaml -g 0 -c 771Sep25 --valid` (the output will be under `./logs/val/`).

- to do prediction for *MaskFlownet* on checkpoint `5adNov03-0005_1000000.params`, run `python main.py MaskFlownet.yaml -g 0 -c 5adNov03 --predict` (the output will be under `./flows/`).
