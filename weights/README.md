## Pre-trained Models

Here, we provide three pre-trained models,

- `771Sep25-0735_500000.params`, the pre-trained *MaskFlownet-S* model, which has been trained on Flying Chairs and Flying Things3D, and it should be evaluated on Sintel *train* + *val* (corresponding to one in step 2).

- `dbbSep30-1206_1000000.params`, the pre-trained *MaskFlownet-S* model, which has been trained on Flying Chairs, Flying Things3D, and Sintel *train*, and it should be evaluated on Sintel *val* (corresponding to one in step 3).

- `5adNov03-0005_1000000.params`, the pre-trained *MaskFlownet* model, which has been trained on Flying Chairs, Flying Things3D, and Sintel *train*, and it should be evaluated on Sintel *val* (corresponding to one in step 6).

## Evaluation

| Network | Checkpoint | Sintel *train* + *val* | Sintel *val* |
|---|---|---|---|
| *MaskFlownet-S* | `771Sep25-0735_500000`  | 2.47, 3.86 | 3.10, 5.54 |
| *MaskFlownet-S* | `dbbSep30-1206_1000000` | - | 2.70, 4.07 |
| *MaskFlownet*   | `5adNov03-0005_1000000` | - | 2.52, 3.83 |

## Inferring

- to do validation for *MaskFlownet-S* on checkpoint `771Sep25-0735_500000.params`, run `python main.py MaskFlownet_S.yaml -g 0 -c 771Sep25 --valid` (the output will be under `./logs/val/`).

- to do prediction for *MaskFlownet* on checkpoint `5adNov03-0005_1000000.params`, run `python main.py MaskFlownet.yaml -g 0 -c 5adNov03 --predict` (the output will be under `./flows/`).
