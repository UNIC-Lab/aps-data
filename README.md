# Map2APS

Code release for the Map2APS benchmark: direct angle power spectrum (APS)
prediction from urban geometry and transmitter/receiver locations.

This repository contains only the methods used in the paper. Data files,
checkpoints, generated figures, and detailed result CSVs are not included.

## Methods

| Paper name | Code path |
| --- | --- |
| MS-AReg | `baselines/ms_areg/` |
| Adv-MLP | `baselines/adv_mlp/` |
| MS-MLP | `baselines/ms_mlp/` |
| RadioUNet | `baselines/radiounet/` |
| ResNet-MLP | `baselines/resnet_mlp/` |
| ResNet-Conv1D | `baselines/resnet_conv1d/` |
| ViT-Reg | `baselines/vit_reg/` |

## Data

Dataset link:

```text
Baidu Netdisk: TBD
```

Expected raw data layout for preprocessing:

```text
<raw_root>/
  map_0/0_adps/aps/aps_*.npy
  map_1/1_adps/aps/aps_*.npy
  ...
<buildings_dir>/
  0.json
  1.json
  ...
```

Preprocessed samples are saved as `.npz` files under:

```text
<processed_dir>/samples/
```

Each sample stores:

- `aps_norm`: normalized APS vector with shape `(180,)`
- `cond_img`: condition image with shape `(3, 256, 256)`

## Setup

```bash
pip install -r requirements.txt
```

The experiments were developed with PyTorch and CUDA GPUs. Before training,
edit each YAML config so that `data.preprocessed_dir` points to your
preprocessed dataset and `logging.checkpoint_dir` points to your output
directory.

## Preprocess

```bash
python data/preprocess.py \
  --aps_root <raw_root> \
  --buildings_dir <buildings_dir> \
  --output_dir <processed_dir> \
  --aps_norm std
```

Optional map subset:

```bash
python data/preprocess.py \
  --aps_root <raw_root> \
  --buildings_dir <buildings_dir> \
  --output_dir <processed_dir> \
  --map_ids 0,1,2 \
  --aps_norm std
```

## Train

```bash
python baselines/ms_areg/train_ms_areg.py --config baselines/ms_areg/config_ms_areg.yaml --gpu 0
python baselines/adv_mlp/train_adv_mlp.py --config baselines/adv_mlp/config_adv_mlp.yaml --gpu 0
python baselines/ms_mlp/train_ms_mlp.py --config baselines/ms_mlp/config_ms_mlp.yaml --gpu 0
python baselines/radiounet/train_radiounet.py --config baselines/radiounet/config_radiounet.yaml --gpu 0
python baselines/resnet_mlp/train_resnet_mlp.py --config baselines/resnet_mlp/config_resnet_mlp.yaml --gpu 0
python baselines/resnet_conv1d/train_resnet_conv1d.py --config baselines/resnet_conv1d/config_resnet_conv1d.yaml --gpu 0
python baselines/vit_reg/train_vit_reg.py --config baselines/vit_reg/config_vit_reg.yaml --gpu 0
```

## Evaluate

```bash
python eval_compare.py \
  --data_dir <processed_dir> \
  --mode full \
  --output_dir <output_dir> \
  --models \
    ms_areg:<path_to_ms_areg_checkpoint> \
    adv_mlp:<path_to_adv_mlp_checkpoint> \
    ms_mlp:<path_to_ms_mlp_checkpoint> \
    radiounet:<path_to_radiounet_checkpoint> \
    resnet_mlp:<path_to_resnet_mlp_checkpoint> \
    resnet_conv1d:<path_to_resnet_conv1d_checkpoint> \
    vit_reg:<path_to_vit_reg_checkpoint>
```

The default train/test split is hard-coded in `data/dataset.py`: maps `0-45`
for training and maps `46-50` for held-out evaluation.
