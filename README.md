# Stage-Aware Spatial–Temporal Placement for Skeleton-Based Action Recognition

Official implementation of the paper:

**Stage-Aware Spatial-Temporal Placement for Skeleton-Based Action Recognition**

This repository is built on a 2s-AGCN-style training and evaluation pipeline. The main purpose of this release is to provide a minimal reproducible implementation of the proposed stage-aware skeleton action recognition model, including data processing, training/testing code, model definition, and configuration files for NTU RGB+D 60 and Kinetics-Skeleton.

# Note

This repository follows the general training style of 2s-AGCN. The major difference is that the proposed model introduces a stage-aware design principle:

- shallow layers use multi-granularity spatial priors;
- middle layers keep stable GCN-TCN abstraction;
- deep layers use lightweight selective temporal-channel gating.

The code release currently includes:

- data processing scripts;
- `main.py` for training and testing;
- the main model file;
- configuration files for NTU RGB+D 60 and Kinetics-Skeleton;
- instructions for reproducing ablation settings and visualization results.

The original datasets are not redistributed in this repository. Please download NTU RGB+D 60 and Kinetics-Skeleton from their official sources.

# Requirements

The code was tested with:

```
Python 3.8
PyTorch 1.13.1
CUDA 11.7
NumPy
PyYAML
tqdm
tensorboardX
scikit-learn
matplotlib
```

Install dependencies with:

```
pip install -r requirements.txt
```

If your CUDA or PyTorch version is different, please install the corresponding PyTorch version from the official PyTorch website.

# Data Preparation

 - ## NTU RGB+D 60

     Download the raw NTU RGB+D 60 skeleton data from the official dataset provider and put the files under the `data` directory.
     
     The expected directory structure is:
     
     ```
     data/
     └── nturgbd_raw/
         ├── nturgb+d_skeletons/
         │   ├── S001C001P001R001A001.skeleton
         │   ├── ...
         │   └── ...
         └── samples_with_missing_skeletons.txt
     ```
     
     Preprocess the NTU RGB+D 60 data with:
     
     ```
     python data_gen/ntu_gendata.py
     ```
     
     After preprocessing, the expected processed data structure is:
     
     ```
     data/
     └── ntu/
         ├── xsub/
         │   ├── train_data_joint.npy
         │   ├── train_label.pkl
         │   ├── val_data_joint.npy
         │   └── val_label.pkl
         │
         └── xview/
             ├── train_data_joint.npy
             ├── train_label.pkl
             ├── val_data_joint.npy
             └── val_label.pkl
     ```
     
     Generate bone data with:
     
     ```
     python data_gen/gen_bone_data.py
     ```
     
     The generated bone files should be placed under the corresponding protocol folders, for example:
     
     ```
     data/
     └── ntu/
         ├── xsub/
         │   ├── train_data_bone.npy
         │   └── val_data_bone.npy
         └── xview/
             ├── train_data_bone.npy
             └── val_data_bone.npy
     ```
     
     ## Kinetics-Skeleton
     
     Download the Kinetics-Skeleton data following the standard ST-GCN / 2s-AGCN format.
     
     The expected raw data structure is:
     
     ```
     data/
     └── kinetics_raw/
         ├── kinetics_train/
         │   ├── ...
         │   └── ...
         ├── kinetics_val/
         │   ├── ...
         │   └── ...
         ├── kinetics_train_label.json
         └── kinetics_val_label.json
     ```
     
     Preprocess the Kinetics-Skeleton data with:
     
     ```
     python data_gen/kinetics-gendata.py
     ```
     
     The processed data should follow the format:
     
     ```
     data/
     └── kinetics/
         ├── train_data_joint.npy
         ├── train_label.pkl
         ├── val_data_joint.npy
         └── val_label.pkl
     ```

# Training & Testing

Change the configuration file depending on the dataset, protocol, stream, and model variant.


    `python main.py --config ./config/nturgbd-cross-view/train_joint_mg_stgm.yaml`
    
    `python main.py --config ./config/nturgbd-cross-view/train_bone_mg_stgm.yaml`
To ensemble the results of joints and bones, run test firstly to generate the scores of the softmax layer. 

    `python main.py --config ./config/nturgbd-cross-view/test_joint_mg_stgm.yaml`
    
    `python main.py --config ./config/nturgbd-cross-view/test_bone_mg_stgm.yaml`

Then combine the generated scores with: 

    `python ensemble.py --datasets ntu/xview`

# Model Configuration

The proposed model is controlled mainly by the following options in the configuration file:

```
model_args:
  use_mg: True
  use_stgm: True
  stgm_layers: [9, 10]
  mg_mode: learned
  mg_fusion: dynamic_subset
  gate_reduction: 8
  gate_dropout: 0.1
  pa_lambda: 0.1
```

The baseline AGCN-style model can be obtained by disabling the proposed components:

```
model_args:
  use_mg: False
  use_stgm: False
```

The final stage-aware setting is:

```
L1-L3: shallow stage, multi-granularity spatial priors
L4-L8: middle stage, standard GCN-TCN abstraction
L9-L10: deep stage, selective temporal-channel gating
```

# Ablation Settings

The ablation experiments in the paper can be reproduced by modifying the model arguments in the provided configuration files.

## Spatial Granularity Ablation

Use the following settings:

```
use_mg: True
use_stgm: False
```

Then change `mg_mode`:

```
mg_mode: joint
```

or:

```
mg_mode: limb
```

or:

```
mg_mode: global
```

or:

```
mg_mode: learned
```

The main variants are:

| Variant           | Key setting                                                  |
| ----------------- | ------------------------------------------------------------ |
| Baseline          | `use_mg: False`, `use_stgm: False`                           |
| Joint-level only  | `use_mg: True`, `mg_mode: joint`, `use_stgm: False`          |
| Limb-level only   | `use_mg: True`, `mg_mode: limb`, `use_stgm: False`           |
| Global-level only | `use_mg: True`, `mg_mode: global`, `use_stgm: False`         |
| Dynamic fusion    | `use_mg: True`, `mg_mode: learned`, `mg_fusion: dynamic_subset`, `use_stgm: False` |

## Temporal Gating Placement Ablation

Use the following settings:

```
use_mg: False
use_stgm: True
```

Then change `stgm_layers`:

```
stgm_layers: [8]
stgm_layers: [9]
stgm_layers: [10]
stgm_layers: [9, 10]
stgm_layers: [8, 9, 10]
```

The final model uses:

```
stgm_layers: [9, 10]
```

## Module Combination Ablation

The following variants are used to analyze the complementary effects of the spatial and temporal components:

| Variant    | Key setting                                               |
| ---------- | --------------------------------------------------------- |
| Baseline   | `use_mg: False`, `use_stgm: False`                        |
| MG only    | `use_mg: True`, `use_stgm: False`                         |
| STGM only  | `use_mg: False`, `use_stgm: True`, `stgm_layers: [9, 10]` |
| Full model | `use_mg: True`, `use_stgm: True`, `stgm_layers: [9, 10]`  |

# Visualization and Diagnostic Analysis

The manuscript reports several diagnostic visualizations, including:

- t-SNE feature visualization;
- class-wise recall comparison;
- STGM gate-response maps;
- layer-wise feature separability;
- sample-wise granularity preference.

These analyses are generated from saved checkpoints, prediction scores, and extracted intermediate features.

## STGM Gate Visualization

Gate values are extracted from the sigmoid output of the temporal gating branch:

```
gate = sigmoid(dt)
```

For visualization, gate tensors are reshaped and averaged over persons and joints to obtain temporal-channel gate maps.

Recommended settings:

```
Dataset: NTU RGB+D 60
Protocol: X-Sub or X-View
Stream: joint
Layers: L9 and L10
Representative actions: pickup, clapping, writing, take off a shoe
Displayed channels: top-60 channels ranked by temporal variance
```

## t-SNE Visualization

For t-SNE visualization:

1. load the trained checkpoint;
2. extract deep features before the classifier, usually from L10;
3. select representative confusing action classes;
4. apply L2 normalization before t-SNE;
5. compute silhouette score in the original feature space;
6. use t-SNE only for visualization.

## Class-Wise Recall

Class-wise recall is computed from the diagonal entries of the normalized confusion matrix.

The visualization should report:

```
protocol
class names
baseline recall
ours recall
absolute recall change
```

# Expected Results

Please refer to the paper for the full experimental results. Due to randomness in training, GPU environment, and preprocessing details, small numerical differences may occur.

Recommended reporting format:

```
mean accuracy ± standard deviation over multiple random seeds
```

Example result table format:

| Dataset           | Protocol | Stream | Method   | Accuracy     |
| ----------------- | -------- | ------ | -------- | ------------ |
| NTU RGB+D 60      | X-Sub    | J+B  | Baseline | 88.64±0.44   |
| NTU RGB+D 60      | X-Sub    | J+B  | Ours     | 88.65±0.04   |
| NTU RGB+D 60      | X-View   | J+B  | Baseline | 94.96±0.11   |
| NTU RGB+D 60      | X-View   | J+B  | Ours     | 95.13±0.12   |
| Kinetics-Skeleton | Val      | J+B  | Baseline | 36.66±0.13   |
| Kinetics-Skeleton | Val      | J+B  | Ours     | 36.23±0.34   |

# Citation

Please cite the following paper if you use this repository in your reseach.

    @article{Yang2026StageAware,
  title   = {Stage-Aware Spatial-Temporal Placement for Skeleton-Based Action Recognition},
  author  = {Yang, Yuanyuan and Zhang, Zhengwen and
             Yang, Cheng and Yu, Yangtao and
             Lonapalawong, Supaporn},
  journal = {Pattern Recognition Letters},
  year    = {2026},
  doi     = {10.1016/j.patrec.2026.10.020}
}
# Acknowledgement

This repository follows the general training and data-processing style of 2s-AGCN. We thank the authors of 2s-AGCN and ST-GCN for releasing their code and preprocessing pipelines.

# Contact

For questions about this repository, please contact:

```
Name: Yuanyuan Yang
Email: 24213037560005@ymu.edu.cn
```
