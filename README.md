# BOLD-Cast: Modeling Individual-Level Long-Range Brain Dynamics from Short fMRI Scans

BOLD-Cast is a two-stage deep-learning framework for modeling long-range
individual brain dynamics from short resting-state fMRI scans.

The framework contains:

-   **Stage I --- Common/Private Representation Learning:** learns
    cohort-shared (`com`) and subject-specific (`priv`) latent
    representations from parcel-level fMRI. After Stage I training, the
    best checkpoint is automatically loaded and `com`/`priv` are
    extracted for **all train, validation, and test subjects** and
    written directly back into each subject-level `.npz` file.
-   **Stage II --- Long-Range BOLD Forecasting:** predicts future fMRI
    dynamics from the observed BOLD sequence. For BOLD-Cast, Stage-I
    `com` and `priv` can optionally be concatenated as `latent_embed`
    and fused into the forecasting model.

The current implementation uses the **Craddock CC200 atlas**. After ROI
quality control, the experiments in this repository use **190 ROIs**.

------------------------------------------------------------------------

## Contents

1.  [Environment Installation](#1-environment-installation)
2.  [Data Application and Download](#2-data-application-and-download)
3.  [Data Organization](#3-data-organization)
4.  [Download GPT-2](#4-download-gpt-2)
5.  [Data Preprocessing](#5-data-preprocessing)
6.  [Pretrained Checkpoints](#6-pretrained-checkpoints)
7.  [Test Stage I](#7-test-stage-i)
8.  [Test Stage II](#8-test-stage-ii)
9.  [Train on Your Own Data](#9-train-on-your-own-data)
10. [Evaluation](#10-evaluation)
11. [Acknowledgements](#11-acknowledgements)
12. [Contact](#12-contact)

------------------------------------------------------------------------

## 1. Environment Installation

### 1.1 Clone the repository

``` bash
git clone https://github.com/CUHK-AIM-Group/BOLD-Cast.git
cd BOLD-Cast
```

### 1.2 Create the environment

The main experiments were developed with:

-   Python 3.10
-   CUDA 11.8
-   PyTorch 2.0.1
-   NVIDIA GPU recommended for training

A typical Conda setup is:

``` bash
conda create -n boldcast python=3.10 -y
conda activate boldcast
pip install -r requirements.txt
```

Please ensure that the installed PyTorch build is compatible with your
local CUDA driver.

------------------------------------------------------------------------

## 2. Data Application and Download

The original neuroimaging datasets are **not redistributed in this
repository** because their use is governed by the corresponding data
providers. Users should apply for/download the datasets from the
official sources and comply with the applicable data-use terms.

  -----------------------------------------------------------------------------------------------------------------------------------------
  Dataset key             Dataset                 Official access
  ----------------------- ----------------------- -----------------------------------------------------------------------------------------
  `ukb`                   UK Biobank              [UK Biobank --- Apply for
                                                  Access](https://www.ukbiobank.ac.uk/use-our-data/apply-for-access/)

  `hcpya`                 Human Connectome        [HCP Young Adult Data
                          Project --- Young Adult Releases](https://humanconnectome.org/study/hcp-young-adult/data-releases)

  `hcpd`                  Human Connectome        [HCP Development Data
                          Project --- Development Releases](https://www.humanconnectome.org/study/hcp-lifespan-development/data-releases)

  `hcpa`                  Human Connectome        [HCP Aging Data
                          Project --- Aging       Releases](https://hcp-db.humanconnectome.org/study/hcp-lifespan-aging/data-releases)

  `abide`                 Autism Brain Imaging    [ABIDE I](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html) / [ABIDE
                          Data Exchange           Preprocessed](https://preprocessed-connectomes-project.org/abide/download.html)
  -----------------------------------------------------------------------------------------------------------------------------------------

For ABIDE, the Preprocessed Connectomes Project provides several
derivatives, including **CC200 ROI time series**.

------------------------------------------------------------------------

## 3. Data Organization

### 3.1 Raw/prepared ROI-level inputs

Before running the unified preprocessing script, prepare one ROI
time-series file per subject using the CC200 atlas.

The default directories used by `BOLDCast_preprocess.py` are:

``` text
BOLD-Cast/
├── dataset/
│   ├── UKB_roi/             # UK Biobank ROI time series
│   ├── HCP-YA_roi/          # HCP Young Adult
│   ├── HCP-D_roi/           # HCP Development
│   ├── HCP-A_roi/           # HCP Aging
│   ├── ABIDE_roi/           # ABIDE
│   │
│   ├── UKB.csv             # optional phenotype file
│   ├── HCP-YA.csv
│   ├── HCP-D.csv
│   ├── HCP-A.csv
│   └── ABIDE.csv
│
├── Stage I/
├── Stage II/
├── BOLDCast_preprocess.py
├── requirements.txt
└── README.md
```

Private/custom paths can be supplied with `--roi_dir` and
`--phenotype_csv`.

### 3.2 Generated model inputs

`BOLDCast_preprocess.py` automatically divides subjects into
train/validation/test splits and creates:

``` text
dataset/
├── UKB_input/
│   ├── ts/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── sp/
│       ├── train/
│       ├── val/
│       └── test/
├── HCP-YA_input/
├── HCP-D_input/
├── HCP-A_input/
└── ABIDE_input/
```

Each subject in `ts/{train,val,test}` is stored as one `.npz` file. The
core keys are:

``` text
fMRI
text
corr
```

Dataset-specific phenotype/cognitive keys may also be present, for
example:

``` text
sex
ASD
ReadEng
ProcSpeed
...
```

After Stage I has been trained, two additional keys are written directly
into the same subject `.npz`:

``` text
com
priv
```

Stage II therefore reads the same subject-level files generated by
preprocessing and enriched by Stage I.

------------------------------------------------------------------------

## 4. Download GPT-2

Stage II uses a local GPT-2 checkpoint for timestamp/text embedding.

Download the official GPT-2 model from Hugging Face:

[openai-community/gpt2](https://huggingface.co/openai-community/gpt2)

Place the downloaded files under:

``` text
BOLD-Cast/
└── Stage II/
    └── gpt2/
        ├── config.json
        ├── model.safetensors / pytorch_model.bin
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── vocab.json
        ├── merges.txt
        └── ...
```

The default Stage-II path is:

``` text
Stage II/gpt2
```

A different local checkpoint can be supplied with:

``` bash
--llm_ckp_dir /path/to/gpt2
```

------------------------------------------------------------------------

## 5. Data Preprocessing

All preprocessing required by Stage I and Stage II is launched from the
repository root with a **single script**:

``` bash
python BOLDCast_preprocess.py --dataset DATASET
```

`DATASET` can be:

``` text
UKB
HCP-YA
HCP-D
HCP-A
ABIDE
```

Examples:

``` bash
python BOLDCast_preprocess.py --dataset UKB
python BOLDCast_preprocess.py --dataset HCP-YA
python BOLDCast_preprocess.py --dataset HCP-D
python BOLDCast_preprocess.py --dataset HCP-A
python BOLDCast_preprocess.py --dataset ABIDE
```

The unified preprocessing pipeline performs the following operations in
order:

``` text
ROI-level subject data
        ↓
train / val / test subject split
        ↓
timestamp + subject-level NPZ generation
        ↓
dataset/<dataset>_input/ts/
        ↓
Stage-I input validation
        ↓
GPT-2 timestamp/text embedding
        ↓
dataset/<dataset>_input/sp/
```

The default Stage-I self-reconstruction lengths are:

  Dataset                       `--time_len`
  --------------------------- --------------
  UK Biobank (`ukb`)                      81
  HCP Young Adult (`hcpya`)               83
  HCP Development (`hcpd`)                85
  HCP Aging (`hcpa`)                      75
  ABIDE (`abide`)                         30

These values can be overridden during preprocessing, for example:

``` bash
python BOLDCast_preprocess.py --dataset ABIDE --time_len 30
```

or:

``` bash
python BOLDCast_preprocess.py --dataset HCP-YA --time_len 83
```

The train/validation/test split ratios and random seed can also be
controlled from the preprocessing CLI.

------------------------------------------------------------------------

## 6. Pretrained Checkpoints

Pretrained weights for both stages are provided separately because the
checkpoint files are too large for the repository.

### Stage I checkpoints

[Download Stage I checkpoints from Google
Drive](https://drive.google.com/file/d/1k2NX-WmFCzWXCBcLmBu2FYo5xeaI_DRi/view?usp=sharing)

Place the downloaded checkpoints under:

``` text
Stage I/
└── checkpoints/
    ├── UKB_best_model.pth
    ├── HCP-YA_best_model.pth
    ├── HCP-D_best_model.pth
    ├── HCP-A_best_model.pth
    └── ABIDE_best_model.pth
```

### Stage II checkpoints

[Download Stage II checkpoints from Google
Drive](https://drive.google.com/file/d/1nbke5ZifaeKS-0NwXGwLbsgL7IOC2Z4X/view?usp=sharing)

Place the downloaded checkpoints under:

``` text
Stage II/
└── checkpoints/
    └── <model>_<dataset>_best_model.pth
```

For example:

``` text
BOLDCast_HCP-YA_best_model.pth
BOLDCast_ABIDE_best_model.pth
```

------------------------------------------------------------------------

## 7. Test Stage I

Stage I learns common and private latent representations using
self-reconstruction:

``` text
x = fMRI[:, :time_len]
y = x
```

No future BOLD segment is used as the Stage-I reconstruction target.

### 7.1 Use a pretrained Stage-I checkpoint

When a pretrained `{dataset}_best_model.pth` already exists, latent
extraction can be run without retraining by setting `--num_iters 0`.

From the repository root:

``` bash
python "Stage I/main.py" --dataset UKB   --time_len 81 --num_iters 0

```

Other examples:

``` bash
python "Stage I/main.py" --dataset HCP-YA --time_len 83 --num_iters 0
python "Stage I/main.py" --dataset HCP-D  --time_len 85 --num_iters 0
python "Stage I/main.py" --dataset HCP-A  --time_len 75 --num_iters 0
python "Stage I/main.py" --dataset ABIDE  --time_len 30 --num_iters 0
```

Stage I loads:

``` text
Stage I/checkpoints/<dataset>_best_model.pth
```

and extracts `com` and `priv` for **all subjects in train, val, and
test**.

The extracted latent representations are written directly back into:

``` text
dataset/<dataset>_input/ts/{train,val,test}/<subject>.npz
```

No separate latent file is required.

> **Important:** Once `com` and `priv` have already been written into
> every subject `.npz`, Stage I does not need to be run again before
> Stage II.

------------------------------------------------------------------------

## 8. Test Stage II

### 8.1 Supported datasets

Stage II supports:

``` text
UKB
HCP-YA
HCP-D
HCP-A
ABIDE
```

### 8.2 BOLD-Cast

The Stage-II temporal dimensions are explicit command-line
hyperparameters:

``` text
--seq_len
--label_len
--pred_len
```

The current defaults are:

``` text
seq_len   = 162
label_len = 81
pred_len  = 81
```

They can be changed directly and are not hard-coded to a dataset.

Test BOLD-Cast using a pretrained checkpoint:

``` bash
python "Stage II/run.py" --dataset hcpya --model BOLDCast --is_training false
```

### 8.3 Baseline models

The current Stage-II code includes the following comparison models:

``` text
BrainTransformer
DLinear
ForecastGrapher
FourierGNN
One_Fit_All
iTransformer
LightTS
MSGNet
PatchTST
SimMTM
TSMixer
```

Select a baseline with `--model`. For example:

``` bash
python "Stage II/run.py" --dataset UKB --model DLinear --is_training false
```

or:

``` bash
python "Stage II/run.py" --dataset HCP-YA --model PatchTST --is_training false
```

Baseline models do **not** require the Stage-I `com`/`priv` inputs.

If the temporal dimensions differ from the defaults, specify them
explicitly:

``` bash
python "Stage II/run.py" \
    --dataset UKB \
    --model BOLDCast \
    --is_training false \
    --seq_len 162 \
    --label_len 81 \
    --pred_len 81 \
    --mix_embeds
```

------------------------------------------------------------------------

## 9. Train on Your Own Data

The recommended workflow is:

``` text
Prepare CC200 ROI time series
        ↓
BOLDCast_preprocess.py
        ↓
Stage I training
        ↓
best Stage-I checkpoint
        ↓
automatic com/priv extraction for ALL subjects
        ↓
com/priv written into subject NPZ files
        ↓
Stage II training
```

### 9.1 Step 1 --- Prepare your data

Convert each subject's preprocessed resting-state fMRI into parcel-level
time series compatible with the repository.

For the current CC200 configuration, the expected ROI dimension is 190
after the ROI filtering/QC used in this project.

Place the files in the appropriate dataset ROI directory, or provide a
custom path:

``` bash
python BOLDCast_preprocess.py \
    --dataset your_data \
    --roi_dir /path/to/your/roi_data \
    --phenotype_csv /path/to/your/phenotype.csv
```

If your data correspond to one of the five supported dataset formats,
use its dataset key so the appropriate metadata/timestamp handling is
selected.

### 9.2 Step 2 --- Run preprocessing

Example:

``` bash
python BOLDCast_preprocess.py --dataset your_data --time_len XX
```

### 9.3 Step 3 --- Train Stage I

Example:

``` bash
python "Stage I/main.py" --dataset your_data --time_len XX
```

Stage I uses the original `train + test` subjects for unsupervised
representation learning and uses the validation split only for
checkpoint selection based on reconstruction loss.

The best checkpoint is saved as:

``` text
Stage I/checkpoints/<dataset>_best_model.pth
```

The training logic is:

``` text
validation reconstruction improves
        → save best checkpoint

3 consecutive non-improving validations
        → reduce learning rate once

10 consecutive non-improving validations
        → early stop
```

After training, the best checkpoint is loaded automatically and Stage I
extracts `com` and `priv` for **train, val, and test subjects**.

These two keys are written back into every subject-level `.npz`.

Therefore, after Stage I finishes successfully, **you do not need to
load the Stage-I checkpoint again to train Stage II**. Stage II can
directly read the saved `com` and `priv` values from the processed
`.npz` files.

### 9.4 Step 4 --- Train Stage II

Train BOLD-Cast without Stage-I latent fusion:

``` bash
python "Stage II/run.py" \
    --dataset your_data \
    --model BOLDCast \
    --is_training true \
    --seq_len 162 \
    --label_len 81 \
    --pred_len 81 \
```
The best Stage-II checkpoint is stored under:

``` text
Stage II/checkpoints/<model>_<dataset>_best_model.pth
```

------------------------------------------------------------------------

## 10. Evaluation

The forecasting evaluation includes metrics such as:

-   MAE
-   RMSE
-   MAPE
-   mPCC
-   FC-PCC
-   FN

### 11. Downstream Evaluation

To evaluate whether the generated BOLD sequences preserve task-relevant functional information, we further conduct downstream brain-network analyses, including ASD classification, sex identification, and cognitive score prediction. The downstream experiments are implemented based on the following publicly available brain-network analysis frameworks:

- **BrainGB** — *BrainGB: A Benchmark for Brain Network Analysis with Graph Neural Networks*  
  - GitHub: https://github.com/HennyJie/BrainGB
  - Paper: https://doi.org/10.1109/TMI.2022.3218745

- **PTGB** — *PTGB: Pre-Train Graph Neural Networks for Brain Network Analysis*  
  - GitHub: https://github.com/Owen-Yang-18/BrainNN-PreTrain
  - Paper: https://proceedings.mlr.press/v209/yang23a.html

  - BrainGB provides a standardized GNN-based framework for brain-network construction and downstream prediction, while PTGB provides a brain-network-specific GNN pre-training framework that can be adapted to downstream classification and regression tasks. Please refer to their original repositories and publications for implementation details and citation information.
------------------------------------------------------------------------

## 12. Acknowledgements

We thank the authors and contributors of the following projects:

-   [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
-   [Hugging Face Transformers](https://github.com/huggingface/transformers)

We also thank the UK Biobank, Human Connectome Project, ABIDE/INDI, and
Preprocessed Connectomes Project teams and participants for making
neuroimaging resources available to the research community.

------------------------------------------------------------------------

## 12. Contact

For questions regarding the code or manuscript, please contact:

**Yu Jiang**\
yuajiang@cuhk.edu.hk

If you use this repository in your research, please cite the
corresponding BOLD-Cast paper.
