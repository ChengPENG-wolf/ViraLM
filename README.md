<img src="logo.png" alt="image" width="800" height=auto>

The official implementation of [ViraLM: Empowering Virus Discovery through the Genome Foundation Model]


## Contents

- [1. Introduction](#1-introduction)
- [2. Setup Environment](#2-setup-environment)
- [3. Quick Start](#3-quick-start)
- [4. Contact](#4-citation)

## 1. Overview

Viral Language Model (ViraLM) is a Python library for virus identification from metagenomic data.

## 2. Setup environment

*Note*: we suggest you install all the packages using [mamba](https://github.com/mamba-org/mamba) or [conda](https://docs.conda.io/en/latest/miniconda.html).

```
# clone the repository to the local
git clone https://github.com/ChengPENG-wolf/ViraLM.git
cd ViraLM

# install and activate environment for ViraLM
conda env create -f viralm.yaml -n viralm
conda activate viralm

# download and setup the model
gdown --id 1EQVPmFbpLGrBLU0xCtZBpwvXrtrRxic1
tar -xzvf model.tar.gz -C .
rm model.tar.gz
```

## 3. Quick start
*Note*: we suggest you run ViraLM on GPU.
**Run ViraLM in one command:**

```
python viralm.py [--input INPUT_FA] [--output OUTPUT_PTH] [--len MINIMUM_LEN] [--threshold THRESHOLD]
```

**Options**

```
  --input INPUT_FA
                        The path of your input fasta file
  --output OUTPUT_PTH
			The path of your output diectory
  --len MINIMUM_LEN
                        predict only for sequence >= len bp (default 500)
  --threshold THRESHOLD
                        Threshold to reject (default 0.5).
```

**Example**

Prediction on the example file:

```
python viralm.py --input test.fasta --out result --len 500 --threshold 0.5
```
## 4. Output explanation

The results of ViraLM will be written to `OUTPUT_PTH/result_INPUT_FA_NAME.csv`:

```
seq_name                                             prediction   virus_score           
--------------------------------------------------   ----------   -----------------
IMGVR_UViG_2531839437_000001|2531839437|2531897698   virus        0.501383100927341      
IMGVR_UViG_2529292823_000001|2529292823|2529351314   virus        0.844078302383422
IMGVR_UViG_2531839021_000001|2531839021|2531843197   virus        0.84503074735403
…
```

This tabular file lists all the inputs and ViraLM's prediction on each input:

- `seq_name`: The identifier of the sequence in the input FASTA file.
- `prediction`: The final prediction of the input sequence, virus or non-virus.
- `virus_score`: A value in [0, 1.0], indicates the likelihood of the input sequence being a viral sequence. The larger the more likely to be a virus.


## 5. Contact

If you have any questions, please email us: cpeng29-c@my.cityu.edu.hk
