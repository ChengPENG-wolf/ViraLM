<img src="logo.png" alt="image" width="800" height="267">

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
# mamba env create -f viralm.yaml -n viralm
conda activate viralm
# mamba activate viralm

# download and setup the model
gdown --id 1EQVPmFbpLGrBLU0xCtZBpwvXrtrRxic1
tar -xzvf model.tar.gz -C .
rm model.tar.gz
```

## 3. Quick start

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

## 4. Contact

If you have any questions, please email us: cpeng29-c@my.cityu.edu.hk
