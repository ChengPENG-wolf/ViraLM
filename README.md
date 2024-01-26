# ViraLM

The official implementation of [ViraLM: Language Model empowers virus discovery]


## Contents

- [1. Introduction](#1-introduction)
- [2. Setup Environment](#2-setup-environment)
- [3. Quick Start](#3-quick-start)
- [4. Contact](#4-citation)

## 1. Overview

Viral Language Model (ViraLM) is a Python library for virus identification from metagenomic data.

## 2. Setup environment

*Note*: we suggest you to install all the package using conda ([miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://anaconda.org/)).


Clone the repository to local

```
git clone https://github.com/ChengPENG-wolf/ViraLM.git
cd ViraLM
```

Install package using `mamba`

```
# if you do not have mamba, install it into the conda base environment
conda install conda-forge::mamba

# install and activate environment for ViraLM
mamba env create -f viralm.yaml -n viralm
mamba activate viralm

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
