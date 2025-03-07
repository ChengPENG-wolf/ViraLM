<img src="logo.png" alt="image" width="800" height=auto>

## The official implementation of [ViraLM: Empowering Virus Discovery through the Genome Foundation Model](https://doi.org/10.1093/bioinformatics/btae704)

![GitHub License](https://img.shields.io/github/license/ChengPENG-wolf/ViraLM)

## News

🎉 ViraLM is accepted by [Bioinformatics](https://doi.org/10.1093/bioinformatics/btae704)!

## Contents

- [1. Introduction](#1-introduction)
- [2. Setup Environment](#2-setup-environment)
- [3. Quick Start](#3-quick-start)
- [4. Output explanation](#4-output-explanation)
- [5. Citation](#5-citation)
- [6. Contact](#6-contact)

## 1. Overview

Viral Language Model (ViraLM) is a Python library for virus identification from metagenomic data. ViraLM employs the latest genome foundation model to capture complex genomic characteristics and is able to distinguish viral genomes from organisms.

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
python viralm.py [-i INPUT_FA] [-o OUTPUT_PTH] [-d DATABASE_PATH] [-l MINIMUM_LEN] [-t THRESHOLD]
```

**Options**

*Note*: we recommend that `MINIMUM_LEN` be larger than 500 for reliable performance.

```
  --input INPUT_FA
                        The name of your input file (FASTA format)
  --output OUTPUT_PTH
                        The path of the output directory
  --filename FILENAME
                        Custom name for output files (option)
  --database DATABASE
                        Model directory
  --len LEN
                        Predict only for sequences >= len bp (default: 500)
  --batch_size BATCH_SIZE
                        Batch size for prediction (default: 16)
  --threshold THRESHOLD
                        Threshold for prediction (default: 0.5)
  --force
                        Force overwrite of the output directory if it exists (option)
```

**Example**

Prediction on the example file:

```
export CUDA_VISIBLE_DEVICES=0,1,2,...,n 	# (option) n is the number of GPUs
python viralm.py --input test.fasta --out result --len 500 --threshold 0.5
```

If you prefer storing your models/databases in a different location, then you can use
`-d` or `--db` parameter:

```
python viralm.py --input test.fasta --out result -d /path/database/downloaded --len 500 --threshold 0.5
```
## 4. Output explanation

#### 1. `OUTPUT_PTH`/result_`INPUT_FA_NAME`.csv:

```
seq_name                                             prediction   virus_score           
--------------------------------------------------   ----------   -----------------
IMGVR_UViG_2531839437_000001|2531839437|2531897698   virus        0.845030747354031      
IMGVR_UViG_2529292823_000001|2529292823|2529351314   virus        0.844078302383422
IMGVR_UViG_2531839021_000001|2531839021|2531843197   virus        0.501383100927341
…
```

This tabular file lists all the inputs and ViraLM's prediction on each input:

- `seq_name`: The identifier of the sequence in the input FASTA file.
- `prediction`: The final prediction of the input sequence, virus or non-virus.
- `virus_score`: A value in [0, 1.0], indicates the likelihood of the input sequence being a viral sequence. The larger the more likely to be a virus.

#### 2. `OUTPUT_PTH`/virus_`INPUT_FA_NAME`.fasta:

This FASTA file contains all the identified virus sequences that have virus_scores larger than `THRESHOLD`.

## 5. Citation

If you use ViraLM in your research, please kindly cite our paper:
```
@article{peng2024viralm,
  title={ViraLM: Empowering Virus Discovery through the Genome Foundation Model},
  author={Peng, Cheng and Shang, Jiayu and Guan, Jiaojiao and Wang, Donglin and Sun, Yanni},
  journal={Bioinformatics},
  pages={btae704},
  year={2024},
  publisher={Oxford University Press}
}
```

## 6. Contact

If you have any questions, please email us: cpeng29-c@my.cityu.edu.hk
